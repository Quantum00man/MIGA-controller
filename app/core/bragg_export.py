from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Tuple
import zipfile

from app.core.pulse_generator import generate_bragg_pulse
from app.drivers.hardware import SequenceEditor


MAX_BRAGG_EXPORT_FILES = 200
MAX_BRAGG_EXPORT_BYTES = 100 * 1024 * 1024


def read_sequence_template(path: str | Path) -> str:
    template_path = Path(path)
    if not template_path.is_file():
        raise ValueError(f"Sequence template not found: {template_path}")
    payload = template_path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode sequence template: {template_path}")


def validate_bragg_template(template_content: str) -> None:
    if "<PARAMETER0>" not in template_content:
        raise ValueError("Bragg export template must contain <PARAMETER0>")


def format_fwhm_value(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError("Bragg FWHM values must be positive finite numbers")
    formatted = f"{numeric:.12f}".rstrip("0").rstrip(".")
    return formatted or "0"


def sequence_filename_stem(sequence_name: str) -> str:
    raw_name = Path(str(sequence_name or "").strip()).name
    if "seq0.mot" in raw_name.lower() and raw_name.lower().startswith("default"):
        raw_name = "seq0.mot"
    stem = Path(raw_name).stem if Path(raw_name).suffix.lower() == ".mot" else raw_name
    sanitized = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._-")
    return sanitized or "seq0"


def normalize_fwhm_values(values: Iterable[float]) -> List[Tuple[float, str]]:
    normalized: List[Tuple[float, str]] = []
    seen_labels = set()
    for value in values:
        numeric = float(value)
        label = format_fwhm_value(numeric)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        normalized.append((numeric, label))
        if len(normalized) > MAX_BRAGG_EXPORT_FILES:
            raise ValueError(f"Bragg ZIP export is limited to {MAX_BRAGG_EXPORT_FILES} files")
    if not normalized:
        raise ValueError("Bragg export contains no FWHM values")
    return normalized


def render_bragg_mot(
    template_content: str,
    fwhm: float,
    shape: str,
    base_timing: int,
    calibration: Dict[str, Any],
) -> str:
    validate_bragg_template(template_content)
    pulse_code, compensation = generate_bragg_pulse(
        fwhm=fwhm,
        shape=shape,
        base_timing=base_timing,
        calibration=calibration,
    )
    return SequenceEditor.render_bragg_sequence_content(template_content, pulse_code, compensation)


def build_single_bragg_export(
    template_content: str,
    sequence_name: str,
    fwhm: float,
    shape: str,
    base_timing: int,
    calibration: Dict[str, Any],
) -> Tuple[bytes, str]:
    label = format_fwhm_value(fwhm)
    rendered = render_bragg_mot(template_content, fwhm, shape, base_timing, calibration)
    payload = rendered.encode("utf-8")
    if len(payload) > MAX_BRAGG_EXPORT_BYTES:
        raise ValueError("Generated Bragg MOT file exceeds the 100 MB export limit")
    return payload, f"{sequence_filename_stem(sequence_name)}_{label}us.mot"


def build_bragg_zip_export(
    template_content: str,
    sequence_name: str,
    fwhm_values: Iterable[float],
    shape: str,
    base_timing: int,
    calibration: Dict[str, Any],
) -> Tuple[bytes, str]:
    normalized_values = normalize_fwhm_values(fwhm_values)
    stem = sequence_filename_stem(sequence_name)
    archive_buffer = BytesIO()
    total_uncompressed_bytes = 0
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for fwhm, label in normalized_values:
            rendered = render_bragg_mot(template_content, fwhm, shape, base_timing, calibration)
            payload = rendered.encode("utf-8")
            total_uncompressed_bytes += len(payload)
            if total_uncompressed_bytes > MAX_BRAGG_EXPORT_BYTES:
                raise ValueError("Generated Bragg ZIP content exceeds the 100 MB export limit")
            archive.writestr(f"{stem}_{label}us.mot", payload)
    first_label = normalized_values[0][1]
    last_label = normalized_values[-1][1]
    shape_label = str(shape or "blackman").strip().lower()
    filename = f"{stem}_{shape_label}_{first_label}-{last_label}us.zip"
    return archive_buffer.getvalue(), filename
