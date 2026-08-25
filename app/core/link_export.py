from __future__ import annotations

from io import BytesIO
import math
import re
from typing import Any, Iterable, List, Sequence, Tuple
import zipfile

from app.core.bragg_export import sequence_filename_stem
from app.core.sequence_markers import encode_mot_text
from app.drivers.hardware import SequenceEditor


MAX_LINK_EXPORT_FILES = 200
MAX_LINK_EXPORT_BYTES = 100 * 1024 * 1024
_UNRESOLVED_PARAMETER = re.compile(r"<PARAMETER\d+>")


def format_link_parameter_value(value: Any) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Link export parameter values must be finite numbers")
    if numeric == 0:
        numeric = 0.0
    label = f"{numeric:.12f}".rstrip("0").rstrip(".")
    return label or "0"


def render_link_mot(template_content: str, parameters: Sequence[Any]) -> str:
    if not parameters:
        raise ValueError("Link export contains no parameters")
    rendered = SequenceEditor.render_sequence_content(template_content, list(parameters))
    unresolved = sorted(set(_UNRESOLVED_PARAMETER.findall(rendered)))
    if unresolved:
        raise ValueError(
            "Link formulas do not provide values for template placeholders: "
            + ", ".join(unresolved)
        )
    return rendered


def _normalize_parameter_sets(
    parameter_sets: Iterable[Sequence[Any]],
) -> List[Tuple[str, List[Any]]]:
    normalized: List[Tuple[str, List[Any]]] = []
    seen_p0_labels = set()
    for raw_parameters in parameter_sets:
        parameters = list(raw_parameters)
        if not parameters:
            continue
        p0_label = format_link_parameter_value(parameters[0])
        if p0_label in seen_p0_labels:
            continue
        seen_p0_labels.add(p0_label)
        normalized.append((p0_label, parameters))
        if len(normalized) > MAX_LINK_EXPORT_FILES:
            raise ValueError(f"Link ZIP export is limited to {MAX_LINK_EXPORT_FILES} files")
    if not normalized:
        raise ValueError("Link export contains no parameter sets")
    return normalized


def build_single_link_export(
    template_content: str,
    template_encoding: str,
    sequence_name: str,
    parameters: Sequence[Any],
) -> Tuple[bytes, str]:
    values = list(parameters)
    if not values:
        raise ValueError("Link export contains no parameters")
    p0_label = format_link_parameter_value(values[0])
    rendered = render_link_mot(template_content, values)
    payload = encode_mot_text(rendered, template_encoding)
    if len(payload) > MAX_LINK_EXPORT_BYTES:
        raise ValueError("Generated Link MOT file exceeds the 100 MB export limit")
    filename = f"{sequence_filename_stem(sequence_name)}_P0_{p0_label}.mot"
    return payload, filename


def build_link_zip_export(
    template_content: str,
    template_encoding: str,
    sequence_name: str,
    parameter_sets: Iterable[Sequence[Any]],
) -> Tuple[bytes, str]:
    normalized = _normalize_parameter_sets(parameter_sets)
    stem = sequence_filename_stem(sequence_name)
    archive_buffer = BytesIO()
    total_uncompressed_bytes = 0
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p0_label, parameters in normalized:
            rendered = render_link_mot(template_content, parameters)
            payload = encode_mot_text(rendered, template_encoding)
            total_uncompressed_bytes += len(payload)
            if total_uncompressed_bytes > MAX_LINK_EXPORT_BYTES:
                raise ValueError("Generated Link ZIP content exceeds the 100 MB export limit")
            archive.writestr(f"{stem}_P0_{p0_label}.mot", payload)
    first_label = normalized[0][0]
    last_label = normalized[-1][0]
    return archive_buffer.getvalue(), f"{stem}_link_P0_{first_label}-{last_label}.zip"
