from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Tuple

from lxml import etree


DDS_MAX_AMPLITUDE = 1023
DDS_MAX_ELEMENT_NUMBER = 500
DDS_MAX_ELEMENT_COUNT = 500
DDS_FREQUENCY_HZ = "80000000"


class DdsTableError(ValueError):
    """Raised when an uploaded or generated DDS table is not safe to use."""


class DdsCommandError(RuntimeError):
    """Raised when writetable.py cannot write or verify a DDS table."""


@dataclass(frozen=True)
class DdsRatioPlan:
    ratio: float
    element: int
    requested_power_r1: float
    requested_power_r2: float
    amplitude_r1: int
    amplitude_r2: int
    actual_power_r1: float
    actual_power_r2: float
    actual_ratio: float
    actual_total_power: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_ratio_values(start: float, stop: float, step: float) -> List[float]:
    """Generate an inclusive, decimal-stable ratio axis."""
    try:
        start_value = Decimal(str(start))
        stop_value = Decimal(str(stop))
        step_value = abs(Decimal(str(step)))
    except (InvalidOperation, ValueError) as exc:
        raise DdsTableError("Ratio start, stop, and step must be finite numbers") from exc

    if not all(value.is_finite() for value in (start_value, stop_value, step_value)):
        raise DdsTableError("Ratio start, stop, and step must be finite numbers")
    if start_value <= 0 or stop_value <= 0:
        raise DdsTableError("R1:R2 ratios must be positive")
    if step_value <= 0:
        raise DdsTableError("Ratio step must be positive")

    direction = Decimal(1) if stop_value >= start_value else Decimal(-1)
    increment = step_value * direction
    values: List[float] = []
    current = start_value
    compare = (lambda value: value <= stop_value) if direction > 0 else (lambda value: value >= stop_value)
    while compare(current):
        values.append(float(current))
        if len(values) > DDS_MAX_ELEMENT_COUNT:
            raise DdsTableError("Ratio scan contains more than 500 points")
        current += increment

    if not values:
        values = [float(start_value)]
    return values


def _calibration_values(calibration: Dict[str, Any]) -> Tuple[float, float, float, float, float, int, int]:
    try:
        lower = float(calibration["lower_asymptote"])
        upper = float(calibration["upper_asymptote"])
        growth = float(calibration["growth_rate"])
        midpoint = float(calibration["midpoint"])
        shape = float(calibration["shape"])
        amplitude_min = int(calibration.get("amplitude_min", 0))
        amplitude_max = int(calibration.get("amplitude_max", DDS_MAX_AMPLITUDE))
    except (KeyError, TypeError, ValueError) as exc:
        raise DdsTableError("Raman power calibration is incomplete") from exc

    numeric = (lower, upper, growth, midpoint, shape)
    if not all(math.isfinite(value) for value in numeric):
        raise DdsTableError("Raman power calibration contains a non-finite value")
    if upper <= lower:
        raise DdsTableError("Calibration upper asymptote must exceed the lower asymptote")
    if growth <= 0 or shape <= 0:
        raise DdsTableError("Calibration growth rate and shape must be positive")
    if not 0 <= amplitude_min <= amplitude_max <= DDS_MAX_AMPLITUDE:
        raise DdsTableError("Calibration amplitude range must stay within 0..1023")
    return lower, upper, growth, midpoint, shape, amplitude_min, amplitude_max


def power_from_amplitude(amplitude: int | float, calibration: Dict[str, Any]) -> float:
    lower, upper, growth, midpoint, shape, amplitude_min, amplitude_max = _calibration_values(calibration)
    amplitude_value = float(amplitude)
    if not amplitude_min <= amplitude_value <= amplitude_max:
        raise DdsTableError(
            f"DDS amplitude {amplitude_value:g} is outside the calibrated range "
            f"{amplitude_min}..{amplitude_max}"
        )
    exponent = -growth * (amplitude_value - midpoint)
    if exponent > 0:
        softplus = exponent + math.log1p(math.exp(-exponent))
    else:
        softplus = math.log1p(math.exp(exponent))
    return lower + (upper - lower) * math.exp(-shape * softplus)


def amplitude_from_power(power: float, calibration: Dict[str, Any]) -> int:
    lower, upper, growth, midpoint, shape, amplitude_min, amplitude_max = _calibration_values(calibration)
    power_value = float(power)
    if not math.isfinite(power_value):
        raise DdsTableError("Requested Raman power must be finite")
    if not lower < power_value < upper:
        raise DdsTableError(
            f"Requested Raman power {power_value:.6g} is outside the invertible "
            f"calibration interval ({lower:.6g}, {upper:.6g})"
        )

    logarithm_argument = ((upper - lower) / (power_value - lower)) ** (1.0 / shape) - 1.0
    if logarithm_argument <= 0 or not math.isfinite(logarithm_argument):
        raise DdsTableError(f"Cannot invert the Raman calibration at power {power_value:.6g}")
    amplitude_value = midpoint - math.log(logarithm_argument) / growth
    amplitude = int(round(amplitude_value))
    if not 0 <= amplitude <= DDS_MAX_AMPLITUDE:
        raise DdsTableError(f"Calculated DDS amplitude {amplitude} is outside 0..1023")
    if not amplitude_min <= amplitude <= amplitude_max:
        raise DdsTableError(
            f"Calculated DDS amplitude {amplitude} is outside the calibrated range "
            f"{amplitude_min}..{amplitude_max}"
        )
    return amplitude


def _xml_parser() -> etree.XMLParser:
    return etree.XMLParser(resolve_entities=False, no_network=True, remove_blank_text=False)


def parse_dds_table(path: str | Path) -> Tuple[etree._ElementTree, List[int]]:
    table_path = Path(path)
    if not table_path.is_file():
        raise DdsTableError(f"DDS XML file not found: {table_path}")
    try:
        tree = etree.parse(str(table_path), parser=_xml_parser())
    except (OSError, etree.XMLSyntaxError) as exc:
        raise DdsTableError(f"Invalid DDS XML: {exc}") from exc

    root = tree.getroot()
    if root.tag != "ad9958":
        raise DdsTableError(f"DDS XML root must be <ad9958>, found <{root.tag}>")

    element_numbers: List[int] = []
    for element in root.findall("./elem"):
        raw_number = element.get("n")
        try:
            number = int(raw_number)
        except (TypeError, ValueError) as exc:
            raise DdsTableError(f"DDS element has invalid n={raw_number!r}") from exc
        if number < 0 or number > DDS_MAX_ELEMENT_NUMBER:
            raise DdsTableError(f"DDS element n={number} is outside 0..500")
        element_numbers.append(number)

    if len(element_numbers) > DDS_MAX_ELEMENT_COUNT:
        raise DdsTableError("DDS XML contains more than 500 elements")
    if len(set(element_numbers)) != len(element_numbers):
        raise DdsTableError("DDS XML contains duplicate element numbers")
    return tree, element_numbers


def validate_dds_table(path: str | Path) -> Dict[str, Any]:
    _, element_numbers = parse_dds_table(path)
    return {
        "element_count": len(element_numbers),
        "max_element": max(element_numbers, default=-1),
    }


def _append_fixed_element(root: etree._Element, number: int, amplitude_r1: int, amplitude_r2: int) -> None:
    element = etree.SubElement(root, "elem", n=str(number))
    for channel_name, amplitude in (("ch0", amplitude_r1), ("ch1", amplitude_r2)):
        channel = etree.SubElement(element, channel_name)
        etree.SubElement(channel, "mode").text = "sf"
        etree.SubElement(channel, "fr").text = DDS_FREQUENCY_HZ
        etree.SubElement(channel, "am").text = str(amplitude)


def build_ac_stark_table(
    source_path: str | Path,
    output_path: str | Path,
    ratios: Iterable[float],
    total_power: float,
    calibration_r1: Dict[str, Any],
    calibration_r2: Dict[str, Any],
) -> List[DdsRatioPlan]:
    tree, element_numbers = parse_dds_table(source_path)
    ratio_values = [float(value) for value in ratios]
    if not ratio_values:
        raise DdsTableError("AC Stark ratio scan is empty")

    total_power_value = float(total_power)
    if not math.isfinite(total_power_value) or total_power_value <= 0:
        raise DdsTableError("Total Raman power must be positive")
    if len(element_numbers) + len(ratio_values) > DDS_MAX_ELEMENT_COUNT:
        raise DdsTableError("Original and generated DDS elements would exceed the 500-element limit")

    first_new_element = max(element_numbers, default=-1) + 1
    last_new_element = first_new_element + len(ratio_values) - 1
    if last_new_element > DDS_MAX_ELEMENT_NUMBER:
        raise DdsTableError(
            f"Generated DDS element n={last_new_element} exceeds the maximum n=500"
        )

    plans: List[DdsRatioPlan] = []
    root = tree.getroot()
    for offset, ratio in enumerate(ratio_values):
        if not math.isfinite(ratio) or ratio <= 0:
            raise DdsTableError(f"Invalid R1:R2 ratio: {ratio}")
        requested_power_r1 = total_power_value * ratio / (1.0 + ratio)
        requested_power_r2 = total_power_value / (1.0 + ratio)
        amplitude_r1 = amplitude_from_power(requested_power_r1, calibration_r1)
        amplitude_r2 = amplitude_from_power(requested_power_r2, calibration_r2)
        actual_power_r1 = power_from_amplitude(amplitude_r1, calibration_r1)
        actual_power_r2 = power_from_amplitude(amplitude_r2, calibration_r2)
        if actual_power_r1 <= 0 or actual_power_r2 <= 0:
            raise DdsTableError(
                "Rounded DDS amplitudes must produce positive R1 and R2 optical powers"
            )
        actual_total = actual_power_r1 + actual_power_r2
        actual_ratio = actual_power_r1 / actual_power_r2
        element_number = first_new_element + offset
        _append_fixed_element(root, element_number, amplitude_r1, amplitude_r2)
        plans.append(
            DdsRatioPlan(
                ratio=ratio,
                element=element_number,
                requested_power_r1=requested_power_r1,
                requested_power_r2=requested_power_r2,
                amplitude_r1=amplitude_r1,
                amplitude_r2=amplitude_r2,
                actual_power_r1=actual_power_r1,
                actual_power_r2=actual_power_r2,
                actual_ratio=actual_ratio,
                actual_total_power=actual_total,
            )
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(destination),
        xml_declaration=True,
        encoding="ISO-8859-1",
        pretty_print=True,
    )
    validate_dds_table(destination)
    return plans


def _run_writetable(writer_path: str | Path, xml_path: str | Path, flag: str, timeout_sec: float) -> str:
    writer = Path(writer_path).expanduser().resolve()
    table = Path(xml_path).resolve()
    if not writer.is_file():
        raise DdsCommandError(f"writetable.py not found: {writer}")
    if not table.is_file():
        raise DdsCommandError(f"DDS XML file not found: {table}")
    if flag not in {"-w", "-v"}:
        raise DdsCommandError(f"Unsupported writetable operation: {flag}")

    command = ["python3", str(writer), flag, str(table)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(writer.parent),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1.0, float(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        raise DdsCommandError(f"DDS command timed out: {' '.join(command)}") from exc
    except OSError as exc:
        raise DdsCommandError(f"Could not start DDS command: {exc}") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    output = "\n".join(part for part in (stdout, stderr) if part)
    if completed.returncode != 0:
        raise DdsCommandError(
            f"DDS command failed with exit code {completed.returncode}: {' '.join(command)}"
            + (f"\n{output}" if output else "")
        )
    return output


def write_and_verify_dds_table(
    writer_path: str | Path,
    xml_path: str | Path,
    timeout_sec: float = 180.0,
) -> Dict[str, str]:
    write_output = _run_writetable(writer_path, xml_path, "-w", timeout_sec)
    verify_output = _run_writetable(writer_path, xml_path, "-v", timeout_sec)
    return {"write_output": write_output, "verify_output": verify_output}
