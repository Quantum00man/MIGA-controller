import math
from typing import Any, Dict, Tuple

import numpy as np

import config


BRAGG_VOLTAGE_MIN = -15.0
BRAGG_VOLTAGE_MAX = 15.0
BRAGG_OFF_VOLTAGE = -3.0
BRAGG_MAX_POINTS = 2_000_000


def _calibration_values(calibration: Dict[str, Any] | None) -> Tuple[float, ...]:
    source = calibration or config.DEFAULT_BRAGG_POWER_CALIBRATION
    try:
        values = (
            float(source["lower_asymptote"]),
            float(source["upper_asymptote"]),
            float(source["growth_rate"]),
            float(source["midpoint"]),
            float(source["shape"]),
            float(source["linear_voltage_min"]),
            float(source["linear_voltage_max"]),
            float(source.get("off_threshold", 0.001)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Bragg power calibration is incomplete") from exc

    lower, upper, growth, _, shape, voltage_min, voltage_max, off_threshold = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Bragg power calibration contains a non-finite value")
    if upper <= lower:
        raise ValueError("Bragg calibration upper asymptote must exceed the lower asymptote")
    if growth <= 0 or shape <= 0:
        raise ValueError("Bragg calibration growth rate and shape must be positive")
    if not BRAGG_VOLTAGE_MIN <= voltage_min < voltage_max <= BRAGG_VOLTAGE_MAX:
        raise ValueError("Bragg linear voltage range must satisfy -15 <= min < max <= 15 V")
    if not 0 <= off_threshold < 1:
        raise ValueError("Bragg off threshold must be in the interval [0, 1)")
    return values


def _power_from_voltage(voltage: float, values: Tuple[float, ...]) -> float:
    lower, upper, growth, midpoint, shape, _, _, _ = values
    exponent = -growth * (float(voltage) - midpoint)
    if exponent > 0:
        softplus = exponent + math.log1p(math.exp(-exponent))
    else:
        softplus = math.log1p(math.exp(exponent))
    return lower + (upper - lower) * math.exp(-shape * softplus)


def normalized_power_to_voltage(
    normalized_power: float,
    calibration: Dict[str, Any] | None = None,
) -> float:
    """Invert the Bragg curve after mapping the selected linear interval to 0..1."""
    values = _calibration_values(calibration)
    lower, upper, growth, midpoint, shape, voltage_min, voltage_max, off_threshold = values
    requested = float(normalized_power)
    if not math.isfinite(requested):
        raise ValueError("Requested normalized Bragg power must be finite")
    if requested <= off_threshold:
        return BRAGG_OFF_VOLTAGE
    if requested > 1.0:
        raise ValueError("Requested normalized Bragg power must not exceed 1")

    power_min = _power_from_voltage(voltage_min, values)
    power_max = _power_from_voltage(voltage_max, values)
    if not power_max > power_min:
        raise ValueError("Bragg calibration is not increasing across the selected linear interval")
    target_power = power_min + requested * (power_max - power_min)

    fraction = (target_power - lower) / (upper - lower)
    if not 0 < fraction < 1:
        raise ValueError("Bragg target power is outside the invertible calibration interval")
    logarithm_argument = fraction ** (-1.0 / shape) - 1.0
    if logarithm_argument <= 0 or not math.isfinite(logarithm_argument):
        raise ValueError("Cannot invert the Bragg calibration at the requested power")
    voltage = midpoint - math.log(logarithm_argument) / growth
    return min(voltage_max, max(voltage_min, voltage))

def generate_bragg_pulse(
    fwhm: float,
    shape: str = "blackman",
    base_timing: int = 331119,
    clock_res: float = 0.2,
    calibration: Dict[str, Any] | None = None,
):
    """Generate channel-32 Bragg pulse code and its PARAMETER1 compensation."""
    fwhm = float(fwhm)
    clock_res = float(clock_res)
    if not math.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("Bragg FWHM must be a positive finite value")
    if not math.isfinite(clock_res) or clock_res <= 0:
        raise ValueError("Bragg clock resolution must be a positive finite value")
    _calibration_values(calibration)
    shape = str(shape).lower()

    if shape == "gaussian":
        std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))
        mean = 4.0 * std_dev
        x_end = 2.0 * mean
        num_points = max(1, int(x_end / clock_res))
    elif shape == "blackman":
        total_duration = fwhm / 0.405
        num_points = max(1, int(total_duration / clock_res))
    else:
        raise ValueError(f"Unsupported shape '{shape}'. Please choose 'gaussian' or 'blackman'.")

    pulse_logic_duration = (num_points + 2) * clock_res
    param1_compensation = float(base_timing) - pulse_logic_duration
    if param1_compensation < 0:
        raise ValueError(
            f"Bragg pulse duration {pulse_logic_duration:.1f} us exceeds base timing {base_timing} us"
        )
    if num_points > BRAGG_MAX_POINTS:
        raise ValueError(f"Bragg pulse contains more than {BRAGG_MAX_POINTS} points")

    if shape == "gaussian":
        x_values = np.linspace(0, x_end, num_points)
        ideal_shape = np.exp(-((x_values - mean) ** 2) / (2 * std_dev ** 2))
    else:
        ideal_shape = np.blackman(num_points)

    y_values = [normalized_power_to_voltage(value, calibration) for value in ideal_shape]
    pulse_name = f"{shape.capitalize()}_pulse"
    pulse_commands = [f"+500.0us {pulse_name} = {BRAGG_OFF_VOLTAGE:.3f}\t\t(32)"]
    pulse_commands.extend(
        f"+{clock_res:.1f}us {pulse_name} = {voltage:.3f}\t\t(32)"
        for voltage in y_values
    )
    pulse_commands.append(
        f"+{clock_res:.1f}us {pulse_name} = {BRAGG_OFF_VOLTAGE:.3f}\t\t(32)"
    )
    return "\n".join(pulse_commands), f"{param1_compensation:.1f}"
