import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOCK_IN_METRIC_FIELDS: Tuple[str, ...] = (
    "atom_number_up", "atom_number_dw",
    "temperature_up", "temperature_dw",
    "sigma_up", "sigma_dw",
    "amplitude_up", "amplitude_dw",
    "tail_mean_up_raw", "tail_mean_dw_raw",
    "arrival_time_up", "arrival_time_dw",
    "transition_probability_up", "transition_probability_dw",
    "intf_n1", "intf_n2", "intf_p1", "intf_p2",
    "atom_number_up_nofit", "atom_number_dw_nofit",
    "temperature_up_nofit", "temperature_dw_nofit",
    "sigma_up_nofit", "sigma_dw_nofit",
    "amplitude_up_nofit", "amplitude_dw_nofit",
    "arrival_time_up_nofit", "arrival_time_dw_nofit",
    "transition_probability_up_nofit", "transition_probability_dw_nofit",
    "intf_n1_nofit", "intf_n2_nofit", "intf_p1_nofit", "intf_p2_nofit",
)

EXPECTED_STATES = ("a", "b", "b", "a")
EXPECTED_REFERENCES = (1, -1, -1, 1)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    count = len(finite)
    if not finite:
        return {"count": 0, "mean": None, "std": None, "sem": None}
    mean = sum(finite) / count
    if count < 2:
        return {"count": count, "mean": mean, "std": 0.0, "sem": 0.0}
    variance = sum((value - mean) ** 2 for value in finite) / (count - 1)
    std = math.sqrt(variance)
    return {"count": count, "mean": mean, "std": std, "sem": std / math.sqrt(count)}


def build_lock_in_analysis(
    points: Iterable[Dict[str, Any]],
    expected_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """Build complete ABBA blocks and aggregate digital lock-in statistics."""
    grouped: Dict[int, Dict[int, Dict[str, Any]]] = {}
    observed_blocks = set()
    for point in points:
        try:
            block_index = int(point.get("lock_in_block_index"))
            position = int(point.get("lock_in_position"))
        except (TypeError, ValueError):
            continue
        if block_index < 1 or position not in range(1, 5):
            continue
        observed_blocks.add(block_index)
        grouped.setdefault(block_index, {})[position] = point

    rows: List[Dict[str, Any]] = []
    for block_index in sorted(grouped):
        positions = grouped[block_index]
        if set(positions) != {1, 2, 3, 4}:
            continue
        ordered = [positions[index] for index in range(1, 5)]
        states = tuple(str(point.get("lock_in_state") or "").strip().lower() for point in ordered)
        references = tuple(int(point.get("lock_in_reference") or 0) for point in ordered)
        if states != EXPECTED_STATES or references != EXPECTED_REFERENCES:
            continue

        row: Dict[str, Any] = {
            "block_index": block_index,
            "shot_steps": [int(point.get("step") or point.get("current_step") or 0) for point in ordered],
            "a_value": _finite(ordered[0].get("parameter")),
            "b_value": _finite(ordered[1].get("parameter")),
        }
        for metric in LOCK_IN_METRIC_FIELDS:
            values = [_finite(point.get(metric)) for point in ordered]
            if any(value is None for value in values):
                for shot_index in range(1, 5):
                    row[f"{metric}_s{shot_index}"] = None
                row[f"{metric}_s_a"] = None
                row[f"{metric}_s_b"] = None
                row[f"{metric}_x"] = None
                row[f"{metric}_r"] = None
                row[f"{metric}_e"] = None
                row[f"{metric}_e_norm"] = None
                continue
            s1, s2, s3, s4 = (float(value) for value in values)
            for shot_index, shot_value in enumerate((s1, s2, s3, s4), start=1):
                row[f"{metric}_s{shot_index}"] = shot_value
            s_a = (s1 + s4) / 2.0
            s_b = (s2 + s3) / 2.0
            x_value = (s1 - s2 - s3 + s4) / 4.0
            r_denominator = s_a + s_b
            e_denominator = s_a + 2.0 * s_b
            row[f"{metric}_s_a"] = s_a
            row[f"{metric}_s_b"] = s_b
            row[f"{metric}_x"] = x_value
            row[f"{metric}_r"] = (s_a - s_b) / r_denominator if r_denominator != 0 else None
            row[f"{metric}_e"] = s_a - 2.0 * s_b
            row[f"{metric}_e_norm"] = (s_a - 2.0 * s_b) / e_denominator if e_denominator != 0 else None
        rows.append(row)

    metric_summary: Dict[str, Dict[str, Any]] = {}
    for metric in LOCK_IN_METRIC_FIELDS:
        summary: Dict[str, Any] = {}
        for quantity in ("x", "r", "e", "e_norm"):
            stats = _sample_stats(
                row[f"{metric}_{quantity}"]
                for row in rows
                if row.get(f"{metric}_{quantity}") is not None
            )
            for stat_name, stat_value in stats.items():
                summary[f"{quantity}_{stat_name}"] = stat_value
        metric_summary[metric] = summary

    expected = max(0, int(expected_blocks or 0))
    if expected == 0 and observed_blocks:
        expected = max(observed_blocks)
    return {
        "reference_sequence": list(EXPECTED_REFERENCES),
        "state_sequence": [state.upper() for state in EXPECTED_STATES],
        "expected_blocks": expected,
        "complete_blocks": len(rows),
        "incomplete_blocks": max(0, expected - len(rows)),
        "blocks": rows,
        "metrics": metric_summary,
    }
