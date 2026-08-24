from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MARKER_RE = re.compile(r"^\s*###(SCAN|STATE|COMP):([A-Za-z0-9_ -]+)###\s*$", re.IGNORECASE)
MARKER_DEFINITION_RE = re.compile(r"^\s*#@MIGA_MARKER_DEF(?:\s+(?P<payload>.*?))?\s*$")
MARKER_DEFINITION_VERSION = 1
DURATION_RE = re.compile(
    r"^(?P<prefix>\s*)\+(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>us|ms|s)\b",
    re.IGNORECASE,
)
DDS_RE = re.compile(r"\bDDS[A-Za-z0-9_]*\s*\[\s*(?P<value>[-+]?\d+)\s*\]", re.IGNORECASE)
DAC_RE = re.compile(r"=\s*(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))(?=\s|\(|#|$)")
DIGITAL_RE = re.compile(r"=\s*(?P<value>ON|OFF)(?=\s|\(|#|$)", re.IGNORECASE)
COMMAND_RE = re.compile(
    r"^\s*\+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:us|ms|s)\s+(?P<command>[A-Za-z0-9_]+)",
    re.IGNORECASE,
)
CHANNEL_RE = re.compile(r"\((?P<channel>\d+)\)")

MARKER_KINDS = {"dds_element", "dac_value", "duration", "digital_state"}


def normalize_marker_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").upper()
    if not normalized:
        raise ValueError("Marker name is required")
    return normalized


def marked_filename(filename: str) -> str:
    original = Path(str(filename or "sequence.mot").strip()).name or "sequence.mot"
    path = Path(original)
    stem = path.stem
    if not stem.lower().endswith("_marked"):
        stem += "_marked"
    return f"{stem}.mot"



def sequence_marker_profile_key(filename: str) -> str:
    name = Path(str(filename or "sequence.mot").strip()).name or "sequence.mot"
    stem = Path(name).stem
    if stem.lower().endswith("_marked"):
        stem = stem[:-7]
    normalized = re.sub(r"\s+", " ", stem.strip()).casefold()
    return normalized or "sequence"


def marker_definitions_for_sequence(
    settings: Dict[str, Any],
    filename: str,
) -> List[Dict[str, Any]]:
    profiles = settings.get("sequence_marker_profiles")
    profile_key = sequence_marker_profile_key(filename)
    if isinstance(profiles, dict) and profile_key in profiles:
        return normalize_marker_definitions(profiles.get(profile_key), strict=False)
    return normalize_marker_definitions(
        settings.get("sequence_marker_definitions"), strict=False
    )

def decode_mot_bytes(payload: bytes) -> Tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode MOT file")


def encode_mot_text(content: str, encoding: str) -> bytes:
    normalized = str(encoding or "utf-8").lower()
    if normalized not in {"utf-8-sig", "utf-8", "latin-1"}:
        normalized = "utf-8"
    try:
        return str(content).encode(normalized)
    except UnicodeEncodeError as exc:
        raise ValueError(f"Marked MOT cannot be encoded as {normalized}: {exc}") from exc


def _line_ending(content: str) -> str:
    if "\r\n" in content:
        return "\r\n"
    if "\r" in content:
        return "\r"
    return "\n"


def _is_executable_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _to_microseconds(value: float, unit: str) -> float:
    factor = {"us": 1.0, "ms": 1000.0, "s": 1_000_000.0}.get(str(unit).lower())
    if factor is None:
        raise ValueError(f"Unsupported time unit: {unit}")
    return float(value) * factor


def _format_number(value: float, decimals: int) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Marker value must be finite")
    if decimals <= 0:
        rounded = round(numeric)
        if not math.isclose(numeric, rounded, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"Marker value {numeric:g} must be an integer")
        return str(int(rounded))
    return f"{numeric:.{decimals}f}"


def _command_and_channel(line: str) -> Tuple[str, str]:
    code = line.split("#", 1)[0]
    command_match = COMMAND_RE.search(code)
    channel_matches = list(CHANNEL_RE.finditer(code))
    command = command_match.group("command") if command_match else ""
    channel = channel_matches[-1].group("channel") if channel_matches else ""
    return command, channel


def detect_line_candidates(line: str, line_number: int) -> List[Dict[str, Any]]:
    if not _is_executable_line(line):
        return []
    code = line.split("#", 1)[0]
    command, channel = _command_and_channel(line)
    candidates: List[Dict[str, Any]] = []

    digital_match = DIGITAL_RE.search(code)
    if digital_match:
        state = digital_match.group("value").upper()
        candidates.append({
            "candidate_id": f"{line_number}:digital_state",
            "line_number": line_number,
            "kind": "digital_state",
            "value": state,
            "unit": "state",
            "command": command,
            "channel": channel,
            "label": f"Digital state {state}",
        })
    duration_match = DURATION_RE.search(code)
    if duration_match:
        value = float(duration_match.group("value"))
        unit = duration_match.group("unit").lower()
        candidates.append({
            "candidate_id": f"{line_number}:duration",
            "line_number": line_number,
            "kind": "duration",
            "value": value,
            "value_us": _to_microseconds(value, unit),
            "unit": unit,
            "command": command,
            "channel": channel,
            "label": f"Duration +{duration_match.group('value')}{unit}",
        })

    dds_match = DDS_RE.search(code)
    if dds_match:
        value = int(dds_match.group("value"))
        candidates.append({
            "candidate_id": f"{line_number}:dds_element",
            "line_number": line_number,
            "kind": "dds_element",
            "value": value,
            "unit": "element",
            "command": command,
            "channel": channel,
            "label": f"DDS element [{value}]",
        })

    dac_match = DAC_RE.search(code)
    if dac_match:
        value = float(dac_match.group("value"))
        candidates.append({
            "candidate_id": f"{line_number}:dac_value",
            "line_number": line_number,
            "kind": "dac_value",
            "value": value,
            "unit": "value",
            "command": command,
            "channel": channel,
            "label": f"DAC value {dac_match.group('value')}",
        })
    return candidates


def normalize_marker_definitions(
    definitions: Optional[Iterable[Dict[str, Any]]],
    *,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for raw in definitions or []:
        if not isinstance(raw, dict):
            if strict:
                raise ValueError("Marker definitions must be objects")
            continue
        try:
            marker_id = normalize_marker_id(raw.get("id") or raw.get("display_name"))
            if marker_id in seen:
                raise ValueError(f"Duplicate marker definition: {marker_id}")
            kind = str(raw.get("kind") or "").strip().lower()
            if kind not in MARKER_KINDS:
                raise ValueError(f"Unsupported marker type for {marker_id}: {kind}")
            if kind == "digital_state":
                decimals = 0
                hard_min, hard_max = 0.0, 1.0
                default_start, default_stop, default_step = 0.0, 1.0, 1.0
                default_method = "step_size"
            else:
                hard_min = float(raw.get("hard_min"))
                hard_max = float(raw.get("hard_max"))
                if not math.isfinite(hard_min) or not math.isfinite(hard_max) or hard_min > hard_max:
                    raise ValueError(f"Invalid hard limits for {marker_id}")
                decimals = int(raw.get("decimals", 3 if kind == "dac_value" else 0))
                if decimals < 0 or decimals > 9:
                    raise ValueError(f"Decimals for {marker_id} must be between 0 and 9")
                default_method = str(raw.get("default_method") or "step_size").strip().lower()
                if default_method not in {"step_size", "n_points"}:
                    raise ValueError(f"Invalid default scan method for {marker_id}")
                default_start = float(raw.get("default_start"))
                default_stop = float(raw.get("default_stop"))
                default_step = float(raw.get("default_step"))
                for default_value in (default_start, default_stop):
                    if not math.isfinite(default_value) or default_value < hard_min or default_value > hard_max:
                        raise ValueError(f"Default scan for {marker_id} exceeds hard limits")
                if not math.isfinite(default_step) or default_step <= 0:
                    raise ValueError(f"Default step/count for {marker_id} must be positive")
                if default_method == "n_points":
                    _format_number(default_step, 0)
                if kind in {"dds_element", "duration"}:
                    for value in (hard_min, hard_max, default_start, default_stop, default_step):
                        _format_number(value, 0)
                    decimals = 0
            normalized.append({
                "id": marker_id,
                "display_name": str(raw.get("display_name") or marker_id.replace("_", " ")).strip(),
                "kind": kind,
                "decimals": decimals,
                "hard_min": hard_min,
                "hard_max": hard_max,
                "default_start": default_start,
                "default_stop": default_stop,
                "default_step": default_step,
                "default_method": default_method,
                "expected_command": str(raw.get("expected_command") or "").strip(),
                "expected_channel": str(raw.get("expected_channel") or "").strip(),
                "has_compensation": bool(raw.get("has_compensation", False)) if kind == "duration" else False,
            })
            seen.add(marker_id)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(str(exc)) from exc
    return normalized

def normalize_marker_profiles(
    profiles: Any,
    *,
    strict: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    if profiles is None:
        return {}
    if not isinstance(profiles, dict):
        if strict:
            raise ValueError("Sequence marker profiles must be an object")
        return {}
    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for raw_key, definitions in profiles.items():
        profile_key = sequence_marker_profile_key(str(raw_key))
        if profile_key in normalized:
            if strict:
                raise ValueError(f"Duplicate sequence marker profile: {profile_key}")
            continue
        normalized[profile_key] = normalize_marker_definitions(
            definitions,
            strict=strict,
        )
    return normalized


def _definition_signature(definition: Dict[str, Any]) -> str:
    return json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_embedded_marker_definitions(
    content: str,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    lines = str(content).splitlines()
    definitions: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    seen = set()
    for index, line in enumerate(lines, start=1):
        match = MARKER_DEFINITION_RE.match(line)
        if not match:
            continue
        try:
            raw = json.loads(match.group("payload"))
            if not isinstance(raw, dict):
                raise ValueError("metadata payload must be a JSON object")
            version = int(raw.get("v", MARKER_DEFINITION_VERSION))
            if version != MARKER_DEFINITION_VERSION:
                raise ValueError(f"unsupported metadata version {version}")
            definition = normalize_marker_definitions([raw], strict=True)[0]
            marker_id = definition["id"]
            previous_index = index - 1
            while previous_index >= 1 and not lines[previous_index - 1].strip():
                previous_index -= 1
            marker_match = MARKER_RE.match(lines[previous_index - 1]) if previous_index >= 1 else None
            if not marker_match or marker_match.group(1).upper() not in {"SCAN", "STATE"}:
                raise ValueError("metadata must immediately follow a SCAN or STATE marker")
            owner_id = normalize_marker_id(marker_match.group(2))
            if owner_id != marker_id:
                raise ValueError(f"metadata ID {marker_id} does not match marker {owner_id}")
            if marker_id in seen:
                raise ValueError(f"duplicate embedded definition for {marker_id}")
            seen.add(marker_id)
            definitions.append(definition)
            records.append({
                "id": marker_id,
                "line_number": index,
                "marker_line_number": previous_index,
                "definition": definition,
            })
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            error = {"line_number": index, "message": str(exc), "source": line}
            errors.append(error)
            if strict:
                raise ValueError(f"Embedded Marker definition at line {index}: {exc}") from exc
    return {"definitions": definitions, "records": records, "errors": errors}


def definitions_with_embedded(
    content: str,
    fallback_definitions: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    fallback = normalize_marker_definitions(fallback_definitions, strict=False)
    embedded_result = extract_embedded_marker_definitions(content, strict=False)
    embedded = embedded_result["definitions"]
    fallback_map = {item["id"]: item for item in fallback}
    embedded_map = {item["id"]: item for item in embedded}
    conflicts = []
    for marker_id in sorted(set(fallback_map) & set(embedded_map)):
        if _definition_signature(fallback_map[marker_id]) != _definition_signature(embedded_map[marker_id]):
            conflicts.append({
                "id": marker_id,
                "message": "Embedded MOT definition differs from Settings; embedded definition is active",
                "embedded": embedded_map[marker_id],
                "settings": fallback_map[marker_id],
            })
    merged = dict(fallback_map)
    merged.update(embedded_map)
    return {
        "definitions": list(merged.values()),
        "embedded_definitions": embedded,
        "embedded_ids": sorted(embedded_map),
        "definition_sources": {
            marker_id: ("embedded" if marker_id in embedded_map else "settings")
            for marker_id in merged
        },
        "conflicts": conflicts,
        "errors": embedded_result["errors"],
        "records": embedded_result["records"],
    }


def resolve_marker_definitions(
    content: str,
    settings: Dict[str, Any],
    filename: str,
) -> Dict[str, Any]:
    fallback = marker_definitions_for_sequence(settings, filename)
    resolved = definitions_with_embedded(content, fallback)
    resolved["profile_key"] = sequence_marker_profile_key(filename)
    resolved["fallback_definitions"] = fallback
    return resolved


def find_matching_marker_definition_suggestions(
    content: str,
    settings: Dict[str, Any],
    filename: str,
) -> Dict[str, Any]:
    resolution = resolve_marker_definitions(content, settings, filename)
    inspection = inspect_sequence_markers(content, resolution["definitions"])
    resolved_ids = {item["id"] for item in resolution["definitions"]}
    candidates_by_id: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}

    sources: List[Tuple[str, Iterable[Dict[str, Any]]]] = []
    profiles = settings.get("sequence_marker_profiles")
    if isinstance(profiles, dict):
        sources.extend((str(profile), definitions) for profile, definitions in profiles.items())
    sources.append(("legacy", settings.get("sequence_marker_definitions") or []))
    for source_name, raw_definitions in sources:
        for definition in normalize_marker_definitions(raw_definitions, strict=False):
            candidates_by_id.setdefault(definition["id"], []).append((definition, source_name))

    suggestions = []
    ambiguities = []
    for marker in inspection["markers"]:
        if marker["role"] not in {"scan", "state"} or marker["id"] in resolved_ids:
            continue
        target = marker.get("candidate") or {}
        compatible: Dict[str, Dict[str, Any]] = {}
        source_names: Dict[str, List[str]] = {}
        for definition, source_name in candidates_by_id.get(marker["id"], []):
            if definition["kind"] != marker["kind"]:
                continue
            expected_command = definition.get("expected_command") or ""
            expected_channel = definition.get("expected_channel") or ""
            if expected_command and expected_command != target.get("command"):
                continue
            if expected_channel and expected_channel != target.get("channel"):
                continue
            signature = _definition_signature(definition)
            compatible[signature] = definition
            source_names.setdefault(signature, []).append(source_name)
        if len(compatible) == 1:
            signature, definition = next(iter(compatible.items()))
            suggestions.append({
                "id": marker["id"],
                "definition": definition,
                "source_profiles": sorted(set(source_names[signature])),
            })
        elif len(compatible) > 1:
            ambiguities.append({
                "id": marker["id"],
                "message": "Multiple compatible profiles contain different definitions",
                "options": [
                    {"definition": definition, "source_profiles": sorted(set(source_names[signature]))}
                    for signature, definition in compatible.items()
                ],
            })
    return {"suggestions": suggestions, "ambiguities": ambiguities}


def serialize_embedded_marker_definition(definition: Dict[str, Any]) -> str:
    normalized = normalize_marker_definitions([definition], strict=True)[0]
    payload = {"v": MARKER_DEFINITION_VERSION, **normalized}
    return "#@MIGA_MARKER_DEF " + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )



def embed_marker_definitions(
    content: str,
    definitions: Iterable[Dict[str, Any]],
    *,
    require_complete: bool = True,
) -> str:
    normalized = normalize_marker_definitions(definitions, strict=True)
    definition_map = {item["id"]: item for item in normalized}
    raw_lines = str(content).splitlines(keepends=True)
    cleaned_lines = [
        line for line in raw_lines
        if not MARKER_DEFINITION_RE.match(line.rstrip("\r\n"))
    ]
    owner_ids = []
    for line in cleaned_lines:
        match = MARKER_RE.match(line.rstrip("\r\n"))
        if match and match.group(1).upper() in {"SCAN", "STATE"}:
            owner_ids.append(normalize_marker_id(match.group(2)))
    missing = sorted(set(owner_ids) - set(definition_map))
    if require_complete and missing:
        raise ValueError(
            "Cannot make MOT self-contained; definitions are missing for: " + ", ".join(missing)
        )
    newline = _line_ending(content)
    output: List[str] = []
    for line in cleaned_lines:
        output.append(line)
        match = MARKER_RE.match(line.rstrip("\r\n"))
        if not match or match.group(1).upper() not in {"SCAN", "STATE"}:
            continue
        marker_id = normalize_marker_id(match.group(2))
        definition = definition_map.get(marker_id)
        if not definition:
            continue
        if not line.endswith(("\n", "\r")):
            output[-1] += newline
        output.append(serialize_embedded_marker_definition(definition) + newline)
    return "".join(output)


def inspect_sequence_markers(
    content: str,
    definitions: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    lines = str(content).splitlines(keepends=True)
    resolution = definitions_with_embedded(content, definitions)
    definition_list = resolution["definitions"]
    definition_map = {item["id"]: item for item in definition_list}
    embedded_ids = set(resolution["embedded_ids"])
    candidates: List[Dict[str, Any]] = []
    candidates_by_line: Dict[int, List[Dict[str, Any]]] = {}
    for index, line in enumerate(lines, start=1):
        detected = detect_line_candidates(line, index)
        if detected:
            candidates.extend(detected)
            candidates_by_line[index] = detected

    markers: List[Dict[str, Any]] = []
    occurrence_counts: Dict[Tuple[str, str], int] = {}
    for index, line in enumerate(lines, start=1):
        match = MARKER_RE.match(line.rstrip("\r\n"))
        if not match:
            continue
        role = match.group(1).upper()
        marker_id = normalize_marker_id(match.group(2))
        target_line = None
        for target_index in range(index + 1, len(lines) + 1):
            if _is_executable_line(lines[target_index - 1]):
                target_line = target_index
                break
        key = (role, marker_id)
        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
        definition = definition_map.get(marker_id)
        target_candidates = candidates_by_line.get(target_line or -1, [])
        if role == "COMP":
            inferred_kind = "duration"
        elif definition and role in {"SCAN", "STATE"}:
            inferred_kind = definition["kind"]
        elif role == "STATE":
            inferred_kind = "digital_state"
        elif any(item["kind"] == "dds_element" for item in target_candidates):
            inferred_kind = "dds_element"
        elif any(item["kind"] == "dac_value" for item in target_candidates):
            inferred_kind = "dac_value"
        else:
            inferred_kind = "duration"
        target_candidate = next(
            (item for item in target_candidates if item["kind"] == inferred_kind),
            None,
        )
        status = "defined" if definition else "undefined"
        message = ""
        if target_line is None:
            status, message = "invalid", "No executable instruction follows this marker"
        elif target_candidate is None:
            status, message = "conflict", f"Target line has no {inferred_kind} value"
        elif definition and role in {"SCAN", "STATE"}:
            expected_command = definition.get("expected_command") or ""
            expected_channel = definition.get("expected_channel") or ""
            if expected_command and target_candidate.get("command") != expected_command:
                status, message = "conflict", f"Expected command {expected_command}"
            elif expected_channel and target_candidate.get("channel") != expected_channel:
                status, message = "conflict", f"Expected channel ({expected_channel})"
            elif role == "STATE" and definition["kind"] != "digital_state":
                status, message = "conflict", "STATE marker requires a digital_state definition"
            elif role == "SCAN" and definition["kind"] == "digital_state":
                status, message = "conflict", "Digital state markers must use STATE"
        markers.append({
            "id": marker_id,
            "role": role.lower(),
            "marker_line_number": index,
            "target_line_number": target_line,
            "target_source": lines[target_line - 1].rstrip("\r\n") if target_line else "",
            "kind": inferred_kind,
            "candidate": target_candidate,
            "definition": definition,
            "definition_source": "embedded" if marker_id in embedded_ids else ("settings" if definition else ""),
            "status": status,
            "message": message,
        })

    for marker in markers:
        if occurrence_counts[(marker["role"].upper(), marker["id"])] > 1:
            marker["status"] = "conflict"
            marker["message"] = f"Duplicate {marker['role'].upper()} marker"

    scan_ids = {marker["id"] for marker in markers if marker["role"] == "scan"}
    for marker in markers:
        if marker["role"] == "comp" and marker["id"] not in scan_ids:
            marker["status"] = "conflict"
            marker["message"] = "Compensation marker has no matching scan marker"

    compensation_map = {
        marker["id"]: marker for marker in markers if marker["role"] == "comp"
    }
    for marker in markers:
        if marker["role"] != "scan" or not marker.get("definition"):
            continue
        compensation = compensation_map.get(marker["id"])
        if marker["definition"].get("has_compensation"):
            if compensation is None:
                marker["status"] = "conflict"
                marker["message"] = "Settings requires a compensation marker"
            elif compensation["status"] != "defined":
                marker["status"] = "conflict"
                marker["message"] = "Compensation marker is not valid"
        elif compensation is not None:
            marker["status"] = "conflict"
            marker["message"] = "File has compensation but Settings compensation is disabled"

    marker_target_lines = {marker["target_line_number"] for marker in markers if marker["target_line_number"]}
    line_records = []
    for line_number, line_candidates in candidates_by_line.items():
        line_records.append({
            "line_number": line_number,
            "source": lines[line_number - 1].rstrip("\r\n"),
            "candidates": line_candidates,
            "marked": line_number in marker_target_lines,
            "markers": [
                marker for marker in markers
                if marker.get("target_line_number") == line_number
            ],
        })
    return {
        "line_count": len(lines),
        "marker_count": len(markers),
        "markers": markers,
        "lines": line_records,
        "embedded_definition_ids": sorted(embedded_ids),
        "embedded_definition_errors": resolution["errors"],
        "definition_conflicts": resolution["conflicts"],
    }


def add_sequence_marker(
    content: str,
    marker_id: str,
    target_line_number: int,
    kind: str,
    compensation_line_number: Optional[int] = None,
) -> str:
    marker_id = normalize_marker_id(marker_id)
    kind = str(kind or "").strip().lower()
    if kind not in MARKER_KINDS:
        raise ValueError(f"Unsupported marker type: {kind}")
    inspection = inspect_sequence_markers(content)
    if any(marker["id"] == marker_id for marker in inspection["markers"]):
        raise ValueError(f"Marker {marker_id} already exists in this file")
    candidate_map = {
        item["candidate_id"]: item
        for record in inspection["lines"]
        for item in record["candidates"]
    }
    target_key = f"{int(target_line_number)}:{kind}"
    if target_key not in candidate_map:
        raise ValueError("Selected target no longer matches the requested marker type")
    if any(record["line_number"] == int(target_line_number) and record["marked"] for record in inspection["lines"]):
        raise ValueError("Selected instruction already has a marker")
    owner_role = "STATE" if kind == "digital_state" else "SCAN"
    insertions = [(int(target_line_number) - 1, f"###{owner_role}:{marker_id}###")]
    if compensation_line_number is not None:
        if kind != "duration":
            raise ValueError("Only duration markers can use compensation")
        compensation_line_number = int(compensation_line_number)
        if compensation_line_number == int(target_line_number):
            raise ValueError("Duration and compensation must use different instructions")
        if f"{compensation_line_number}:duration" not in candidate_map:
            raise ValueError("Selected compensation is not a duration instruction")
        compensation_record = next(
            (record for record in inspection["lines"] if record["line_number"] == compensation_line_number),
            None,
        )
        if compensation_record and any(
            marker["role"] in {"scan", "state"} and marker.get("kind") == "duration"
            for marker in compensation_record.get("markers", [])
        ):
            raise ValueError("Selected duration is already used by another scan marker")
        insertions.append((compensation_line_number - 1, f"###COMP:{marker_id}###"))

    lines = str(content).splitlines(keepends=True)
    newline = _line_ending(content)
    for index, marker_line in sorted(insertions, key=lambda item: item[0], reverse=True):
        lines.insert(index, marker_line + newline)
    return "".join(lines)


def update_sequence_marker(
    content: str,
    old_marker_id: str,
    new_marker_id: str,
    target_line_number: int,
    kind: str,
    compensation_line_number: Optional[int] = None,
) -> str:
    old_marker_id = normalize_marker_id(old_marker_id)
    new_marker_id = normalize_marker_id(new_marker_id)
    inspection = inspect_sequence_markers(content)
    existing = [marker for marker in inspection["markers"] if marker["id"] == old_marker_id]
    if not any(marker["role"] in {"scan", "state"} for marker in existing):
        raise ValueError(f"Marker {old_marker_id} was not found")
    if new_marker_id != old_marker_id and any(
        marker["id"] == new_marker_id for marker in inspection["markers"]
    ):
        raise ValueError(f"Marker {new_marker_id} already exists in this file")

    lines = str(content).splitlines(keepends=True)
    embedded_record_lines = {
        record["line_number"]
        for record in extract_embedded_marker_definitions(content)["records"]
        if record["id"] == old_marker_id
    }
    marker_line_numbers = {
        index
        for index, line in enumerate(lines, start=1)
        if ((match := MARKER_RE.match(line.rstrip("\r\n")))
            and normalize_marker_id(match.group(2)) == old_marker_id)
        or index in embedded_record_lines
    }

    def map_after_removal(line_number: int) -> int:
        numeric = int(line_number)
        if numeric < 1 or numeric > len(lines):
            raise ValueError("Selected marker line is outside the sequence")
        return numeric - sum(1 for marker_line in marker_line_numbers if marker_line < numeric)

    mapped_target = map_after_removal(target_line_number)
    mapped_compensation = (
        map_after_removal(compensation_line_number)
        if compensation_line_number is not None
        else None
    )
    without_old_marker = remove_sequence_marker(content, old_marker_id)
    return add_sequence_marker(
        without_old_marker,
        new_marker_id,
        mapped_target,
        kind,
        mapped_compensation,
    )


def remove_sequence_marker(content: str, marker_id: str) -> str:
    marker_id = normalize_marker_id(marker_id)
    lines = str(content).splitlines(keepends=True)
    metadata_lines = {
        record["line_number"]
        for record in extract_embedded_marker_definitions(content)["records"]
        if record["id"] == marker_id
    }
    kept = []
    removed = 0
    for index, line in enumerate(lines, start=1):
        match = MARKER_RE.match(line.rstrip("\r\n"))
        if (match and normalize_marker_id(match.group(2)) == marker_id) or index in metadata_lines:
            removed += 1
            continue
        kept.append(line)
    if not removed:
        raise ValueError(f"Marker {marker_id} was not found")
    return "".join(kept)


def _definition_map(definitions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in normalize_marker_definitions(definitions, strict=True)}


def _replace_candidate(line: str, kind: str, value: Any, definition: Dict[str, Any]) -> str:
    code, separator, comment = line.partition("#")
    decimals = int(definition.get("decimals", 0))
    if kind == "dds_element":
        formatted = _format_number(value, 0)
        if not DDS_RE.search(code):
            raise ValueError("DDS marker target no longer contains an element index")
        code = DDS_RE.sub(
            lambda match: (
                match.group(0)[:match.start("value") - match.start(0)]
                + formatted
                + match.group(0)[match.end("value") - match.start(0):]
            ),
            code,
            count=1,
        )
    elif kind == "dac_value":
        formatted = _format_number(value, decimals)
        if not DAC_RE.search(code):
            raise ValueError("DAC marker target no longer contains a numeric value")
        code = DAC_RE.sub(lambda match: match.group(0).replace(match.group("value"), formatted, 1), code, count=1)
    elif kind == "duration":
        formatted = _format_number(value, 0)
        if not DURATION_RE.search(code):
            raise ValueError("Duration marker target no longer contains a leading time")
        code = DURATION_RE.sub(lambda match: f"{match.group('prefix')}+{formatted}us", code, count=1)
    elif kind == "digital_state":
        formatted = str(value or "").strip().upper()
        if formatted not in {"ON", "OFF"}:
            raise ValueError("Digital state must be ON or OFF")
        if not DIGITAL_RE.search(code):
            raise ValueError("Digital state marker target no longer contains ON or OFF")
        code = DIGITAL_RE.sub(
            lambda match: match.group(0).replace(match.group("value"), formatted, 1),
            code,
            count=1,
        )
    else:
        raise ValueError(f"Unsupported marker type: {kind}")
    return code + (separator + comment if separator else "")


def _validate_value(marker_id: str, value: float, definition: Dict[str, Any]) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Marker {marker_id} value must be finite")
    hard_min = float(definition["hard_min"])
    hard_max = float(definition["hard_max"])
    if numeric < hard_min or numeric > hard_max:
        raise ValueError(
            f"Marker {marker_id} value {numeric:g} is outside hard limits {hard_min:g} to {hard_max:g}"
        )
    if definition["kind"] in {"dds_element", "duration"}:
        _format_number(numeric, 0)
    return numeric


def render_auto_marker_sequence(
    content: str,
    marker_axes: Sequence[str],
    values: Sequence[float],
    definitions: Iterable[Dict[str, Any]],
) -> str:
    axes = [normalize_marker_id(value) for value in marker_axes]
    if not axes or len(axes) != len(values):
        raise ValueError("Auto Marker axes and scan values do not match")
    if len(set(axes)) != len(axes):
        raise ValueError("Auto Marker axes must be unique")
    definition_resolution = definitions_with_embedded(content, definitions)
    if definition_resolution["errors"]:
        first_error = definition_resolution["errors"][0]
        raise ValueError(
            f"Invalid embedded Marker definition at line {first_error['line_number']}: {first_error['message']}"
        )
    definition_map = _definition_map(definition_resolution["definitions"])
    inspection = inspect_sequence_markers(content, definition_map.values())
    scan_markers = {marker["id"]: marker for marker in inspection["markers"] if marker["role"] == "scan"}
    comp_markers = {marker["id"]: marker for marker in inspection["markers"] if marker["role"] == "comp"}
    lines = str(content).splitlines(keepends=True)
    replacements: Dict[int, List[Tuple[str, float, Dict[str, Any]]]] = {}
    compensation_adjustments: Dict[int, Dict[str, Any]] = {}

    for marker_id, raw_value in zip(axes, values):
        definition = definition_map.get(marker_id)
        if not definition:
            raise ValueError(f"Marker {marker_id} has no embedded or Settings definition")
        if definition["kind"] == "digital_state":
            raise ValueError(f"Digital state marker {marker_id} cannot be used as a scan axis")
        marker = scan_markers.get(marker_id)
        if not marker:
            raise ValueError(f"Marker {marker_id} was not found in the sequence")
        if marker["status"] != "defined":
            raise ValueError(f"Marker {marker_id} is {marker['status']}: {marker['message']}")
        value = _validate_value(marker_id, raw_value, definition)
        target_line = int(marker["target_line_number"])
        replacements.setdefault(target_line, []).append((definition["kind"], value, definition))

        compensation = comp_markers.get(marker_id)
        if definition.get("has_compensation"):
            if not compensation or compensation["status"] != "defined":
                raise ValueError(f"Duration marker {marker_id} requires a valid compensation marker")
            target_candidate = marker.get("candidate") or {}
            comp_candidate = compensation.get("candidate") or {}
            comp_line = int(compensation["target_line_number"])
            if comp_line == target_line or any(
                replacement_kind == "duration"
                for replacement_kind, _, _ in replacements.get(comp_line, [])
            ):
                raise ValueError(f"Compensation target for {marker_id} conflicts with another scan marker")
            comp_definition = {**definition, "kind": "duration", "decimals": 0}
            adjustment = compensation_adjustments.setdefault(comp_line, {
                "initial": float(comp_candidate.get("value_us")),
                "delta": 0.0,
                "definition": comp_definition,
                "marker_ids": [],
            })
            adjustment["delta"] += float(target_candidate.get("value_us")) - value
            adjustment["marker_ids"].append(marker_id)
        elif compensation:
            raise ValueError(f"Marker {marker_id} has a compensation marker but compensation is disabled by the active definition")

    for comp_line, adjustment in compensation_adjustments.items():
        if any(
            replacement_kind == "duration"
            for replacement_kind, _, _ in replacements.get(comp_line, [])
        ):
            marker_names = ", ".join(adjustment["marker_ids"])
            raise ValueError(f"Compensation target for {marker_names} conflicts with another scan marker")
        new_compensation = adjustment["initial"] + adjustment["delta"]
        if new_compensation <= 0:
            marker_names = ", ".join(adjustment["marker_ids"])
            raise ValueError(
                f"Markers {marker_names} produce compensation {new_compensation:g} us; compensation must be greater than 0"
            )
        replacements.setdefault(comp_line, []).append(
            ("duration", new_compensation, adjustment["definition"])
        )

    for line_number, line_replacements in replacements.items():
        updated = lines[line_number - 1]
        for kind, value, definition in line_replacements:
            updated = _replace_candidate(updated, kind, value, definition)
        lines[line_number - 1] = updated
    return "".join(lines)


def render_digital_marker_states(
    content: str,
    states: Dict[str, str],
    definitions: Iterable[Dict[str, Any]],
) -> str:
    if not states:
        return str(content)
    definition_resolution = definitions_with_embedded(content, definitions)
    if definition_resolution["errors"]:
        first_error = definition_resolution["errors"][0]
        raise ValueError(
            f"Invalid embedded Marker definition at line {first_error['line_number']}: {first_error['message']}"
        )
    definition_map = _definition_map(definition_resolution["definitions"])
    inspection = inspect_sequence_markers(content, definition_map.values())
    state_markers = {
        marker["id"]: marker
        for marker in inspection["markers"]
        if marker["role"] == "state"
    }
    normalized_states: Dict[str, str] = {}
    for raw_marker_id, raw_state in states.items():
        marker_id = normalize_marker_id(raw_marker_id)
        if marker_id in normalized_states:
            raise ValueError(f"Duplicate digital state marker: {marker_id}")
        state = str(raw_state or "").strip().upper()
        if state not in {"ON", "OFF"}:
            raise ValueError(f"Digital state for {marker_id} must be ON or OFF")
        normalized_states[marker_id] = state

    lines = str(content).splitlines(keepends=True)
    for marker_id, state in normalized_states.items():
        definition = definition_map.get(marker_id)
        marker = state_markers.get(marker_id)
        if not definition or definition.get("kind") != "digital_state":
            raise ValueError(f"State marker {marker_id} has no digital_state definition")
        if not marker:
            raise ValueError(f"State marker {marker_id} was not found in the sequence")
        if marker["status"] != "defined":
            raise ValueError(f"State marker {marker_id} is {marker['status']}: {marker['message']}")
        target_line = int(marker["target_line_number"])
        lines[target_line - 1] = _replace_candidate(
            lines[target_line - 1],
            "digital_state",
            state,
            definition,
        )
    return "".join(lines)


def validate_auto_marker_scan(
    content: str,
    marker_axes: Sequence[str],
    parameter_sets: Iterable[Sequence[float]],
    definitions: Iterable[Dict[str, Any]],
) -> None:
    checked = 0
    for parameter_set in parameter_sets:
        values = list(parameter_set) if isinstance(parameter_set, (list, tuple)) else [parameter_set]
        render_auto_marker_sequence(content, marker_axes, values, definitions)
        checked += 1
    if checked == 0:
        raise ValueError("Auto Marker scan contains no parameter points")
