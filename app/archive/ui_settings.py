"""Hardware-free, device-scoped scientific preferences stored as immutable versions."""
import math
import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.analysis import fitting, interferometer_phase
from app.archive.backup import atomic_json

class ServerSettings:
    def __init__(self, directory):
        self.directory = Path(directory)

    def _load_user_json_payload(self):
        pointer = self.directory / 'latest.json'
        if not pointer.exists():
            return {}
        version = json.loads(pointer.read_text())['id']
        if len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
            raise ValueError('Invalid settings version')
        return json.loads((self.directory / (version + '.json')).read_text())

    def _save_user_json_payload(self, payload):
        self.directory.mkdir(parents=True, exist_ok=True)
        version = uuid.uuid4().hex
        atomic_json(self.directory / (version + '.json'), payload)
        atomic_json(self.directory / 'latest.json', {'id': version})

    def _normalize_custom_scan_fit_models(self, models: List[Dict[str, Any]], strict: bool = False) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for idx, model in enumerate(models or []):
            try:
                norm = fitting.normalize_fit_model_definition(model, fallback_key=f"user_scan_fit_model_{idx + 1}")
                raw_key = str(norm.get('key') or '').strip()
                if raw_key.startswith('user_'):
                    base_key = raw_key
                else:
                    base_key = f"user_{fitting.sanitize_model_key(norm.get('label') or raw_key or f'user_scan_fit_model_{idx + 1}') }"
                key = base_key
                suffix = 1
                while key in seen:
                    suffix += 1
                    key = f"{base_key}_{suffix}"
                norm['key'] = key
                error = fitting.validate_fit_model_definition(norm)
                if error:
                    raise ValueError(error)
                normalized.append(norm)
                seen.add(key)
            except Exception as exc:
                if strict:
                    raise ValueError(f"Invalid custom scan fit model #{idx + 1}: {exc}")
                print(f"[User JSON] Skipping invalid custom scan fit model #{idx + 1}: {exc}")
        return normalized

    def get_custom_scan_fit_models(self) -> List[Dict[str, Any]]:
        payload = self._load_user_json_payload()
        raw_models = payload.get('scan_fit_models')
        if not isinstance(raw_models, list):
            raw_models = []
        return self._normalize_custom_scan_fit_models(raw_models, strict=False)

    def get_scan_fit_models(self) -> List[Dict[str, Any]]:
        return fitting.get_default_scan_fit_models() + self.get_custom_scan_fit_models()

    def save_custom_scan_fit_model(self, model_definition: Dict[str, Any], name: str) -> Dict[str, Any]:
        label = str(name or '').strip()
        if not label:
            raise ValueError('Model name is required')

        source_key = str((model_definition or {}).get('key') or '').strip()
        requested_key = source_key if source_key.startswith('user_') else label
        candidate = fitting.normalize_fit_model_definition(
            {
                **(model_definition or {}),
                'key': requested_key,
                'label': label,
            },
            fallback_key=label,
        )
        if not candidate['key'].startswith('user_'):
            candidate['key'] = f"user_{candidate['key']}"

        validation_error = fitting.validate_fit_model_definition(candidate)
        if validation_error:
            raise ValueError(f"Invalid model: {validation_error}")

        payload = self._load_user_json_payload()
        models = self.get_custom_scan_fit_models()
        replaced = False
        for idx, model in enumerate(models):
            if model.get('key') == candidate['key']:
                models[idx] = candidate
                replaced = True
                break
        if not replaced:
            models.append(candidate)

        payload['scan_fit_models'] = self._normalize_custom_scan_fit_models(models, strict=True)
        self._save_user_json_payload(payload)
        return candidate

    def _normalize_bragg_phase_calibration(self, calibration: Dict[str, Any]) -> Dict[str, Any]:
        source = dict(calibration or {})
        if source.get('model_key') not in {None, '', 'bragg_fringes'}:
            raise ValueError('Only Bragg fringe fits can be saved as phase calibrations')
        parameters = dict(source.get('parameter_values') or {})
        bragg = dict(source.get('bragg') or {})
        required = {
            'A': parameters.get('A'),
            'C': parameters.get('C'),
            'phi0': parameters.get('phi0'),
            'omega': bragg.get('angular_frequency_rad_per_us2'),
        }
        try:
            numeric = {key: float(value) for key, value in required.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError('Bragg phase calibration is missing required parameters') from exc
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError('Bragg phase calibration contains non-finite parameters')
        if numeric['A'] <= 0 or numeric['omega'] <= 0:
            raise ValueError('Bragg phase calibration requires positive A and angular frequency')
        try:
            fit_x = [float(value) for value in source.get('fit_x') or []]
            fit_y = [float(value) for value in source.get('fit_y') or []]
        except (TypeError, ValueError) as exc:
            raise ValueError('Bragg phase calibration curve must be numeric') from exc
        if len(fit_x) != len(fit_y) or len(fit_x) < 2:
            raise ValueError('Bragg phase calibration requires a fitted reference curve')
        if not all(math.isfinite(value) for value in fit_x + fit_y):
            raise ValueError('Bragg phase calibration curve contains non-finite values')
        try:
            fit_min = float(source.get('fit_min', min(fit_x)))
            fit_max = float(source.get('fit_max', max(fit_x)))
        except (TypeError, ValueError) as exc:
            raise ValueError('Bragg phase calibration range must be numeric') from exc
        if fit_min > fit_max:
            fit_min, fit_max = fit_max, fit_min
        source_metadata = dict(source.get('source') or {})
        metric_tab = str(source.get('metric_tab') or source_metadata.get('metric_tab') or 'intf').strip().lower()
        channel = 'dw' if str(source.get('channel') or source_metadata.get('channel') or 'up').strip().lower() == 'dw' else 'up'
        source_key = str(source.get('source_key') or source_metadata.get('source_key') or 'intf_p').strip()
        source_mode = str(source.get('source_mode') or source_metadata.get('source_mode') or '').strip().lower()
        if source_mode not in {'fit', 'raw'}:
            source_mode = 'raw' if source_key.startswith('nf_') or 'nofit' in source_key.lower() else 'fit'
        mid_fringe_x = []
        for value in bragg.get('mid_fringe_x') or []:
            try:
                numeric_mid = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_mid) and numeric_mid >= 0:
                mid_fringe_x.append(numeric_mid)
        default_reference = min(mid_fringe_x, key=lambda value: abs(value - (fit_min + fit_max) / 2.0)) if mid_fringe_x else max(0.0, (fit_min + fit_max) / 2.0)
        reference_t2 = source.get('reference_t2_us2')
        try:
            reference_t2 = float(reference_t2) if reference_t2 is not None else float(default_reference)
        except (TypeError, ValueError):
            reference_t2 = float(default_reference)
        if not math.isfinite(reference_t2) or reference_t2 < 0:
            reference_t2 = float(default_reference)
        try:
            reference_value = float(source.get('reference_value'))
        except (TypeError, ValueError):
            reference_value = reference_t2
        if not math.isfinite(reference_value) or reference_value < 0:
            reference_value = reference_t2
        monotonic_direction = str(source.get('monotonic_slope') or '').strip().lower()
        if monotonic_direction not in {'negative', 'positive'}:
            derivative = -numeric['A'] * numeric['omega'] * math.sin(numeric['omega'] * reference_t2 + numeric['phi0'])
            monotonic_direction = 'positive' if derivative > 0 else 'negative'
        return {
            'id': str(source.get('id') or f"bragg_{time.time_ns()}"),
            'name': str(source.get('name') or '').strip(),
            'created_at': str(source.get('created_at') or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
            'model_key': 'bragg_fringes',
            'model_label': 'Bragg Fringes',
            'source': source_metadata,
            'metric_tab': metric_tab,
            'metric_label': str(source.get('metric_label') or ''),
            'source_key': source_key,
            'source_label': str(source.get('source_label') or ''),
            'source_mode': source_mode,
            'source_field': interferometer_phase.source_field({
                'metric_tab': metric_tab, 'channel': channel,
                'source_key': source_key, 'source_mode': source_mode,
            }),
            'channel': channel,
            'channel_label': str(source.get('channel_label') or ''),
            'fit_min': fit_min,
            'fit_max': fit_max,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'parameter_values': {**parameters, **{key: numeric[key] for key in ('A', 'C', 'phi0')}},
            'bragg': {**bragg, 'angular_frequency_rad_per_us2': numeric['omega'], 'mid_fringe_x': mid_fringe_x},
            'reference_input_mode': 't' if str(source.get('reference_input_mode') or '').lower() == 't' else 't2',
            'reference_t_unit': 'ms' if str(source.get('reference_t_unit') or '').lower() == 'ms' else 'us',
            'reference_value': reference_value,
            'reference_t2_us2': reference_t2,
            'monotonic_slope': monotonic_direction,
            'phase_conversion_mode': 'monotonic_half_fringe',
        }

    def get_bragg_phase_calibrations(self) -> List[Dict[str, Any]]:
        payload = self._load_user_json_payload()
        calibrations = payload.get('bragg_phase_calibrations')
        if not isinstance(calibrations, list):
            return []
        normalized = []
        for calibration in calibrations:
            try:
                normalized.append(self._normalize_bragg_phase_calibration(calibration))
            except (TypeError, ValueError) as exc:
                print(f"[User JSON] Skipping invalid Bragg phase calibration: {exc}")
        return sorted(normalized, key=lambda item: item['created_at'], reverse=True)

    def build_bragg_phase_calibration(self, name: str, fit_result: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        label = str(name or '').strip()
        if not label:
            raise ValueError('Calibration name is required')
        return self._normalize_bragg_phase_calibration({
            **dict(fit_result or {}),
            'name': label,
            'source': dict(source or {}),
        })

    def save_bragg_phase_calibration(
        self,
        name: str,
        fit_result: Dict[str, Any],
        source: Dict[str, Any],
        *,
        activate_if_empty: bool = True,
    ) -> Dict[str, Any]:
        calibration = self.build_bragg_phase_calibration(name, fit_result, source)
        payload = self._load_user_json_payload()
        calibrations = self.get_bragg_phase_calibrations()
        calibrations.append(calibration)
        payload['bragg_phase_calibrations'] = calibrations
        if activate_if_empty and not str(payload.get('active_bragg_phase_calibration_id') or '').strip():
            payload['active_bragg_phase_calibration_id'] = calibration['id']
        elif 'active_bragg_phase_calibration_id' not in payload:
            payload['active_bragg_phase_calibration_id'] = ''
        self._save_user_json_payload(payload)
        return calibration

    def get_active_bragg_phase_calibration(self) -> Optional[Dict[str, Any]]:
        payload = self._load_user_json_payload()
        target = str(payload.get('active_bragg_phase_calibration_id') or '').strip()
        calibrations = self.get_bragg_phase_calibrations()
        if 'active_bragg_phase_calibration_id' in payload and not target:
            return None
        if target:
            selected = next((item for item in calibrations if item.get('id') == target), None)
            if selected is not None:
                return selected
        return calibrations[0] if calibrations else None

    def update_bragg_phase_calibration(self, calibration_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        target = str(calibration_id or '').strip()
        payload = self._load_user_json_payload()
        calibrations = self.get_bragg_phase_calibrations()
        for index, calibration in enumerate(calibrations):
            if calibration.get('id') != target:
                continue
            merged = {**calibration, **dict(updates or {}), 'id': target}
            calibrations[index] = self._normalize_bragg_phase_calibration(merged)
            payload['bragg_phase_calibrations'] = calibrations
            self._save_user_json_payload(payload)
            return calibrations[index]
        raise ValueError('Interferometer phase calibration not found')

    def set_active_bragg_phase_calibration(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        target = str(calibration_id or '').strip()
        calibration = next((item for item in self.get_bragg_phase_calibrations() if item.get('id') == target), None)
        if target and calibration is None:
            raise ValueError('Interferometer phase calibration not found')
        payload = self._load_user_json_payload()
        payload['active_bragg_phase_calibration_id'] = target
        self._save_user_json_payload(payload)
        return calibration

    def delete_bragg_phase_calibration(self, calibration_id: str) -> bool:
        target = str(calibration_id or '').strip()
        payload = self._load_user_json_payload()
        calibrations = self.get_bragg_phase_calibrations()
        kept = [item for item in calibrations if item.get('id') != target]
        if len(kept) == len(calibrations):
            return False
        payload['bragg_phase_calibrations'] = kept
        if str(payload.get('active_bragg_phase_calibration_id') or '') == target:
            payload['active_bragg_phase_calibration_id'] = kept[0]['id'] if kept else ''
        self._save_user_json_payload(payload)
        return True

