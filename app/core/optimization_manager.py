import copy
import csv
import json
import math
import threading
import time
import traceback
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import config
from app.core.data_manager import DataManager
from app.core.experiment_manager import ExperimentManager


OBJECTIVE_METRICS: Dict[str, str] = {
    'atom_number_up': 'Atom Number UP',
    'atom_number_dw': 'Atom Number DOWN',
    'amplitude_up': 'Max Voltage UP',
    'amplitude_dw': 'Max Voltage DOWN',
    'temperature_up': 'Temperature UP',
    'temperature_dw': 'Temperature DOWN',
    'arrival_time_up': 'Arrival Time UP',
    'arrival_time_dw': 'Arrival Time DOWN',
    'transition_probability_up': 'Transition Probability UP',
    'transition_probability_dw': 'Transition Probability DOWN',
    'intf_n1': 'Interferometer N1',
    'intf_n2': 'Interferometer N2',
    'intf_p1': 'Interferometer P1',
    'intf_p2': 'Interferometer P2',
}


class OptimizationManager:
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self._status_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._status = self._build_idle_status()
        self._artifact_paths: Dict[str, Path] = {}

    def _build_idle_status(self) -> Dict[str, Any]:
        return {
            'is_running': False,
            'phase': 'idle',
            'message': 'IDLE',
            'run_id': None,
            'started_at_ms': None,
            'ended_at_ms': None,
            'average_count': 0,
            'max_trials': 0,
            'initial_random_trials': 0,
            'completed_trials': 0,
            'current_trial': 0,
            'current_repeat': 0,
            'objective_metric_key': 'atom_number_up',
            'objective_metric_label': OBJECTIVE_METRICS['atom_number_up'],
            'objective_source': 'fit',
            'objective_mode': 'maximize',
            'target_value': None,
            'target_tolerance': 0.0,
            'plateau_tolerance': 0.0,
            'plateau_window': 5,
            'stop_reason': None,
            'best_trial_index': None,
            'best_score': None,
            'best_metric_mean': None,
            'best_metric_std': None,
            'best_parameters': {},
            'current_parameters': {},
            'latest_trial': None,
            'history': [],
            'export_urls': {},
            'error': None,
        }

    def _set_status(self, **updates: Any):
        with self._status_lock:
            self._status.update(updates)

    def _append_history(self, entry: Dict[str, Any]):
        with self._status_lock:
            history = list(self._status.get('history') or [])
            history.append(entry)
            self._status['history'] = history
            self._status['latest_trial'] = entry
            self._status['completed_trials'] = len(history)
            self._status['current_trial'] = len(history)

    def _snapshot_status(self) -> Dict[str, Any]:
        with self._status_lock:
            return copy.deepcopy(self._status)

    def get_status(self) -> Dict[str, Any]:
        return self._snapshot_status()

    def is_running(self) -> bool:
        with self._status_lock:
            return bool(self._status.get('is_running'))

    def start_optimization(self, config_payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_running():
            return {'status': 'error', 'message': 'Optimization already running'}

        acquired, busy_message = self.experiment_manager.acquire_run_slot('optimization')
        if not acquired:
            return {'status': 'error', 'message': busy_message}

        try:
            self.experiment_manager.refresh_runtime_settings_from_disk()
            fit_config = self.experiment_manager.build_fit_config(config_payload)
            self._ensure_ax_importable()
        except Exception as exc:
            self.experiment_manager.release_run_slot('optimization')
            return {'status': 'error', 'message': str(exc)}

        self._stop_requested = False
        self._artifact_paths = {}
        initial_status = self._build_idle_status()
        initial_status.update({
            'is_running': True,
            'phase': 'running',
            'message': 'Starting optimization...',
            'started_at_ms': int(time.time() * 1000),
            'average_count': int(config_payload.get('average_count', 1) or 1),
            'max_trials': int(config_payload.get('max_trials', 1) or 1),
            'initial_random_trials': int(config_payload.get('initial_random_trials', 0) or 0),
            'objective_metric_key': str(config_payload.get('objective_metric_key') or 'atom_number_up'),
            'objective_metric_label': OBJECTIVE_METRICS.get(str(config_payload.get('objective_metric_key') or 'atom_number_up'), 'Objective'),
            'objective_source': str(config_payload.get('objective_source') or 'fit'),
            'objective_mode': str(config_payload.get('objective_mode') or 'maximize'),
            'target_value': config_payload.get('target_value'),
            'target_tolerance': float(config_payload.get('target_tolerance', 0.0) or 0.0),
            'plateau_tolerance': float(config_payload.get('plateau_tolerance', 0.0) or 0.0),
            'plateau_window': int(config_payload.get('plateau_window', 5) or 5),
            'message': 'Optimization queued',
        })
        with self._status_lock:
            self._status = initial_status

        self._thread = threading.Thread(
            target=self._run_optimization,
            args=(copy.deepcopy(config_payload), fit_config),
            daemon=True,
        )
        self._thread.start()
        return {'status': 'success', 'message': 'Optimization started', 'data': self.get_status()}

    def stop_optimization(self) -> Dict[str, Any]:
        if not self.is_running():
            return {'status': 'warning', 'message': 'No optimization running'}
        self._stop_requested = True
        self._set_status(message='Stopping optimization...', phase='stopping')
        return {'status': 'success', 'message': 'Stop signal sent'}

    def get_export_file(self, kind: str) -> Tuple[Path, str]:
        path = self._artifact_paths.get(kind)
        if path is None or not path.exists():
            raise FileNotFoundError(f'Optimization export not available: {kind}')
        return path, path.name

    def _ensure_ax_importable(self):
        try:
            from ax.api.client import Client  # noqa: F401
            from ax.api.configs import ChoiceParameterConfig  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                'ax-platform is not available. Start the server with ./start_controller.sh so the project virtual environment is used instead of a system or sudo-only Python. '
                f'Original import error: {exc}'
            )

    def _emit(self, payload: Dict[str, Any]):
        callback = self.experiment_manager.on_data_ready
        if callback:
            callback(payload)

    def _parameter_name(self, index: int) -> str:
        return f'parameter_{index}'

    def _normalize_numeric_value(self, value: float, parameter_type: str) -> Any:
        if parameter_type == 'int':
            return int(round(float(value)))
        return round(float(value), 6)

    def _build_choice_values(self, variable: Dict[str, Any]) -> List[Any]:
        lower = float(variable.get('lower', 0.0))
        upper = float(variable.get('upper', lower))
        step = float(variable.get('step', 1.0))
        parameter_type = str(variable.get('parameter_type') or 'float').strip().lower()
        if step <= 0:
            raise ValueError(f"PARAMETER{variable.get('index', 0)} step must be positive")
        direction = 1.0 if upper >= lower else -1.0
        effective_step = abs(step) * direction
        tolerance = abs(effective_step) * 1e-9 + 1e-12
        values: List[Any] = []
        current = lower
        compare = (lambda raw: raw <= upper + tolerance) if direction > 0 else (lambda raw: raw >= upper - tolerance)
        guard = 0
        while compare(current):
            values.append(self._normalize_numeric_value(current, parameter_type))
            current += effective_step
            guard += 1
            if guard > 5000:
                raise ValueError(f"PARAMETER{variable.get('index', 0)} has too many discrete points; increase step size")
        if not values:
            values = [self._normalize_numeric_value(lower, parameter_type)]
        deduped: List[Any] = []
        seen = set()
        for item in values:
            key = repr(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _snap_to_grid(self, value: Optional[float], variable: Dict[str, Any], values: List[Any]) -> Any:
        if value is None:
            return values[0]
        target = float(value)
        return min(values, key=lambda item: abs(float(item) - target))

    def _resolve_metric_field(self, metric_key: str, source: str) -> str:
        return f"{metric_key}_nofit" if source == 'nofit' else metric_key

    def _extract_metric_value(self, result: Any, metric_key: str, source: str) -> float:
        field_name = self._resolve_metric_field(metric_key, source)
        value = getattr(result, field_name, None)
        if value is None:
            raise ValueError(f'Objective metric unavailable: {field_name}')
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f'Objective metric is not finite: {field_name}')
        return numeric

    def _score_metric(self, mean_value: float, objective_mode: str, target_value: Optional[float]) -> float:
        if objective_mode == 'maximize':
            return float(mean_value)
        if objective_mode == 'minimize':
            return float(-mean_value)
        if target_value is None or not math.isfinite(float(target_value)):
            raise ValueError('Target mode requires a numeric target value')
        return float(-abs(mean_value - float(target_value)))

    def _should_stop_for_plateau(self, history: List[Dict[str, Any]], plateau_window: int, plateau_tolerance: float) -> Tuple[bool, Optional[float]]:
        if len(history) <= plateau_window:
            return False, None
        current_best = float(history[-1]['best_score_so_far'])
        previous_best = float(history[-1 - plateau_window]['best_score_so_far'])
        improvement = current_best - previous_best
        return improvement <= plateau_tolerance, improvement

    def _evaluate_trial(
        self,
        data_manager: DataManager,
        fit_config: Dict[str, Any],
        config_payload: Dict[str, Any],
        trial_index: int,
        trial_parameters: Dict[str, Any],
        global_shot_index: int,
    ) -> Tuple[Dict[str, Any], int]:
        average_count = int(config_payload.get('average_count', 1) or 1)
        metric_key = str(config_payload.get('objective_metric_key') or 'atom_number_up')
        source = str(config_payload.get('objective_source') or 'fit')
        shot_values: List[float] = []
        shot_records: List[Dict[str, Any]] = []
        ordered_variable_indices = [int(variable['index']) for variable in config_payload.get('variables', [])]
        params_to_write = [trial_parameters[self._parameter_name(index)] for index in ordered_variable_indices]
        max_trials = int(config_payload.get('max_trials', 1) or 1)
        total_shots = max_trials * average_count

        for repeat_index in range(average_count):
            if self._stop_requested:
                raise InterruptedError('Optimization stopped by user')
            metadata = {
                'trial_index': trial_index,
                'repeat_index': repeat_index + 1,
                'average_count': average_count,
                'parameter_map': {f'PARAMETER{index}': trial_parameters[self._parameter_name(index)] for index in ordered_variable_indices},
            }
            job = self.experiment_manager.execute_single_measurement(
                params_to_write,
                config_payload,
                idx=global_shot_index + repeat_index,
                total_steps=total_shots,
                scan_dimensions=1,
                metadata=metadata,
            )
            result, payload = self.experiment_manager.process_measurement_job(
                job,
                fit_config,
                data_manager=data_manager,
                save_step_index=global_shot_index + repeat_index + 1,
                stream_type='optimization_shot',
                extra_payload={
                    'trial_index': trial_index,
                    'repeat_index': repeat_index + 1,
                    'average_count': average_count,
                },
            )
            self._emit(payload)
            if result is None:
                raise RuntimeError(payload.get('error') or 'Optimization shot failed')
            metric_value = self._extract_metric_value(result, metric_key, source)
            shot_values.append(metric_value)
            shot_records.append({
                'repeat_index': repeat_index + 1,
                'metric_value': float(metric_value),
                'shot_step': global_shot_index + repeat_index + 1,
            })
            self._set_status(
                current_trial=trial_index,
                current_repeat=repeat_index + 1,
                current_parameters={f'PARAMETER{index}': trial_parameters[self._parameter_name(index)] for index in ordered_variable_indices},
                message=f'Trial {trial_index}/{max_trials} · repeat {repeat_index + 1}/{average_count}',
            )

        metric_mean = float(mean(shot_values))
        metric_std = float(pstdev(shot_values)) if len(shot_values) > 1 else 0.0
        metric_sem = float(metric_std / math.sqrt(len(shot_values))) if len(shot_values) > 1 else 0.0
        score = self._score_metric(metric_mean, str(config_payload.get('objective_mode') or 'maximize'), config_payload.get('target_value'))
        return {
            'trial_index': trial_index,
            'parameters': {f'PARAMETER{index}': trial_parameters[self._parameter_name(index)] for index in ordered_variable_indices},
            'metric_mean': metric_mean,
            'metric_std': metric_std,
            'metric_sem': metric_sem,
            'score': float(score),
            'shot_records': shot_records,
        }, global_shot_index + average_count

    def _build_ax_client(self, config_payload: Dict[str, Any], has_initial_guess: bool):
        from ax.api.client import Client
        from ax.api.configs import ChoiceParameterConfig

        client = Client()
        parameter_configs = []
        for variable in config_payload.get('variables', []):
            values = self._build_choice_values(variable)
            parameter_configs.append(
                ChoiceParameterConfig(
                    name=self._parameter_name(int(variable['index'])),
                    parameter_type=str(variable.get('parameter_type') or 'float').strip().lower(),
                    values=values,
                    is_ordered=True,
                )
            )

        client.configure_experiment(
            name=str(config_payload.get('run_label') or 'optimization').strip() or 'optimization',
            parameters=parameter_configs,
        )
        if hasattr(client, 'configure_generation_strategy'):
            initialization_budget = int(config_payload.get('initial_random_trials', 0) or 0)
            if initialization_budget <= 0 and not has_initial_guess:
                initialization_budget = 1
            try:
                client.configure_generation_strategy(
                    initialization_budget=initialization_budget,
                    initialize_with_center=False,
                    allow_exceeding_initialization_budget=True,
                    use_existing_trials_for_initialization=has_initial_guess,
                )
            except Exception:
                pass
        client.configure_optimization(objective='optimization_score')
        if hasattr(client, 'configure_tracking_metrics'):
            try:
                client.configure_tracking_metrics(['measured_value'])
            except Exception:
                pass
        return client

    def _complete_ax_trial(self, client: Any, trial_index: int, evaluation: Dict[str, Any]):
        payload_candidates = [
            {
                'optimization_score': (evaluation['score'], evaluation['metric_sem']),
                'measured_value': (evaluation['metric_mean'], evaluation['metric_sem']),
            },
            {
                'optimization_score': evaluation['score'],
                'measured_value': evaluation['metric_mean'],
            },
            {
                'optimization_score': (evaluation['score'], evaluation['metric_sem']),
            },
            {
                'optimization_score': evaluation['score'],
            },
        ]
        last_error = None
        for raw_data in payload_candidates:
            try:
                client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                return
            except Exception as exc:
                last_error = exc
                print(
                    f"[Optimization Ax] complete_trial fallback failed for trial {trial_index} "
                    f"with payload keys {list(raw_data.keys())}: {exc}"
                )
        raise RuntimeError(f"Ax complete_trial failed for trial {trial_index}: {last_error}")

    def _write_history_csv(self, run_dir: Path, history: List[Dict[str, Any]]) -> Path:
        path = run_dir / 'optimization_history.csv'
        parameter_keys: List[str] = []
        for entry in history:
            for key in entry.get('parameters', {}).keys():
                if key not in parameter_keys:
                    parameter_keys.append(key)
        parameter_keys.sort(key=lambda key: int(key.replace('PARAMETER', '')))
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'trial_index',
                *parameter_keys,
                'metric_mean',
                'metric_std',
                'metric_sem',
                'score',
                'best_score_so_far',
                'best_metric_mean_so_far',
                'repeat_values',
            ])
            for entry in history:
                repeat_values = ';'.join(f"{shot['metric_value']:.8f}" for shot in entry.get('shot_records', []))
                writer.writerow([
                    entry.get('trial_index'),
                    *[entry.get('parameters', {}).get(key, '') for key in parameter_keys],
                    f"{float(entry.get('metric_mean', 0.0)):.8f}",
                    f"{float(entry.get('metric_std', 0.0)):.8f}",
                    f"{float(entry.get('metric_sem', 0.0)):.8f}",
                    f"{float(entry.get('score', 0.0)):.8f}",
                    f"{float(entry.get('best_score_so_far', 0.0)):.8f}",
                    f"{float(entry.get('best_metric_mean_so_far', 0.0)):.8f}",
                    repeat_values,
                ])
        return path

    def _write_report_json(self, run_dir: Path, config_payload: Dict[str, Any], final_status: Dict[str, Any]) -> Path:
        path = run_dir / 'optimization_report.json'
        report_payload = {
            'generated_at_ms': int(time.time() * 1000),
            'config': config_payload,
            'status': final_status,
        }
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(report_payload, handle, ensure_ascii=False, indent=2)
        return path

    def _write_best_sequence(self, run_dir: Path, best_parameters: Dict[str, Any]) -> Optional[Path]:
        if not best_parameters:
            return None
        ordered_keys = sorted(best_parameters.keys(), key=lambda key: int(key.replace('PARAMETER', '')))
        values = [best_parameters[key] for key in ordered_keys]
        output_path = run_dir / 'best_sequence.mot'
        template_path = config.SEQUENCE_TEMPLATE_PATH_WIN if config.USE_SIMULATION else self.experiment_manager.settings['template_path']
        self.experiment_manager.seq_editor.generate_sequence(str(template_path), str(output_path), values)
        return output_path

    def _run_optimization(self, config_payload: Dict[str, Any], fit_config: Dict[str, Any]):
        data_manager = DataManager()
        run_config = copy.deepcopy(config_payload)
        run_config['_system_settings_snapshot'] = copy.deepcopy(self.experiment_manager.settings)
        run_config['_optimization_mode'] = True
        run_config['sequence_name'] = str(config_payload.get('sequence_name') or '').strip()
        stop_reason = 'max_trials_reached'
        error_message = None
        best_score = None
        best_metric_mean = None
        best_metric_std = None
        best_parameters: Dict[str, Any] = {}
        best_trial_index = None
        plateau_improvement = None
        current_history: List[Dict[str, Any]] = []

        try:
            data_manager.init_run(run_config)
            self._set_status(run_id=data_manager.current_run_id_str, message='Optimization run initialized')

            initial_guess_parameters: Dict[str, Any] = {}
            all_have_initial_guess = True
            normalized_variables: List[Dict[str, Any]] = []
            for variable in config_payload.get('variables', []):
                variable_dict = dict(variable)
                values = self._build_choice_values(variable_dict)
                variable_dict['choices'] = values
                normalized_variables.append(variable_dict)
                if variable_dict.get('initial_guess') is None:
                    all_have_initial_guess = False
                initial_guess_parameters[self._parameter_name(int(variable_dict['index']))] = self._snap_to_grid(
                    variable_dict.get('initial_guess'),
                    variable_dict,
                    values,
                )
            config_payload['variables'] = normalized_variables

            client = self._build_ax_client(config_payload, has_initial_guess=all_have_initial_guess)
            global_shot_index = 0
            trial_counter = 0

            if all_have_initial_guess:
                trial_counter += 1
                attached_trial_index = client.attach_trial(parameters=initial_guess_parameters, arm_name='initial_guess')
                print(f"[Optimization] Attached initial guess as Ax trial {attached_trial_index}: {initial_guess_parameters}")
                evaluation, global_shot_index = self._evaluate_trial(
                    data_manager,
                    fit_config,
                    config_payload,
                    trial_counter,
                    initial_guess_parameters,
                    global_shot_index,
                )
                self._complete_ax_trial(client, attached_trial_index, evaluation)
                best_score = evaluation['score']
                best_metric_mean = evaluation['metric_mean']
                best_metric_std = evaluation['metric_std']
                best_parameters = dict(evaluation['parameters'])
                best_trial_index = trial_counter
                evaluation['best_score_so_far'] = best_score
                evaluation['best_metric_mean_so_far'] = best_metric_mean
                current_history.append(evaluation)
                self._append_history(evaluation)
                self._emit({
                    'stream_type': 'optimization_trial',
                    'trial_index': trial_counter,
                    'completed_trials': len(current_history),
                    'max_trials': int(config_payload.get('max_trials', 1) or 1),
                    'evaluation': evaluation,
                    'best_parameters': best_parameters,
                    'best_score': best_score,
                    'best_metric_mean': best_metric_mean,
                })

            max_trials = int(config_payload.get('max_trials', 1) or 1)
            objective_mode = str(config_payload.get('objective_mode') or 'maximize')
            target_value = config_payload.get('target_value')
            target_tolerance = float(config_payload.get('target_tolerance', 0.0) or 0.0)
            plateau_tolerance = float(config_payload.get('plateau_tolerance', 0.0) or 0.0)
            plateau_window = int(config_payload.get('plateau_window', 5) or 5)

            while len(current_history) < max_trials:
                if self._stop_requested:
                    stop_reason = 'stopped_by_user'
                    break

                generated = client.get_next_trials(max_trials=1)
                if not generated:
                    stop_reason = 'no_more_candidates'
                    break
                ax_trial_index, trial_parameters = next(iter(generated.items()))
                print(f"[Optimization] Evaluating Ax trial {ax_trial_index}: {trial_parameters}")
                trial_counter += 1
                evaluation, global_shot_index = self._evaluate_trial(
                    data_manager,
                    fit_config,
                    config_payload,
                    trial_counter,
                    trial_parameters,
                    global_shot_index,
                )
                self._complete_ax_trial(client, ax_trial_index, evaluation)

                if best_score is None or evaluation['score'] > best_score:
                    best_score = evaluation['score']
                    best_metric_mean = evaluation['metric_mean']
                    best_metric_std = evaluation['metric_std']
                    best_parameters = dict(evaluation['parameters'])
                    best_trial_index = trial_counter

                evaluation['best_score_so_far'] = float(best_score)
                evaluation['best_metric_mean_so_far'] = float(best_metric_mean)
                current_history.append(evaluation)
                self._append_history(evaluation)
                self._set_status(
                    best_trial_index=best_trial_index,
                    best_score=float(best_score),
                    best_metric_mean=float(best_metric_mean),
                    best_metric_std=float(best_metric_std or 0.0),
                    best_parameters=best_parameters,
                    current_repeat=0,
                    message=f'Trial {trial_counter}/{max_trials} completed',
                )
                self._emit({
                    'stream_type': 'optimization_trial',
                    'trial_index': trial_counter,
                    'completed_trials': len(current_history),
                    'max_trials': max_trials,
                    'evaluation': evaluation,
                    'best_parameters': best_parameters,
                    'best_score': best_score,
                    'best_metric_mean': best_metric_mean,
                })

                if objective_mode == 'target' and target_value is not None:
                    if abs(float(evaluation['metric_mean']) - float(target_value)) <= target_tolerance:
                        stop_reason = 'target_reached'
                        break
                if objective_mode in {'maximize', 'minimize'}:
                    should_stop, plateau_improvement = self._should_stop_for_plateau(current_history, plateau_window, plateau_tolerance)
                    if should_stop:
                        stop_reason = 'plateau_reached'
                        break

            run_dir = data_manager.current_run_dir
            history_path = self._write_history_csv(run_dir, current_history)
            best_sequence_path = self._write_best_sequence(run_dir, best_parameters)
            final_status = self._snapshot_status()
            final_status.update({
                'is_running': False,
                'phase': 'completed' if error_message is None and stop_reason != 'stopped_by_user' else ('stopped' if stop_reason == 'stopped_by_user' else 'error'),
                'message': 'Optimization completed' if error_message is None and stop_reason != 'stopped_by_user' else ('Optimization stopped' if stop_reason == 'stopped_by_user' else 'Optimization failed'),
                'ended_at_ms': int(time.time() * 1000),
                'stop_reason': stop_reason,
                'best_trial_index': best_trial_index,
                'best_score': best_score,
                'best_metric_mean': best_metric_mean,
                'best_metric_std': best_metric_std,
                'best_parameters': best_parameters,
                'plateau_improvement': plateau_improvement,
                'error': error_message,
            })
            report_path = self._write_report_json(run_dir, config_payload, final_status)
            artifact_paths = {
                'optimization_history': history_path,
                'optimization_report': report_path,
            }
            if best_sequence_path is not None:
                artifact_paths['best_sequence'] = best_sequence_path
            self._artifact_paths = artifact_paths
            export_urls = {kind: f'/optimization/download/{kind}' for kind in artifact_paths.keys()}
            final_status['export_urls'] = export_urls
            with self._status_lock:
                self._status.update(final_status)
            self._emit({
                'stream_type': 'optimization_complete',
                'status': final_status,
            })
        except InterruptedError:
            stop_reason = 'stopped_by_user'
            with self._status_lock:
                self._status.update({
                    'is_running': False,
                    'phase': 'stopped',
                    'message': 'Optimization stopped',
                    'ended_at_ms': int(time.time() * 1000),
                    'stop_reason': stop_reason,
                })
            self._emit({'stream_type': 'optimization_complete', 'status': self.get_status()})
        except Exception as exc:
            error_message = str(exc)
            print(f"[Optimization Error] {traceback.format_exc()}")
            with self._status_lock:
                self._status.update({
                    'is_running': False,
                    'phase': 'error',
                    'message': 'Optimization failed',
                    'ended_at_ms': int(time.time() * 1000),
                    'stop_reason': 'error',
                    'error': error_message,
                })
            self._emit({'stream_type': 'optimization_complete', 'status': self.get_status()})
        finally:
            data_manager.close_run()
            self.experiment_manager.release_run_slot('optimization')
            self._thread = None
            self._stop_requested = False
