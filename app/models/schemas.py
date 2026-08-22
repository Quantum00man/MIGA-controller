from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any


class FitParameterConfig(BaseModel):
    name: str
    guess: str = Field("")
    fixed: bool = Field(False)


class FitModelDefinition(BaseModel):
    key: str
    label: str
    formula: str
    parameters: List[FitParameterConfig] = Field(default_factory=list)
    roles: Dict[str, Optional[str]] = Field(default_factory=dict)
    area_mode: str = Field("window_integral")


class ScanConfig(BaseModel):
    """Full-Feature Scan Configuration."""
    scan_dimensions: int = Field(1)
    dim1_type: str = Field("range")
    param_type: str = Field("float")
    dim1_method: str = Field("step_size")
    start: float = Field(0)
    stop: float = Field(10)
    step: float = Field(1)
    custom_list: str = Field("")
    dim2_enabled: bool = Field(False)
    dim2_type: str = Field("range")
    dim2_param_type: str = Field("float")
    dim2_method: str = Field("step_size")
    dim2_start: float = Field(0)
    dim2_stop: float = Field(10)
    dim2_step: float = Field(1)
    dim2_list: str = Field("")
    dim3_enabled: bool = Field(False)
    dim3_type: str = Field("range")
    dim3_param_type: str = Field("float")
    dim3_method: str = Field("step_size")
    dim3_start: float = Field(0)
    dim3_stop: float = Field(10)
    dim3_step: float = Field(1)
    dim3_list: str = Field("")
    averages: int = Field(1)
    randomize: bool = Field(False)
    ext_trigger: bool = Field(False)
    run_label: str = Field("")
    sequence_name: str = Field("")
    mode: str = Field("standard")
    mode_param: Optional[float] = Field(None)
    link_formulas: List[str] = Field(default_factory=list)

    @validator("mode_param", pre=True)
    def normalize_mode_param(cls, value):
        if value == "":
            return None
        return value

    fit_center_up: float = Field(0)
    fit_width_up: float = Field(0)
    fit_center_dw: float = Field(0)
    fit_width_dw: float = Field(0)
    fit_model_key: str = Field("gaussian")
    fit_models: List[FitModelDefinition] = Field(default_factory=list)
    intf_alpha: float = 0.35
    intf_beta: float = 0.07636
    intf_gamma: float = 0.25
    bragg_shape: str = Field("blackman")
    bragg_base_timing: int = Field(331119)
    ac_stark_raman_group: str = Field("up")
    ac_stark_left_p0: int = Field(11)
    ac_stark_right_p0: int = Field(22)
    ac_stark_ratio_start: float = Field(0.5)
    ac_stark_ratio_stop: float = Field(2.0)
    ac_stark_ratio_step: float = Field(0.1)
    ac_stark_total_power: float = Field(100.0)
    ac_stark_dds_xml_name: str = Field("")
    lock_in_a_value: float = Field(1.0)
    lock_in_b_value: float = Field(0.5)
    parameter_source: str = Field("classic")
    marker_axes: List[str] = Field(default_factory=list)

    @validator("parameter_source")
    def normalize_parameter_source(cls, value):
        normalized = str(value or "classic").strip().lower()
        if normalized not in {"classic", "markers"}:
            raise ValueError("Parameter source must be classic or markers")
        return normalized

    @validator("marker_axes", pre=True)
    def normalize_marker_axes(cls, value):
        axes = value if isinstance(value, list) else []
        normalized = []
        for raw in axes[:3]:
            marker_id = str(raw or "").strip().upper()
            if marker_id:
                normalized.append(marker_id)
        return normalized

    @validator("ac_stark_raman_group")
    def normalize_ac_stark_raman_group(cls, value):
        normalized = str(value or "up").strip().lower()
        if normalized not in {"up", "down"}:
            raise ValueError("AC Stark Raman group must be 'up' or 'down'")
        return normalized


class BraggSingleExportRequest(BaseModel):
    sequence_name: str = Field("")
    fwhm: float = Field(..., gt=0)
    bragg_shape: str = Field("blackman")
    bragg_base_timing: int = Field(331119, gt=0)

    @validator("bragg_shape")
    def validate_bragg_shape(cls, value):
        normalized = str(value or "blackman").strip().lower()
        if normalized not in {"blackman", "gaussian"}:
            raise ValueError("Bragg shape must be blackman or gaussian")
        return normalized


class BraggScanExportRequest(BaseModel):
    sequence_name: str = Field("")
    scan_config: ScanConfig


class SequenceMarkerDefinition(BaseModel):
    id: str
    display_name: str
    kind: str
    decimals: int = Field(0, ge=0, le=9)
    hard_min: float
    hard_max: float
    default_start: float
    default_stop: float
    default_step: float = Field(..., gt=0)
    default_method: str = Field("step_size")
    expected_command: str = Field("")
    expected_channel: str = Field("")
    has_compensation: bool = Field(False)

    @validator("id")
    def validate_marker_id(cls, value):
        normalized = str(value or "").strip().upper()
        if not normalized:
            raise ValueError("Marker ID is required")
        return normalized

    @validator("kind")
    def validate_marker_kind(cls, value):
        normalized = str(value or "").strip().lower()
        if normalized not in {"dds_element", "dac_value", "duration", "digital_state"}:
            raise ValueError("Unsupported marker type")
        return normalized

    @validator("default_method")
    def validate_marker_default_method(cls, value):
        normalized = str(value or "step_size").strip().lower()
        if normalized not in {"step_size", "n_points"}:
            raise ValueError("Marker default method must be step_size or n_points")
        return normalized

    @validator("hard_max")
    def validate_marker_limits(cls, value, values):
        minimum = values.get("hard_min")
        if minimum is not None and value < minimum:
            raise ValueError("Marker hard maximum must be greater than or equal to hard minimum")
        return value


class SequenceMarkerInspectRequest(BaseModel):
    filename: str = Field("sequence.mot")
    content: str
    encoding: str = Field("utf-8")
    definitions: List[SequenceMarkerDefinition] = Field(default_factory=list)


class SequenceMarkerAnnotateRequest(SequenceMarkerInspectRequest):
    marker_id: str
    target_line_number: int = Field(..., gt=0)
    kind: str
    compensation_line_number: Optional[int] = Field(None, gt=0)



class SequenceMarkerUpdateRequest(SequenceMarkerInspectRequest):
    old_marker_id: str
    marker_id: str
    target_line_number: int = Field(..., gt=0)
    kind: str
    compensation_line_number: Optional[int] = Field(None, gt=0)

class SequenceMarkerRemoveRequest(SequenceMarkerInspectRequest):
    marker_id: str


class SequenceMarkerPreviewRequest(BaseModel):
    marker_axes: List[str]
    sequence_name: str = Field("")
    values: List[float]


class RamanPowerCalibration(BaseModel):
    """Generalized-logistic DDS amplitude to optical-power calibration."""

    lower_asymptote: float = Field(-6.30426)
    upper_asymptote: float = Field(151.029)
    growth_rate: float = Field(0.015975)
    midpoint: float = Field(198.203)
    shape: float = Field(1.29924)
    amplitude_min: int = Field(0, ge=0, le=1023)
    amplitude_max: int = Field(1023, ge=0, le=1023)

    @validator("growth_rate", "shape")
    def validate_positive_curve_parameter(cls, value):
        if value <= 0:
            raise ValueError("Calibration growth rate and shape must be positive")
        return value

    @validator("amplitude_max")
    def validate_amplitude_range(cls, value, values):
        minimum = int(values.get("amplitude_min", 0))
        if value < minimum:
            raise ValueError("Calibration amplitude_max must be >= amplitude_min")
        return value


class BraggPowerCalibration(BaseModel):
    """Generalized-logistic voltage to normalized optical-power calibration."""

    lower_asymptote: float = Field(0.0)
    upper_asymptote: float = Field(1.0)
    growth_rate: float = Field(1.0, gt=0)
    midpoint: float = Field(0.0)
    shape: float = Field(1.0, gt=0)
    linear_voltage_min: float = Field(-0.5, ge=-3.0, le=3.0)
    linear_voltage_max: float = Field(0.5, ge=-3.0, le=3.0)
    off_threshold: float = Field(0.001, ge=0.0, lt=1.0)

    @validator("upper_asymptote")
    def validate_power_range(cls, value, values):
        lower = float(values.get("lower_asymptote", 0.0))
        if value <= lower:
            raise ValueError("Bragg calibration upper_asymptote must exceed lower_asymptote")
        return value

    @validator("linear_voltage_max")
    def validate_linear_voltage_range(cls, value, values):
        minimum = float(values.get("linear_voltage_min", -0.5))
        if value <= minimum:
            raise ValueError("Bragg linear_voltage_max must exceed linear_voltage_min")
        return value


class ScheduleRequest(BaseModel):
    timingMode: str = Field("sequential")
    sequentialGapSec: float = Field(0, ge=0)
    tasks: List[Dict[str, Any]] = Field(default_factory=list)


class MarkerOptimizationStepConfig(BaseModel):
    marker_id: str
    start: float
    stop: float
    step: float = Field(..., gt=0)
    scan_method: str = Field("step_size")
    average_count: int = Field(1, ge=1, le=1000)
    randomize: bool = Field(False)
    metric_key: str = Field("transition_probability_up")
    metric_source: str = Field("fit")
    objective: str = Field("maximize")
    minimum_r_squared: float = Field(0.75, ge=-1.0, le=1.0)

    digital_states: Dict[str, str] = Field(default_factory=dict)

    @validator("digital_states", pre=True)
    def validate_digital_states(cls, value):
        states = value if isinstance(value, dict) else {}
        normalized = {}
        for marker_id, state in states.items():
            key = str(marker_id or "").strip().upper()
            selected = str(state or "current").strip().lower()
            if not key:
                continue
            if selected not in {"current", "on", "off"}:
                raise ValueError(f"Digital state for {key} must be current, on, or off")
            normalized[key] = selected
        return normalized

class MarkerOptimizationConfig(BaseModel):
    workflow_name: str = Field("")
    run_label: str = Field("")
    sequence_name: str = Field("")
    ext_trigger: bool = Field(False)
    fit_center_up: float = Field(0)
    fit_width_up: float = Field(0)
    fit_center_dw: float = Field(0)
    fit_width_dw: float = Field(0)
    steps: List[MarkerOptimizationStepConfig] = Field(default_factory=list)


class MarkerOptimizationPresetRequest(BaseModel):
    sequence_name: str
    name: str
    workflow: Dict[str, Any] = Field(default_factory=dict)


class OptimizationVariableConfig(BaseModel):
    index: int = Field(0, ge=0)
    lower: float = Field(0)
    upper: float = Field(1)
    step: float = Field(1, gt=0)
    parameter_type: str = Field("float")
    initial_guess: Optional[float] = Field(None)

    @validator("parameter_type")
    def normalize_parameter_type(cls, value):
        normalized = str(value or "float").strip().lower()
        if normalized not in {"float", "int"}:
            raise ValueError("parameter_type must be 'float' or 'int'")
        return normalized


class OptimizationConfig(BaseModel):
    run_label: str = Field("")
    sequence_name: str = Field("")
    ext_trigger: bool = Field(False)
    average_count: int = Field(1, ge=1)
    max_trials: int = Field(30, ge=1)
    initial_random_trials: int = Field(5, ge=0)
    objective_metric_key: str = Field("atom_number_up")
    objective_source: str = Field("fit")
    objective_mode: str = Field("maximize")
    target_value: Optional[float] = Field(None)
    target_tolerance: float = Field(0.0, ge=0.0)
    plateau_tolerance: float = Field(0.0, ge=0.0)
    plateau_window: int = Field(5, ge=1)
    variables: List[OptimizationVariableConfig] = Field(default_factory=list)
    fit_center_up: float = Field(0)
    fit_width_up: float = Field(0)
    fit_center_dw: float = Field(0)
    fit_width_dw: float = Field(0)

    @validator("objective_source")
    def normalize_objective_source(cls, value):
        normalized = str(value or "fit").strip().lower()
        if normalized not in {"fit", "nofit"}:
            raise ValueError("objective_source must be 'fit' or 'nofit'")
        return normalized

    @validator("objective_mode")
    def normalize_objective_mode(cls, value):
        normalized = str(value or "maximize").strip().lower()
        if normalized not in {"maximize", "minimize", "target"}:
            raise ValueError("objective_mode must be maximize, minimize, or target")
        return normalized

    @validator("variables")
    def validate_variables(cls, value):
        if not value:
            raise ValueError("At least one optimization variable is required")
        seen = set()
        for item in value:
            if item.index in seen:
                raise ValueError(f"Duplicate PARAMETER index: {item.index}")
            seen.add(item.index)
        return sorted(value, key=lambda item: item.index)


class AnalysisSettings(BaseModel):
    alpha: float
    beta: float
    R: float
    K: float
    k_detection_velocity_m_s: float = Field(3.58, gt=0)
    k_wavelength_nm: float = Field(780.0, gt=0)
    k_light_sheet_height_cm: float = Field(1.0, gt=0)
    k_transimpedance_gain_mohm: float = Field(10.0, gt=0)
    k_collection_efficiency: float = Field(0.02, gt=0)
    k_photodiode_responsivity_a_w: float = Field(0.5, gt=0)
    k_saturation_ratio: float = Field(1.5, gt=0)
    k_detuning_mhz: float = Field(0.0)
    k_natural_linewidth_mhz: float = Field(6.02, gt=0)
    z_up: float
    z_dw: float
    launch_velocity: float
    chan_launch: str
    chan_trigger: str
    gain_up: float
    gain_dw: float
    max_low: float = 0.01
    voltage_limit: float = 0.015
    decimation: int = 64
    intf_alpha: float = 0.35
    intf_beta: float = 0.07636
    intf_gamma: float = 0.25
    atom_area_method: str = Field("legacy")
    atom_area_baseline_points: int = Field(2)


class ArchiveAnalysisSettings(AnalysisSettings):
    fit_model_key: str = Field("gaussian")
    fit_models: List[FitModelDefinition] = Field(default_factory=list)


class SystemSettings(BaseModel):
    """Comprehensive System Settings."""
    hardware_platform: str = Field("redpitaya")
    rp_ip_red: str
    rp_ip_green: str
    daq_rate: Optional[float] = None
    network_timeout: int
    alpha: float
    beta: float
    R: float
    K: float
    k_detection_velocity_m_s: float = Field(3.58, gt=0)
    k_wavelength_nm: float = Field(780.0, gt=0)
    k_light_sheet_height_cm: float = Field(1.0, gt=0)
    k_transimpedance_gain_mohm: float = Field(10.0, gt=0)
    k_collection_efficiency: float = Field(0.02, gt=0)
    k_photodiode_responsivity_a_w: float = Field(0.5, gt=0)
    k_saturation_ratio: float = Field(1.5, gt=0)
    k_detuning_mhz: float = Field(0.0)
    k_natural_linewidth_mhz: float = Field(6.02, gt=0)
    z_up: float
    z_dw: float
    g_const: float
    launch_velocity: float
    chan_launch: str
    chan_trigger: str
    gain_up: float
    gain_dw: float
    max_low: float = 0.01
    decimation: int = 64
    link_total_time: float
    tmot_path: str
    tmot_args: Optional[str] = None
    cmot_path: str
    template_path: str
    voltage_limit: float = 0.015
    intf_alpha: float = 0.35
    intf_beta: float = 0.07636
    intf_gamma: float = 0.25
    atom_area_method: str = Field("legacy")
    atom_area_baseline_points: int = Field(2)
    fit_model_key: str = Field("gaussian")
    fit_models: List[FitModelDefinition] = Field(default_factory=list)
    update_repo_url: str = Field("")
    update_branch: str = Field("main")
    dds_writetable_path: str = Field("")
    raman_up_r1_calibration: RamanPowerCalibration = Field(default_factory=RamanPowerCalibration)
    raman_up_r2_calibration: RamanPowerCalibration = Field(default_factory=RamanPowerCalibration)
    raman_down_r1_calibration: RamanPowerCalibration = Field(default_factory=RamanPowerCalibration)
    raman_down_r2_calibration: RamanPowerCalibration = Field(default_factory=RamanPowerCalibration)
    bragg_power_calibration: BraggPowerCalibration = Field(default_factory=BraggPowerCalibration)
    sequence_marker_definitions: List[SequenceMarkerDefinition] = Field(default_factory=list)
    sequence_marker_profiles: Dict[str, List[SequenceMarkerDefinition]] = Field(default_factory=dict)


class SystemUpdateRequest(BaseModel):
    repo_url: Optional[str] = Field(None)
    branch: Optional[str] = Field(None)


class ExperimentResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class ReAnalysisRequest(BaseModel):
    year: str
    month: str
    day: str
    run_id: str
    new_settings: ArchiveAnalysisSettings
    updated_data: Optional[List[Dict[str, Any]]] = None


class ArchiveRunReference(BaseModel):
    year: str
    month: str
    day: str
    run_id: str


class ArchiveWaveformRequest(BaseModel):
    year: str
    month: str
    day: str
    run_id: str
    step_indices: List[int]
    new_settings: ArchiveAnalysisSettings


class ArchiveAllanRequest(BaseModel):
    year: str
    month: str
    day: str
    run_id: str
    order: int = Field(1, ge=1)
    display_mode: str = Field("saved")
    p0_min: Optional[float] = Field(None)
    p0_max: Optional[float] = Field(None)
    new_settings: ArchiveAnalysisSettings


class ArchiveScanFitRequest(BaseModel):
    x_values: List[float] = Field(default_factory=list)
    y_values: List[float] = Field(default_factory=list)
    fit_min: Optional[float] = Field(None)
    fit_max: Optional[float] = Field(None)
    eval_points: int = Field(400, ge=32, le=4000)
    model: FitModelDefinition


class ScanFitModelSaveRequest(BaseModel):
    name: str
    model: FitModelDefinition


class IndexUiStateRequest(BaseModel):
    state: Dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[str] = Field(None)
