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


class ScheduleRequest(BaseModel):
    timingMode: str = Field("sequential")
    sequentialGapSec: float = Field(0, ge=0)
    tasks: List[Dict[str, Any]] = Field(default_factory=list)


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
