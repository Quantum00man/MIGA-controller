from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ScanConfig(BaseModel):
    """Full-Feature Scan Configuration."""
    dim1_type: str = Field("range")
    param_type: str = Field("float")
    dim1_method: str = Field("step_size")
    start: float = Field(0)
    stop: float = Field(10)
    step: float = Field(1)
    custom_list: str = Field("")
    dim2_enabled: bool = Field(False)
    dim2_type: str = Field("range")
    dim2_method: str = Field("step_size")
    dim2_start: float = Field(0)
    dim2_stop: float = Field(10)
    dim2_step: float = Field(1)
    dim2_list: str = Field("")
    averages: int = Field(1)
    randomize: bool = Field(False)
    mode: str = Field("standard")
    mode_param: Optional[float] = Field(None)
    link_formulas: List[str] = Field(default_factory=list)
    fit_center_up: float = Field(0)
    fit_width_up: float = Field(0)
    fit_center_dw: float = Field(0)
    fit_width_dw: float = Field(0)
    intf_alpha: float = 0.35
    intf_beta: float = 0.07636
    intf_gamma: float = 0.25

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
    # [NEW]
    decimation: int = 64
    intf_alpha: float = 0.35
    intf_beta: float = 0.07636
    intf_gamma: float = 0.25

class SystemSettings(BaseModel):
    """Comprehensive System Settings."""
    rp_ip_red: str
    rp_ip_green: str
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
    # [NEW]
    decimation: int = 64
    link_total_time: float
    tmot_path: str
    cmot_path: str
    template_path: str
    voltage_limit: float = 0.015
    intf_alpha: float = 0.35
    intf_beta: float = 0.07636
    intf_gamma: float = 0.25

class ExperimentResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class ReAnalysisRequest(BaseModel):
    year: str
    month: str
    day: str
    run_id: str
    new_settings: AnalysisSettings
    updated_data: List[Dict[str, Any]]