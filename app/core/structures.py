from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ExperimentStatus:
    """Data model representing the current state of the experiment."""
    is_running: bool = False
    current_step: int = 0
    total_steps: int = 0
    message: str = "IDLE"

@dataclass
class ScanResult:
    """Data model representing the result of a single scan point."""
    parameter: float
    timestamp: float
    
    run_id: str = ""
    current_step: int = 0
    total_steps: int = 1
    
    # [新增] 用于前端显示的实际物理延时 (秒)
    detected_delay: Optional[float] = None
    
    raw_data_up: List[float] = None
    raw_data_dw: List[float] = None
    fit_data_up: Optional[List[float]] = None
    fit_data_dw: Optional[List[float]] = None
    time_axis: Optional[List[float]] = None
    
    window_up: Optional[Tuple[float, float]] = None
    window_dw: Optional[Tuple[float, float]] = None
    all_parameters: Optional[List[float]] = None

    # Metrics (FIT)
    atom_number_up: Optional[float] = None
    atom_number_dw: Optional[float] = None
    amplitude_up: Optional[float] = None
    amplitude_dw: Optional[float] = None
    sigma_up: Optional[float] = None
    sigma_dw: Optional[float] = None
    temperature_up: Optional[float] = None
    temperature_dw: Optional[float] = None
    arrival_time_up: Optional[float] = None
    arrival_time_dw: Optional[float] = None
    transition_probability_up: Optional[float] = None
    transition_probability_dw: Optional[float] = None

    # Metrics (NO FIT)
    atom_number_up_nofit: Optional[float] = None
    atom_number_dw_nofit: Optional[float] = None
    amplitude_up_nofit: Optional[float] = None
    amplitude_dw_nofit: Optional[float] = None
    sigma_up_nofit: Optional[float] = None
    sigma_dw_nofit: Optional[float] = None
    temperature_up_nofit: Optional[float] = None
    temperature_dw_nofit: Optional[float] = None
    arrival_time_up_nofit: Optional[float] = None
    arrival_time_dw_nofit: Optional[float] = None
    transition_probability_up_nofit: Optional[float] = None
    transition_probability_dw_nofit: Optional[float] = None

    # [New] Interferometer Output (Fit)
    intf_n1: Optional[float] = None
    intf_n2: Optional[float] = None
    intf_p1: Optional[float] = None  # P_N1
    intf_p2: Optional[float] = None  # P_N2

    # [New] Interferometer Output (NoFit)
    intf_n1_nofit: Optional[float] = None
    intf_n2_nofit: Optional[float] = None
    intf_p1_nofit: Optional[float] = None
    intf_p2_nofit: Optional[float] = None