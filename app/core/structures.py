from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ExperimentStatus:
    """Data model representing the current state of the experiment."""
    is_running: bool = False
    current_step: int = 0
    total_steps: int = 0
    message: str = "IDLE"
    is_paused: bool = False
    pause_reason: str = ""

@dataclass
class ScanResult:
    """Data model representing the result of a single scan point."""
    parameter: float
    timestamp: float
    scan_dimensions: int = 1

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

    # Transfer Function provenance. Empty for other scan modes.
    transfer_frequency_hz: Optional[float] = None
    transfer_repeat: Optional[int] = None
    transfer_phase_deg: Optional[float] = None
    # Zero-phase provenance.  Keep this with each shot so raw calibration shots
    # can be retained without being mistaken for transfer-response samples.
    transfer_zero_phase_baseline: bool = False
    transfer_zero_phase_block_id: Optional[int] = None
    transfer_zero_phase_repeat: Optional[int] = None
    power_meter_power_w: Optional[float] = None
    power_meter_measured_at: str = ""
    power_meter_wavelength_nm: Optional[float] = None
    power_meter_serial_number: str = ""
    power_meter_reference_w: Optional[float] = None
    power_meter_deviation_percent: Optional[float] = None
    power_meter_valid: Optional[bool] = None
    power_meter_invalid_reason: str = ""
    transfer_frequency_attempt: int = 1
    intf_alpha_applied: Optional[float] = None
    intf_alpha_calibration_block_id: Optional[int] = None
    intf_alpha_calibration_shot: Optional[int] = None

    # Bragg Fringes Calibration provenance. Empty for ordinary scan points.
    bragg_calibration_stage: str = ""

    # Ramsey Interferometer provenance. The displayed parameter is delta f in MHz.
    ramsey_delta_f_mhz: Optional[float] = None
    ramsey_repeat: Optional[int] = None
    ramsey_center_frequency_mhz: Optional[float] = None
    ramsey_ch1_frequency_mhz: Optional[float] = None
    ramsey_ch2_frequency_mhz: Optional[float] = None
    ramsey_ch1_power_dbm: Optional[float] = None
    ramsey_ch2_power_dbm: Optional[float] = None

    # AC Stark scan provenance. These stay empty for every other scan mode.
    ac_stark_ratio: Optional[float] = None
    ac_stark_side: Optional[str] = None
    ac_stark_dds_element: Optional[int] = None
    ac_stark_power_r1: Optional[float] = None
    ac_stark_power_r2: Optional[float] = None
    ac_stark_amplitude_r1: Optional[int] = None
    ac_stark_amplitude_r2: Optional[int] = None
    ac_stark_actual_power_r1: Optional[float] = None
    ac_stark_actual_power_r2: Optional[float] = None

    # ABBA digital lock-in provenance. Positions are 1-based within each block.
    lock_in_block_index: Optional[int] = None
    lock_in_position: Optional[int] = None
    lock_in_state: Optional[str] = None
    lock_in_reference: Optional[int] = None

    # Sequential Marker Optimization provenance. Empty for ordinary scans.
    workflow_step: Optional[int] = None
    workflow_marker: Optional[str] = None
    workflow_point: Optional[int] = None
    workflow_repeat: Optional[int] = None
    workflow_shot: Optional[int] = None
    workflow_randomized: Optional[bool] = None

    # Mean of the last 200 raw TOF samples before offset removal.
    tail_mean_up_raw: Optional[float] = None
    tail_mean_dw_raw: Optional[float] = None

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

    # Single calibrated interferometer phase channel. Its source metric,
    # channel and Fit/NoFit mode are stored in the calibration snapshot.
    interferometer_phase: Optional[float] = None
    # Calibrated phase before the run-local transfer zero is subtracted.
    # This makes archive re-referencing reversible.
    interferometer_phase_raw: Optional[float] = None
    interferometer_phase_valid: bool = False
    interferometer_phase_source_value: Optional[float] = None
    interferometer_phase_calibration_id: str = ""
    interferometer_phase_calibration_name: str = ""
    interferometer_phase_reference_t2_us2: Optional[float] = None
