import os
import platform
from pathlib import Path

# --- Environment ---
IS_WINDOWS = platform.system() == "Windows"
USE_SIMULATION = IS_WINDOWS

# --- Base Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_BASE_DIR = BASE_DIR / "Data_log"

# Persistent Settings File
SETTINGS_FILE_PATH = BASE_DIR / "user_settings.json"

# --- Hardware Paths ---
TMOT_BINARY_PATH_WIN = str(BASE_DIR / "mock_bin" / "tmot4_mock.exe")
CMOT_BINARY_PATH_WIN = str(BASE_DIR / "mock_bin" / "cmot4_mock.exe")

# Real Hardware Default
MOT_BASE_DIR = BASE_DIR.parent / "mot4ztex"
TMOT_BINARY_PATH_LINUX = str(MOT_BASE_DIR / "tmot4")
CMOT_BINARY_PATH_LINUX = str(MOT_BASE_DIR / "cmot4")

SEQUENCE_TEMPLATE_PATH_WIN = str(BASE_DIR / "temp" / "seq0.mot")
SEQUENCE_TEMPLATE_PATH_LINUX = "./temp/seq0.mot"
SEQUENCE_OUTPUT_PATH = str(BASE_DIR / "temp" / "seq.mot")
VCD_OUTPUT_PATH = str(BASE_DIR / "temp" / "seq.vcd")

# --- Network ---
RP_IP_RED_REAL = "192.168.2.5"
RP_IP_GREEN_REAL = "192.168.3.5"
RP_PORT_REAL = 8000
RP_IP_MOCK = "127.0.0.1"
RP_PORT_MOCK = 8001

# --- Constants ---
NETWORK_TIMEOUT = 2 
G_CONST = 9.81
LINK_TOTAL_TIME = 100.0 

# --- Default Analysis Settings ---
DEFAULT_ANALYSIS_SETTINGS = {
    "alpha": 0.0151,
    "beta": 0.0188,
    "R": 1.1,
    "K": 7000.0,
    "z_up": 0.275,
    "z_dw": 0.255,
    "launch_velocity": 4.05,
    "chan_launch": "60",
    "chan_trigger": "68",
    "gain_up": -35.0, 
    "gain_dw": -35.0,
    "max_low": 0.0001,
    # [CRITICAL FIX] Default Decimation set to 8192 to match 1500pts -> ~100ms
    "decimation": 8192 
}
