import time
import random
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Realistic Mock RedPitaya")

# Global storage for the generated waveforms
virtual_storage = {
    "ch1": "",
    "ch2": ""
}

# Constants matching real hardware defaults
DECIMATION = 64
BASE_CLOCK = 125e6 # 125 MHz
DT = (1 / BASE_CLOCK) * DECIMATION # 8ns * 64 = 512ns
DURATION = 0.1 # 100ms
NUM_POINTS = int(DURATION / DT) # ~195312 points

def generate_new_shot():
    # 1. Generate Time Axis
    # Note: In real hardware, we only send voltage, but we need time here to generate shapes
    t = np.linspace(0, DURATION, NUM_POINTS)
    
    # 2. Generate Noise
    # Real hardware has some baseline offset and noise
    noise_level = 0.005
    noise_ch1 = np.random.normal(0, noise_level, NUM_POINTS)
    noise_ch2 = np.random.normal(0, noise_level, NUM_POINTS)
    
    # 3. Generate Physics Signals (Gaussian Absorption Dips/Peaks)
    # UP Signal (centered ~30ms)
    center_up = 0.030 + random.uniform(-0.002, 0.002) 
    # Use negative amplitude to simulate absorption dip (common in MOT) or positive for fluorescence
    # Let's assume positive peaks based on previous code, but apply Gain=-35 later will flip it.
    # If gain is negative, raw signal should be "dip" (negative)? 
    # Actually usually photodiodes give positive voltage, and Inverting Amp gives negative.
    # Let's generate NEGATIVE pulses like a raw inverted signal, so dividing by -35 makes it positive atom number.
    # Wait, previous logic was: RAW / GAIN -> if Gain=-35, Raw must be negative to get positive atoms?
    # Let's stick to the previous visual: Raw was usually displayed as a peak or dip.
    # Let's generate POSITIVE raw voltage (0 to 1V), assuming the preamp does the inversion later or physically.
    # Actually, let's stick to the visual we had: A dip or peak.
    # Let's generate a DIP (Gaussian subtracted from baseline)
    baseline_up = 0.8
    amp_up = 0.2  # Depth
    width_up = 0.004 + random.uniform(-0.0005, 0.0005) # ~4ms
    
    # Signal = Baseline - Gaussian (Absorption)
    # sig_up = baseline_up - amp_up * np.exp(-(t - center_up)**2 / (2 * width_up**2)) + noise_ch1
    
    # OR simpler: Just a peak on zero background (Fluorescence) which matches previous mock
    amp_up = 0.5 + random.uniform(-0.05, 0.05)
    sig_up = amp_up * np.exp(-(t - center_up)**2 / (2 * width_up**2)) + noise_ch1
    
    # DOWN Signal (centered ~70ms)
    center_dw = 0.070 + random.uniform(-0.002, 0.002)
    amp_dw = 0.3 + random.uniform(-0.05, 0.05)
    width_dw = 0.004 + random.uniform(-0.0005, 0.0005)
    sig_dw = amp_dw * np.exp(-(t - center_dw)**2 / (2 * width_dw**2)) + noise_ch2

    # 4. Format Data
    # First line: Timestamp (Unix Epoch)
    # Subsequent lines: Voltage only
    timestamp_str = f"{time.time():.6f}"
    
    # Efficient string conversion for large arrays
    # We join rounded values to simulate ADC precision and save bandwidth
    # Using numpy char module for fast string formatting is better than list comprehension for 200k points
    
    # Convert to formatted strings (e.g. "0.123456")
    # Adding newline char to each element then joining is memory heavy but simple
    
    # Optimization: Use simple iteration or numpy tofile approach if needed, 
    # but for 200k points, join is acceptable in local mock.
    
    # CH1
    ch1_data = [timestamp_str]
    ch1_data.extend([f"{v:.5f}" for v in sig_up])
    virtual_storage["ch1"] = "\n".join(ch1_data)

    # CH2
    ch2_data = [timestamp_str]
    ch2_data.extend([f"{v:.5f}" for v in sig_dw])
    virtual_storage["ch2"] = "\n".join(ch2_data)
    
    print(f"[MockDevice] Shot Generated. Points: {NUM_POINTS}, UP: {center_up*1000:.1f}ms")

@app.get("/")
def read_root():
    return {"status": "online", "points": NUM_POINTS, "duration": DURATION}

@app.post("/trigger")
def hardware_trigger():
    generate_new_shot()
    return {"status": "triggered"}

@app.get("/ch1.dat", response_class=PlainTextResponse)
def get_ch1():
    if not virtual_storage["ch1"]: generate_new_shot()
    return virtual_storage["ch1"]

@app.get("/ch2.dat", response_class=PlainTextResponse)
def get_ch2():
    if not virtual_storage["ch2"]: generate_new_shot()
    return virtual_storage["ch2"]

if __name__ == "__main__":
    # Pre-generate one shot so it's not empty on start
    generate_new_shot()
    print(f">>> Starting Realistic Mock Device on Port 8001")
    print(f">>> Emulating: 125MHz Clock, Decimation {DECIMATION}, {DURATION}s duration")
    uvicorn.run(app, host="127.0.0.1", port=8001)
