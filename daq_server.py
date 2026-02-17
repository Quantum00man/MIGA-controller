import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, HTMLResponse
import ctypes
from ctypes import *
import numpy as np
import time
import sys
import threading
import random

# --- 1. 导入驱动 ---
try:
    import libvkdaq
    print("[DAQ Server] Library imported.")
except Exception as e:
    print(f"[DAQ Server] Error: {e}")
    sys.exit(1)

app = FastAPI()

# --- 2. 全局共享内存 ---
SHARED_MEMORY = {
    "ch1": None,
    "ch2": None, # Demo里只有CH1，为了网页不报错，我们还是会采CH2
    "timestamp": 0.0,
    "lock": threading.Lock()
}

# 默认配置 (复刻 Demo: 400点, 3990Hz)
CONFIG = {
    "fsamp": 3990,
    "Npoint": 390,
    "running": True,
    "needs_reinit": True
}

# --- 3. 极简复刻版核心逻辑 ---
def daq_worker_loop():
    print("[Worker] Started (Concise Demo Clone).")
    
    # 使用随机名防止冲突
    task_name = f"VkDaqServer_{random.randint(1000,9999)}".encode('utf-8')
    task_p = c_char_p(task_name)
    task_created = False

    while CONFIG["running"]:
        try:
            # === 初始化部分 (对应 Demo 的 main 开头) ===
            if CONFIG["needs_reinit"]:
                if task_created:
                    libvkdaq.VkDaqStopTask(task_p)
                    libvkdaq.VkDaqClearTask(task_p)
                
                Npoint = int(CONFIG["Npoint"])
                fsamp = int(CONFIG["fsamp"])
                
                # 1. Create Task
                libvkdaq.VkDaqCreateTask(task_p)
                task_created = True
                
                # 2. Config Channels (Demo只用了AIN1，但为了Server兼容性我们开两个)
                #    Demo acq_range = 0.1
                libvkdaq.VkDaqCreateAIVoltageChan(task_p, b"dev1/AIN1", b"", 0, -0.2, 0.2, 0, b"")
                libvkdaq.VkDaqCreateAIVoltageChan(task_p, b"dev1/AIN2", b"", 0, -0.2, 0.2, 0, b"")
                
                # 3. Config Clock (Demo 参数)
                #    Rising=1, Finite=1
                libvkdaq.VkDaqCfgSampClkTiming(task_p, 0, float(fsamp), 1, 1, Npoint)
                
                # 4. Config Trigger
                libvkdaq.VkDaqCfgDigEdgeRefTrig(task_p, b"dev1/DIN1.1", 1, 0)
                
                # 5. Start Task (Start 在 While 循环外面!)
                libvkdaq.VkDaqStartTask(task_p)
                print("[Worker] Task Started. Waiting for triggers...")
                
                CONFIG["needs_reinit"] = False

            # === 循环读取部分 (对应 Demo 的 while True) ===
            
            # 准备 Buffer (2个通道 * Npoint)
            pts = int(CONFIG["Npoint"])
            data_buffer = (ctypes.c_double * (pts * 2))()
            
            # Read Code (Timeout=1, GroupByChannel=1)
            # 这里的 1.0 对应 Demo 里的阻塞等待
            read = libvkdaq.VkDaqGetTaskData(task_p, data_buffer, pts, 1, 1.0)
            
            if read > 0:
                # 对应 Demo: print(read)
                print(f"[Worker] Captured {read} pts.")
                
                # 解析数据
                arr = np.ctypeslib.as_array(data_buffer)
                
                # GroupByChannel: 前一半是CH1，后一半是CH2
                with SHARED_MEMORY["lock"]:
                    SHARED_MEMORY["ch1"] = arr[0:pts].tolist()
                    SHARED_MEMORY["ch2"] = arr[pts:2*pts].tolist()
                    SHARED_MEMORY["timestamp"] = time.time()
                
                # [关键] 模拟 Demo 中 np.savetxt 的耗时
                # 这就是为什么 Demo 不会多次触发的原因！
                time.sleep(0.05) 
                
        except Exception as e:
            print(f"[Worker Error] {e}")
            time.sleep(1)
            CONFIG["needs_reinit"] = True

    # 清理
    if task_created:
        libvkdaq.VkDaqStopTask(task_p)
        libvkdaq.VkDaqClearTask(task_p)

# --- 4. 启动与API ---
@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=daq_worker_loop, daemon=True)
    t.start()

@app.on_event("shutdown")
def shutdown_event():
    CONFIG["running"] = False

@app.post("/configure")
def configure(sample_rate: int = 3990, points: int = 400):
    CONFIG["fsamp"] = sample_rate
    CONFIG["Npoint"] = points
    CONFIG["needs_reinit"] = True
    return {"status": "ok"}

# 保持 API 接口不变，让网页能用
@app.get("/", response_class=HTMLResponse)
def index(): return "<html><body><h1>DAQ Server (Demo Clone)</h1></body></html>"

@app.get("/ch1.dat", response_class=PlainTextResponse)
def get_ch1():
    with SHARED_MEMORY["lock"]:
        if SHARED_MEMORY["ch1"] is None: return f"{time.time()}\n0.0"
        return f"{SHARED_MEMORY['timestamp']}\n" + "\n".join([f"{v:.6f}" for v in SHARED_MEMORY["ch1"]])

@app.get("/ch2.dat", response_class=PlainTextResponse)
def get_ch2():
    with SHARED_MEMORY["lock"]:
        if SHARED_MEMORY["ch2"] is None: return f"{time.time()}\n0.0"
        return f"{SHARED_MEMORY['timestamp']}\n" + "\n".join([f"{v:.6f}" for v in SHARED_MEMORY["ch2"]])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
