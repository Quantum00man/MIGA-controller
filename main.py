import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json

# Import app modules
from app.api.routes import router
from app.core.experiment_manager import ExperimentManager

# --- 1. FastAPI App Initialization ---
app = FastAPI(title="MIGA Experiment Controller")

# Allow CORS (Cross-Origin Resource Sharing)
# Essential for allowing the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. WebSocket Connection Manager ---
class ConnectionManager:
    """
    Manages active WebSocket connections.
    Handles accepting connections, disconnecting, and broadcasting messages.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Send JSON message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to websocket: {e}")

ws_manager = ConnectionManager()

# --- 3. The Sync-to-Async Bridge ---
# Global variable to store the main event loop
main_event_loop = None

def data_callback_bridge(data: Dict[str, Any]):
    """
    Bridge function called by ExperimentManager (running in a background thread).
    It schedules the async broadcast task on the main event loop to be thread-safe.
    """
    if main_event_loop and ws_manager.active_connections:
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(data), 
            main_event_loop
        )

# --- 4. Startup Events ---
@app.on_event("startup")
async def startup_event():
    """
    Executed when the server starts.
    Initializes the event loop reference and hooks the manager to the WebSocket bridge.
    """
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    
    # Inject the callback into the Singleton ExperimentManager
    manager = ExperimentManager()
    manager.on_data_ready = data_callback_bridge
    print(">>> Server Started. ExperimentManager hooked to WebSocket.")

# --- 5. Mount API Routes ---
app.include_router(router)

# --- [CRITICAL ORDER] 6. Define WebSocket Endpoint ---
# The WebSocket route MUST be defined BEFORE mounting StaticFiles.
# Otherwise, the catch-all static route will intercept and block WebSocket handshake.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages (if any)
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# --- 7. Mount Static Files (LAST) ---
# Mounts the 'static' directory to root '/'.
# This acts as a catch-all for any request not matched by API or WebSocket above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")