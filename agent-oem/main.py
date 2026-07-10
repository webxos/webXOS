import os
import json
import logging
import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn

from core.orchestra import OEMOrchestrator

# =============================================================================
# Logging setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("agent-backend")

# =============================================================================
# Load configuration
# =============================================================================
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# =============================================================================
# Initialize orchestrator (loads enabled plugins)
# =============================================================================
orchestrator = OEMOrchestrator(CONFIG)

# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(
    title="Agent Grounding OEM Backend",
    version="2.3.1",
    description="Modular multi‑agent backend for the Agent Grounding protocol"
)

# -----------------------------------------------------------------------------
# Optional CORS – uncomment if frontend is served from a different origin
# -----------------------------------------------------------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],               # or specify your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main Agent Grounding frontend."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("index.html not found")
        return HTMLResponse(
            "<h1>index.html missing</h1><p>Please ensure index.html is in the root directory.</p>",
            status_code=404
        )

class PhaseRequest(BaseModel):
    phase: int
    data: Dict[str, Any] = {}

@app.post("/api")
async def handle_phase(req: PhaseRequest):
    """
    Execute a protocol phase (1‑10) with the provided data.
    Returns the result or an error.
    """
    phase = req.phase
    data = req.data

    if phase < 1 or phase > 10:
        logger.warning(f"Invalid phase requested: {phase}")
        raise HTTPException(status_code=400, detail="Phase must be between 1 and 10")

    logger.info(f"Processing phase {phase} with data: {json.dumps(data, default=str)[:200]}")
    try:
        result = await orchestrator.route_phase(phase, data)
        logger.info(f"Phase {phase} completed successfully")
        return JSONResponse({"status": "ok", "result": result})
    except ValueError as e:
        logger.error(f"Phase {phase} validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Phase {phase} unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health():
    """Health check endpoint – returns status and active plugins."""
    return {
        "status": "healthy",
        "enabled_plugins": list(orchestrator.active_agents.keys())
    }

# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    port = CONFIG["system"].get("port", 8000)
    logger.info(f"Starting backend on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
