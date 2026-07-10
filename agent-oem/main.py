import os
import json
import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn

from core.orchestra import OEMOrchestrator

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Initialize orchestrator (loads enabled plugins)
orchestrator = OEMOrchestrator(CONFIG)

app = FastAPI(title="Agent Grounding OEM Backend", version="2.3.1")

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

# API endpoint – handles all 10 phases
class PhaseRequest(BaseModel):
    phase: int
    data: Dict[str, Any] = {}

@app.post("/api")
async def handle_phase(req: PhaseRequest):
    phase = req.phase
    data = req.data
    if phase < 1 or phase > 10:
        raise HTTPException(status_code=400, detail="Phase must be 1-10")
    try:
        result = await orchestrator.route_phase(phase, data)
        return JSONResponse({"status": "ok", "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Optional: serve static files if needed
# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=CONFIG["system"].get("port", 8000), reload=True)
