# main/server/mcp/agents/global_mcp_agents.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import os
from datetime import datetime
from typing import List
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from ..db.db_manager import DBManager

app = FastAPI(title="Vial MCP Global Agents")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["vial_mcp"]
agents_collection = db["agents"]
metrics = PerformanceMetrics()
db_manager = DBManager()

class AgentTask(BaseModel):
    task_id: str
    type: str
    status: str = "pending"
    parameters: dict
    user_id: str

class AgentTaskResponse(AgentTask):
    created_at: str
    updated_at: str

@app.post("/agents/tasks", response_model=AgentTaskResponse)
async def create_task(task: AgentTask, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("create_task", {"task_id": task.task_id, "user_id": task.user_id}):
        try:
            metrics.verify_token(token)
            task_dict = task.dict()
            task_dict["created_at"] = datetime.utcnow()
            task_dict["updated_at"] = task_dict["created_at"]
            task_id = db_manager.insert_one("agents", task_dict)
            return AgentTaskResponse(**task_dict, created_at=task_dict["created_at"], updated_at=task_dict["updated_at"])
        except Exception as e:
            handle_generic_error(e, context="create_task")
            raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.get("/agents/tasks/{user_id}", response_model=List[AgentTaskResponse])
async def get_tasks(user_id: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("get_tasks", {"user_id": user_id}):
        try:
            metrics.verify_token(token)
            tasks = db_manager.find_many("agents", {"user_id": user_id})
            return [AgentTaskResponse(**task, created_at=task["created_at"], updated_at=task["updated_at"]) for task in tasks]
        except Exception as e:
            handle_generic_error(e, context="get_tasks")
            raise HTTPException(status_code=500, detail=f"Failed to fetch tasks: {str(e)}")

@app.post("/agents/tasks/{task_id}/execute")
async def execute_task(task_id: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("execute_task", {"task_id": task_id}):
        try:
            metrics.verify_token(token)
            task = db_manager.find_one("agents", {"task_id": task_id})
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            # Placeholder for task execution logic (e.g., quantum circuit execution, AI task)
            db_manager.update_one("agents", {"task_id": task_id}, {"status": "completed", "updated_at": datetime.utcnow()})
            return {"status": "success", "message": f"Task {task_id} executed"}
        except Exception as e:
            handle_generic_error(e, context="execute_task")
            raise HTTPException(status_code=500, detail=f"Failed to execute task: {str(e)}")
