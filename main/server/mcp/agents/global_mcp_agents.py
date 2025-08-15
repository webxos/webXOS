# main/server/mcp/agents/global_mcp_agents.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
import os
import json

app = FastAPI(title="Vial MCP Global Agents")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class AgentTask(BaseModel):
    task_id: str
    user_id: str
    task_type: str
    parameters: Dict
    status: str = "pending"

class AgentTaskResponse(BaseModel):
    task_id: str
    user_id: str
    task_type: str
    status: str
    result: Dict
    timestamp: str

class GlobalAgents:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.db_manager = DBManager()

    async def execute_task(self, task: AgentTask) -> Dict:
        with self.metrics.track_span("execute_task", {"task_id": task.task_id, "task_type": task.task_type}):
            try:
                # Simulate task execution (e.g., scheduling, resource allocation)
                result = {"status": "completed", "output": f"Processed {task.task_type} with params {task.parameters}"}
                task_data = task.dict()
                task_data["result"] = result
                task_data["timestamp"] = datetime.utcnow()
                task_data["status"] = "completed"
                self.db_manager.update_one("tasks", {"task_id": task.task_id}, task_data)
                return result
            except Exception as e:
                handle_generic_error(e, context="execute_task")
                raise

global_agents = GlobalAgents()

@app.post("/agents/tasks", response_model=AgentTaskResponse)
async def create_task(task: AgentTask, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("create_task", {"task_id": task.task_id, "user_id": task.user_id}):
        try:
            metrics.verify_token(token)
            task_data = task.dict()
            task_data["timestamp"] = datetime.utcnow()
            task_id = db_manager.insert_one("tasks", task_data)
            result = await global_agents.execute_task(task)
            return AgentTaskResponse(task_id=task_id, result=result, **task_data)
        except Exception as e:
            handle_generic_error(e, context="create_task")
            raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.get("/agents/tasks/{user_id}", response_model=List[AgentTaskResponse])
async def list_tasks(user_id: str, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("list_tasks", {"user_id": user_id}):
        try:
            metrics.verify_token(token)
            tasks = db_manager.find_many("tasks", {"user_id": user_id})
            return [AgentTaskResponse(task_id=str(task["_id"]), **task) for task in tasks]
        except Exception as e:
            handle_generic_error(e, context="list_tasks")
            raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")
