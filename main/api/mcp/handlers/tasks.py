from fastapi import HTTPException
from ...utils.logging import log_error, log_info
from ..mcp_schemas import MCPTask
import uuid

class TaskHandler:
    def __init__(self):
        self.tasks = {}

    def create_task(self, name: str, description: str, priority: int = 0, dependencies: List[str] = []) -> MCPTask:
        task_id = str(uuid.uuid4())
        task = MCPTask(
            task_id=task_id,
            name=name,
            description=description,
            priority=priority,
            dependencies=dependencies
        )
        self.tasks[task_id] = task
        log_info(f"Task created: {task_id}")
        return task

    async def handle_task(self, task_id: str, action: str) -> MCPTask:
        if task_id not in self.tasks:
            log_error(f"Task {task_id} not found")
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        task = self.tasks[task_id]
        if action == "start":
            task.status = "running"
        elif action == "complete":
            task.status = "completed"
        elif action == "cancel":
            task.status = "cancelled"
        else:
            log_error(f"Invalid task action: {action}")
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
        log_info(f"Task {task_id} updated to {task.status}")
        return task
