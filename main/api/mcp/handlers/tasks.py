from typing import Dict, Any
from ...utils.logging import log_error, log_info
from ...config.redis_config import get_redis

class TaskHandler:
    async def create_task(self, task_id: str, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            redis = await get_redis()
            task = {"id": task_id, "type": task_type, "params": params, "status": "pending"}
            await redis.set(f"task:{task_id}", json.dumps(task), ex=86400)
            log_info(f"Task created: {task_id}")
            return task
        except Exception as e:
            log_error(f"Task creation failed: {str(e)}")
            raise

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        try:
            redis = await get_redis()
            task_data = await redis.get(f"task:{task_id}")
            if not task_data:
                log_error(f"Task not found: {task_id}")
                raise ValueError(f"Task not found: {task_id}")
            log_info(f"Task retrieved: {task_id}")
            return json.loads(task_data)
        except Exception as e:
            log_error(f"Task retrieval failed: {str(e)}")
            raise

    async def update_task_status(self, task_id: str, status: str) -> Dict[str, Any]:
        try:
            redis = await get_redis()
            task_data = await redis.get(f"task:{task_id}")
            if not task_data:
                log_error(f"Task not found: {task_id}")
                raise ValueError(f"Task not found: {task_id}")
            task = json.loads(task_data)
            task["status"] = status
            await redis.set(f"task:{task_id}", json.dumps(task), ex=86400)
            log_info(f"Task status updated: {task_id} to {status}")
            return task
        except Exception as e:
            log_error(f"Task status update failed: {str(e)}")
            raise
