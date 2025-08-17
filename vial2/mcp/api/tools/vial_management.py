from config.config import DatabaseConfig
from tools.agent_templates import initialize_model
import torch
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class VialManager:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.project_id = db.project_id

    async def execute(self, data: dict) -> dict:
        try:
            method = data.get("method")
            user_id = data.get("user_id")
            vial_id = data.get("vial_id")
            project_id = data.get("project_id", self.project_id)
            if project_id != self.project_id:
                error_message = f"Invalid project ID: {project_id} [vial_management.py:20] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            if method == "createVial":
                return await self.create_vial(user_id, vial_id, project_id)
            elif method == "prompt":
                return await self.process_prompt(user_id, vial_id, data.get("prompt"), project_id)
            elif method == "task":
                return await self.assign_task(user_id, vial_id, data.get("task"), project_id)
            elif method == "config":
                return await self.update_config(user_id, vial_id, data.get("config"), project_id)
            else:
                error_message = f"Invalid vial method: {method} [vial_management.py:25] [ID:vial_method_error]"
                logger.error(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Vial operation failed: {str(e)} [vial_management.py:30] [ID:vial_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def create_vial(self, user_id: str, vial_id: str, project_id: str) -> dict:
        try:
            model = initialize_model(vial_id)
            model_version = "1.0.0"
            await self.db.query(
                "INSERT INTO vials (vial_id, user_id, status, code, tasks, config, wallet_id, project_id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                [vial_id, user_id, "active", "initial_code", json.dumps([]), json.dumps({"model_version": model_version}), str(uuid.uuid4()), project_id]
            )
            logger.info(f"Vial created: {vial_id} for user {user_id} [vial_management.py:35] [ID:vial_create_success]")
            return {"status": "success", "vial_id": vial_id, "model_version": model_version}
        except Exception as e:
            error_message = f"Vial creation failed: {str(e)} [vial_management.py:40] [ID:vial_create_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def process_prompt(self, user_id: str, vial_id: str, prompt: str, project_id: str) -> dict:
        try:
            model = initialize_model(vial_id)
            input_tensor = torch.tensor([ord(c) for c in prompt[:10]], dtype=torch.float32)
            output = model(input_tensor)
            result = {"output": output.tolist()}
            await self.db.query(
                "INSERT INTO vial_states (vial_id, state, user_id, project_id) VALUES ($1, $2, $3, $4)",
                [vial_id, json.dumps(result), user_id, project_id]
            )
            logger.info(f"Prompt processed for vial {vial_id} [vial_management.py:45] [ID:prompt_success]")
            return {"status": "success", "result": result}
        except Exception as e:
            error_message = f"Prompt processing failed: {str(e)} [vial_management.py:50] [ID:prompt_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def assign_task(self, user_id: str, vial_id: str, task: dict, project_id: str) -> dict:
        try:
            await self.db.query(
                "UPDATE vials SET tasks = tasks || $1::jsonb WHERE vial_id = $2 AND user_id = $3 AND project_id = $4",
                [json.dumps([task]), vial_id, user_id, project_id]
            )
            logger.info(f"Task assigned to vial {vial_id} [vial_management.py:55] [ID:task_success]")
            return {"status": "success", "task": task}
        except Exception as e:
            error_message = f"Task assignment failed: {str(e)} [vial_management.py:60] [ID:task_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def update_config(self, user_id: str, vial_id: str, config: dict, project_id: str) -> dict:
        try:
            await self.db.query(
                "UPDATE vials SET config = $1::jsonb WHERE vial_id = $2 AND user_id = $3 AND project_id = $4",
                [json.dumps(config), vial_id, user_id, project_id]
            )
            logger.info(f"Config updated for vial {vial_id} [vial_management.py:65] [ID:config_success]")
            return {"status": "success", "config": config}
        except Exception as e:
            error_message = f"Config update failed: {str(e)} [vial_management.py:70] [ID:config_error]"
            logger.error(error_message)
            return {"error": error_message}
