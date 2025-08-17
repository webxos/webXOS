from config.config import DatabaseConfig
from tools.agent_templates import initialize_model
import torch
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class ModelVersioning:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.project_id = db.project_id

    async def save_model_version(self, user_id: str, vial_id: str, model: torch.nn.Module, version: str) -> dict:
        try:
            model_data = {"state_dict": {k: v.tolist() for k, v in model.state_dict().items()}}
            await self.db.query(
                "INSERT INTO vial_states (vial_id, state, user_id, project_id) VALUES ($1, $2, $3, $4)",
                [vial_id, json.dumps({"model_version": version, "model_data": model_data}), user_id, self.project_id]
            )
            logger.info(f"Model version saved: {vial_id} v{version} [model_versioning.py:20] [ID:model_version_success]")
            return {"status": "success", "vial_id": vial_id, "version": version}
        except Exception as e:
            error_message = f"Model version save failed: {str(e)} [model_versioning.py:25] [ID:model_version_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def load_model_version(self, user_id: str, vial_id: str, version: str) -> dict:
        try:
            result = await self.db.query(
                "SELECT state FROM vial_states WHERE vial_id = $1 AND user_id = $2 AND project_id = $3 AND state->>'model_version' = $4",
                [vial_id, user_id, self.project_id, version]
            )
            if not result:
                error_message = f"Model version not found: {vial_id} v{version} [model_versioning.py:30] [ID:model_version_not_found]"
                logger.error(error_message)
                return {"error": error_message}
            state = json.loads(result[0]["state"])
            model = initialize_model(vial_id)
            state_dict = {k: torch.tensor(v) for k, v in state["model_data"]["state_dict"].items()}
            model.load_state_dict(state_dict)
            logger.info(f"Model version loaded: {vial_id} v{version} [model_versioning.py:35] [ID:model_version_load_success]")
            return {"status": "success", "vial_id": vial_id, "version": version}
        except Exception as e:
            error_message = f"Model version load failed: {str(e)} [model_versioning.py:40] [ID:model_version_load_error]"
            logger.error(error_message)
            return {"error": error_message}
