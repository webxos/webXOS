import torch
from ..error_logging.error_log import error_logger
from ..pytorch.models import QuantumAgentModel
import logging
import os

logger = logging.getLogger(__name__)

async def save_model_version(vial_id: str, model: QuantumAgentModel, version: str):
    try:
        version_dir = f"models/{vial_id}/versions"
        os.makedirs(version_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{version_dir}/{version}.pth")
        return {"status": "success", "vial_id": vial_id, "version": version}
    except Exception as e:
        error_logger.log_error("model_versioning", f"Model version save failed for {vial_id}: {str(e)}", str(e.__traceback__))
        logger.error(f"Model version save failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def load_model_version(vial_id: str, version: str):
    try:
        model = QuantumAgentModel()
        model.load_state_dict(torch.load(f"models/{vial_id}/versions/{version}.pth"))
        model.eval()
        return {"status": "success", "vial_id": vial_id, "version": version}
    except Exception as e:
        error_logger.log_error("model_versioning", f"Model version load failed for {vial_id}: {str(e)}", str(e.__traceback__))
        logger.error(f"Model version load failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #pytorch #model_versioning #neon_mcp
