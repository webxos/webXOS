import torch
from ..pytorch.models import QuantumAgentModel
from ..error_logging.error_log import error_logger
import logging
import pickle

logger = logging.getLogger(__name__)

async def serialize_model(vial_id: str, model: QuantumAgentModel):
    try:
        serialized_data = pickle.dumps(model.state_dict())
        with open(f"models/{vial_id}/serialized_model.pkl", "wb") as f:
            f.write(serialized_data)
        return {"status": "success", "vial_id": vial_id}
    except Exception as e:
        error_logger.log_error("model_serialization", f"Model serialization failed for {vial_id}: {str(e)}", str(e.__traceback__))
        logger.error(f"Model serialization failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def deserialize_model(vial_id: str):
    try:
        with open(f"models/{vial_id}/serialized_model.pkl", "rb") as f:
            serialized_data = pickle.load(f)
        model = QuantumAgentModel()
        model.load_state_dict(serialized_data)
        model.eval()
        return {"status": "success", "vial_id": vial_id}
    except Exception as e:
        error_logger.log_error("model_serialization", f"Model deserialization failed for {vial_id}: {str(e)}", str(e.__traceback__))
        logger.error(f"Model deserialization failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #pytorch #model_serialization #neon_mcp
