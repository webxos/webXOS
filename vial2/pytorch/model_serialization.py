import torch
from ..error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

def serialize_model(model, path: str):
    try:
        torch.save(model.state_dict(), path)
        return {"status": "success", "path": path}
    except Exception as e:
        error_logger.log_error("model_serialization", str(e), str(e.__traceback__))
        logger.error(f"Model serialization failed: {str(e)}")
        raise

def deserialize_model(model_class, path: str):
    try:
        model = model_class()
        model.load_state_dict(torch.load(path))
        return model
    except Exception as e:
        error_logger.log_error("model_deserialization", str(e), str(e.__traceback__))
        logger.error(f"Model deserialization failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #pytorch #model_serialization #neon_mcp
