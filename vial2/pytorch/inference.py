import torch
from ..pytorch.models import QuantumAgentModel
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def run_inference(vial_id: str, input_data: list):
    try:
        model = QuantumAgentModel()
        model.load_state_dict(torch.load(f"models/{vial_id}/versions/latest.pth"))
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(input_data, dtype=torch.float32, device=model.device)
            outputs = model(inputs)
        return {"status": "success", "vial_id": vial_id, "predictions": outputs.tolist()}
    except Exception as e:
        error_logger.log_error("inference", f"Inference failed for {vial_id}: {str(e)}", str(e.__traceback__))
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #pytorch #inference #neon_mcp
