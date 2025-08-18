import torch
from ..pytorch.models import QuantumAgentModel
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def evaluate_model(vial_id: str, test_data: list, labels: list):
    try:
        model = QuantumAgentModel()
        model.load_state_dict(torch.load(f"models/{vial_id}/versions/latest.pth"))
        model.eval()
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            inputs = torch.tensor(test_data, dtype=torch.float32, device=model.device)
            targets = torch.tensor(labels, dtype=torch.float32, device=model.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        return {"status": "success", "vial_id": vial_id, "loss": loss.item()}
    except Exception as e:
        error_logger.log_error("model_evaluation", f"Model evaluation failed for {vial_id}: {str(e)}", str(e.__traceback__))
        logger.error(f"Model evaluation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #pytorch #model_evaluation #neon_mcp
