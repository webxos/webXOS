import torch
import torch.nn.utils.prune as prune
from ..pytorch.models import QuantumAgentModel
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

async def prune_model(vial_id: str, pruning_rate: float = 0.3):
    try:
        model = QuantumAgentModel()
        model.load_state_dict(torch.load(f"models/{vial_id}/versions/latest.pth"))
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=pruning_rate)
        torch.save(model.state_dict(), f"models/{vial_id}/versions/pruned.pth")
        return {"status": "success", "vial_id": vial_id, "pruning_rate": pruning_rate}
    except Exception as e:
        error_logger.log_error("model_pruning", f"Model pruning failed for {vial_id}: {str(e)}", str(e.__traceback__))
        logger.error(f"Model pruning failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #pytorch #model_pruning #neon_mcp
