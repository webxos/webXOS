import torch
import torch.nn.utils.prune as prune
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

def prune_model(model, amount: float = 0.3):
    try:
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
        return model
    except Exception as e:
        error_logger.log_error("model_pruning", str(e), str(e.__traceback__))
        logger.error(f"Model pruning failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #pytorch #model_pruning #neon_mcp
