import torch
import torch.nn.utils.prune as prune
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

def quantize_model(model):
    try:
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.3)
                prune.remove(module, 'weight')
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        error_logger.log_error("model_quantization", str(e), str(e.__traceback__))
        logger.error(f"Model quantization failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #pytorch #model_quantization #neon_mcp
