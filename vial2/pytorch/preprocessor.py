import torch
from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def preprocess_data(self, vial_id: str, raw_data: list):
        try:
            tensor_data = torch.tensor(raw_data, device=self.device, dtype=torch.float32)
            normalized = torch.nn.functional.normalize(tensor_data, dim=0)
            scaled = (normalized - normalized.mean()) / normalized.std()
            return {"status": "success", "vial_id": vial_id, "preprocessed_data": scaled.tolist()}
        except Exception as e:
            error_logger.log_error("preprocessor", f"Data preprocessing failed for {vial_id}: {str(e)}", str(e.__traceback__))
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

data_preprocessor = DataPreprocessor()

# xAI Artifact Tags: #vial2 #pytorch #preprocessor #neon_mcp
