import torch
from fastapi import HTTPException
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def load_data(self, vial_id: str, data: list):
        try:
            tensor_data = torch.tensor(data, device=self.device, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(tensor_data)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return {"status": "success", "vial_id": vial_id, "batches": len(loader)}
        except Exception as e:
            error_logger.log_error("data_loader", f"Data loading failed for {vial_id}: {str(e)}", str(e.__traceback__))
            logger.error(f"Data loading failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

data_loader = DataLoader()

# xAI Artifact Tags: #vial2 #pytorch #data_loader #neon_mcp
