from fastapi import HTTPException
from ..database import get_db
from ..pytorch.quantum_link import QuantumLink
from ..error_logging.error_log import error_logger
import logging
import torch

logger = logging.getLogger(__name__)

async def manage_agent(vial_id: str, action: str, data: dict = None):
    try:
        db = await get_db()
        if action == "train":
            model_class = data.get("model_class")
            model_data = data.get("model_data")
            if not model_class or not model_data:
                raise HTTPException(status_code=400, detail="Model class or data missing")
            quantum_link = QuantumLink()
            model = await quantum_link.train_model(model_class, torch.tensor(model_data))
            await db.execute(
                "UPDATE vials SET status=$1, code=$2 WHERE vial_id=$3",
                "running", str(model), vial_id
            )
            return {"status": "success", "vial_id": vial_id}
        elif action == "status":
            result = await db.execute("SELECT status FROM vials WHERE vial_id=$1", vial_id)
            return result[0] if result else {"status": "not found"}
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
    except Exception as e:
        error_logger.log_error("manage_agent", str(e), str(e.__traceback__))
        logger.error(f"Agent management failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #agents #pytorch #neon_mcp
