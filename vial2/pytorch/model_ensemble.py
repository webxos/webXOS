import torch
from ..pytorch.models import QuantumAgentModel
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class ModelEnsemble:
    def __init__(self, vial_id: str, model_versions: list):
        self.models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for version in model_versions:
            model = QuantumAgentModel()
            model.load_state_dict(torch.load(f"models/{vial_id}/versions/{version}.pth"))
            model.eval()
            model.to(self.device)
            self.models.append(model)

    async def ensemble_predict(self, vial_id: str, input_data: list):
        try:
            inputs = torch.tensor(input_data, dtype=torch.float32, device=self.device)
            predictions = []
            with torch.no_grad():
                for model in self.models:
                    outputs = model(inputs)
                    predictions.append(outputs)
            ensemble_output = torch.mean(torch.stack(predictions), dim=0)
            return {"status": "success", "vial_id": vial_id, "predictions": ensemble_output.tolist()}
        except Exception as e:
            error_logger.log_error("model_ensemble", f"Ensemble prediction failed for {vial_id}: {str(e)}", str(e.__traceback__))
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

# xAI Artifact Tags: #vial2 #pytorch #model_ensemble #neon_mcp
