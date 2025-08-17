from fastapi import APIRouter, HTTPException
from ...utils.logging import log_error, log_info
import dspy
import torch

router = APIRouter()

@router.get("/quantum-link")
async def quantum_link():
    try:
        dspy_model = dspy.LM('gpt-3.5-turbo')
        quantum_state = dspy_model.predict({"input": "Generate quantum link state"}).output
        tensor = torch.rand(4, 4)
        result = {"status": "success", "quantum_state": quantum_state, "tensor_output": tensor.tolist()}
        log_info("Quantum link executed successfully")
        return result
    except Exception as e:
        log_error(f"Traceback: Quantum link failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quantum link error: {str(e)}")
