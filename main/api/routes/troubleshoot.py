from fastapi import APIRouter, HTTPException
from ...utils.logging import log_error, log_info
import dspy

router = APIRouter()

@router.get("/troubleshoot")
async def troubleshoot():
    try:
        dspy_model = dspy.LM('gpt-3.5-turbo')
        diagnosis = dspy_model.predict({"input": "Diagnose MCP Gateway issues"}).output
        log_info("Troubleshooting executed successfully")
        return {"status": "success", "diagnosis": diagnosis}
    except Exception as e:
        log_error(f"Traceback: Troubleshooting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Troubleshoot error: {str(e)}")
