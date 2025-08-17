from fastapi import APIRouter, Depends
from utils.auth import verify_token
import uuid

router = APIRouter()

@router.get("/generate-credentials")
async def generate_credentials(token: str = Depends(verify_token)):
    return {"key": f"key_{uuid.uuid4()}", "secret": f"secret_{uuid.uuid4()}"}
