from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from ...config.settings import settings
from ...security.authentication import verify_token

router = APIRouter()

class QuantumLinkRequest(BaseModel):
    state: str = "synced"

@router.post("/quantum-link")
async def quantum_link(request: QuantumLinkRequest, token: str = Depends(verify_token)):
    try:
        client = MongoClient(settings.database.url)
        db = client[settings.database.db_name]
        quantum_state = {"qubits": [], "entanglement": request.state}
        result = db.quantum_states.update_one(
            {"user_id": token["sub"]},
            {"$set": {"quantum_state": quantum_state}},
            upsert=True
        )
        client.close()
        return {"message": "Quantum sync complete", "quantumState": quantum_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
