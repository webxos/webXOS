from fastapi import APIRouter, Depends, HTTPException
from pymongo import MongoClient
from ...config.settings import settings
from ...security.authentication import verify_token

router = APIRouter()

@router.post("/quantum-link")
async def quantum_link(data: dict, token: str = Depends(verify_token)):
    try:
        client = MongoClient(settings.database.url)
        db = client[settings.database.db_name]
        result = db.quantum_states.update_one(
            {"user_id": token["sub"]},
            {"$set": {"quantum_state": {"qubits": [], "entanglement": "synced"}}},
            upsert=True
        )
        client.close()
        return {"message": "Quantum sync complete", "quantumState": {"qubits": [], "entanglement": "synced"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
