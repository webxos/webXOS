from fastapi import APIRouter, Depends
from utils.auth import verify_token

router = APIRouter()

@router.post("/void")
async def void_transaction(token: str = Depends(verify_token)):
    return {"status": "success", "message": "Transaction voided"}

@router.post("/export")
async def export_data(token: str = Depends(verify_token)):
    return {"status": "success", "file": "vial_wallet_export_2025-08-17T00-22-00Z.md"}

@router.post("/import")
async def import_data(file: str, token: str = Depends(verify_token)):
    return {"status": "success", "message": f"Imported {file}"}
