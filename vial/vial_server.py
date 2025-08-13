from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from vial.auth_manager import AuthManager
from vial.vial_manager import VialManager
from vial.webxos_wallet import WebXOSWallet
import logging

app = FastAPI()
security = HTTPBearer()
auth_manager = AuthManager()
vial_manager = VialManager()
wallet = WebXOSWallet()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/api/vial/update", dependencies=[Depends(verify_token)])
async def update_vial(vial_id: str, data: dict, user: dict = Depends(verify_token)):
    try:
        if not vial_manager.validate_vials({vial_id: data}):
            raise HTTPException(status_code=400, detail="Invalid vial data")
        success = vial_manager.update_vial(vial_id, data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update vial")
        logger.info(f"Vial {vial_id} updated for user: {user['userId']}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Vial update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vial/wallet", dependencies=[Depends(verify_token)])
async def get_vial_wallet(vial_id: str, user: dict = Depends(verify_token)):
    try:
        balance = wallet.get_balance(vial_id)
        return {"vial_id": vial_id, "balance": balance}
    except Exception as e:
        logger.error(f"Wallet retrieval error for vial {vial_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
