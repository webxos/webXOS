# main/server/mcp/auth/mcp_server_auth.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from webauthn import generate_registration_options, verify_registration_response
from webauthn.helpers.structs import PublicKeyCredentialCreationOptions
from pymongo import MongoClient
from datetime import datetime
import os
import oci
from ..utils.error_handler import handle_auth_error
from ..utils.performance_metrics import PerformanceMetrics

app = FastAPI(title="Vial MCP Auth Server")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["vial_mcp"]
users_collection = db["users"]
metrics = PerformanceMetrics()

class User(BaseModel):
    user_id: str
    email: str | None = None
    display_name: str | None = None

class Token(BaseModel):
    access_token: str
    token_type: str

class WebAuthnRegistration(BaseModel):
    credential: dict
    client_data: str
    attestation_object: str

@app.post("/token", response_model=Token)
async def login_for_access_token(user: User):
    with metrics.track_span("auth_login"):
        try:
            oci_config = oci.config.from_file()
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            user_data = {"user_id": user.user_id, "email": user.email, "display_name": user.display_name, "last_login": datetime.utcnow()}
            users_collection.update_one({"user_id": user.user_id}, {"$set": user_data}, upsert=True)
            access_token = metrics.create_access_token(data={"sub": user.user_id})
            return {"access_token": access_token, "token_type": "bearer"}
        except Exception as e:
            handle_auth_error(e)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Authentication failed: {str(e)}")

@app.post("/webauthn/register")
async def webauthn_register(user: User):
    with metrics.track_span("webauthn_register"):
        try:
            registration_options = generate_registration_options(
                rp_id="localhost",
                rp_name="Vial MCP Controller",
                user_id=user.user_id.encode(),
                user_name=user.email or user.user_id,
                user_display_name=user.display_name or user.user_id,
                authenticator_selection={"user_verification": "required"}
            )
            users_collection.update_one(
                {"user_id": user.user_id},
                {"$set": {"webauthn_challenge": registration_options.challenge.hex()}},
                upsert=True
            )
            return PublicKeyCredentialCreationOptions.parse_obj(registration_options.__dict__)
        except Exception as e:
            handle_auth_error(e)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"WebAuthn registration failed: {str(e)}")

@app.post("/webauthn/verify")
async def webauthn_verify(registration: WebAuthnRegistration, user_id: str):
    with metrics.track_span("webauthn_verify"):
        try:
            user = users_collection.find_one({"user_id": user_id})
            if not user or not user.get("webauthn_challenge"):
                raise ValueError("No registration challenge found")
            verified = verify_registration_response(
                credential=registration.credential,
                expected_challenge=bytes.fromhex(user["webauthn_challenge"]),
                expected_origin="http://localhost:3000",
                expected_rp_id="localhost"
            )
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"webauthn_credential": verified.credential_id.hex(), "last_login": datetime.utcnow()}}
            )
            access_token = metrics.create_access_token(data={"sub": user_id})
            return {"access_token": access_token, "token_type": "bearer"}
        except Exception as e:
            handle_auth_error(e)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"WebAuthn verification failed: {str(e)}")

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    with metrics.track_span("read_users_me"):
        try:
            payload = metrics.verify_token(token)
            user = users_collection.find_one({"user_id": payload["sub"]})
            if not user:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
            return {"user_id": payload["sub"], "last_login": user.get("last_login"), "email": user.get("email")}
        except Exception as e:
            handle_auth_error(e)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token verification failed: {str(e)}")
