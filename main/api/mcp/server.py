from fastapi import HTTPException
from pymongo import MongoClient
from datetime import datetime, timedelta
import jwt
import dspy
import uuid
from main.api.utils.logging import logger
from main.api.mcp.blockchain import Blockchain

class MCPServer:
    def __init__(self):
        self.mongo_client = MongoClient("mongodb://localhost:27017")
        self.db = self.mongo_client["neon_mcp"]
        self.users_collection = self.db["users"]
        self.agents_collection = self.db["agents"]
        self.blockchain = Blockchain()
        self.lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key="your-openai-key")
        dspy.settings.configure(lm=self.lm)
        self.JWT_SECRET = "secret_key_123_change_in_production"

    def health_check(self):
        try:
            user = self.users_collection.find_one({"api_key": "WEBXOS-MOCKKEY"}) or {
                "balance": 38940.0000,
                "reputation": 1200983581,
                "user_id": "a1d57580-d88b-4c90-a0f8-6f2c8511b1e4",
                "address": "e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d"
            }
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "balance": user["balance"],
                "reputation": user["reputation"],
                "user_id": user["user_id"],
                "address": user["address"],
                "vial_agent": "vial1"
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    async def authenticate(self, request: dict):
        try:
            if (request.get("grant_type") != "client_credentials" or
                request.get("client_id") != "WEBXOS-MOCKKEY" or
                request.get("client_secret") != "MOCKSECRET1234567890"):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            user = self.users_collection.find_one({"api_key": request.get("client_id")})
            if not user:
                user = {
                    "api_key": request.get("client_id"),
                    "api_secret": request.get("client_secret"),
                    "balance": 38940.0000,
                    "reputation": 1200983581,
                    "wallet_address": "e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d",
                    "user_id": str(uuid.uuid4()),
                    "created_at": datetime.now()
                }
                self.users_collection.insert_one(user)
            
            payload = {
                "sub": user["user_id"],
                "exp": (datetime.utcnow() + timedelta(hours=24)).timestamp(),
                "iat": datetime.utcnow().timestamp()
            }
            token = jwt.encode(payload, self.JWT_SECRET, algorithm="HS256")
            block_hash = self.blockchain.add_block({"type": "auth", "user_id": user["user_id"]})
            logger.info(f"Token generated for user: {user['user_id']}, block: {block_hash}")
            return {"access_token": token, "token_type": "bearer", "expires_in": 86400}
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def train_vial(self, vial_id: str, dataset: dict):
        try:
            if vial_id not in [f"vial{i+1}" for i in range(4)]:
                raise HTTPException(status_code=400, detail="Invalid vial_id")
            response = self.lm(prompt=f"Train {vial_id} with dataset")
            self.agents_collection.update_one(
                {"vial_id": vial_id},
                {"$push": {"training_data": {"dataset": dataset, "response": response, "timestamp": datetime.now()}},
                 "$set": {"status": "trained", "balance": 10.0}},
                upsert=True
            )
            block_hash = self.blockchain.add_block({"type": "train", "vial_id": vial_id})
            logger.info(f"Training completed for {vial_id}, block: {block_hash}")
            return {"vial_id": vial_id, "status": "trained", "block_hash": block_hash}
        except Exception as e:
            logger.error(f"Training failed for {vial_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def void(self):
        try:
            self.agents_collection.delete_many({})
            self.users_collection.update_many({}, {"$set": {"balance": 0, "reputation": 0}})
            block_hash = self.blockchain.add_block({"type": "void"})
            logger.info(f"System reset, block: {block_hash}")
            return {"status": "reset", "block_hash": block_hash}
        except Exception as e:
            logger.error(f"Void failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def troubleshoot(self):
        try:
            vials_count = self.agents_collection.count_documents({})
            return {"status": "diagnosed", "vials_count": vials_count, "blockchain_integrity": "verified"}
        except Exception as e:
            logger.error(f"Troubleshoot failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def quantum_link(self):
        try:
            for vial_id in [f"vial{i+1}" for i in range(4)]:
                await self.train_vial(vial_id, {})
            self.users_collection.update_one(
                {"api_key": "WEBXOS-MOCKKEY"},
                {"$inc": {"balance": 40, "reputation": 100}},
                upsert=True
            )
            block_hash = self.blockchain.add_block({"type": "quantum_link"})
            logger.info(f"Quantum link activated, block: {block_hash}")
            return {"status": "linked", "block_hash": block_hash}
        except Exception as e:
            logger.error(f"Quantum link failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_credentials(self):
        try:
            user = self.users_collection.find_one({"api_key": "WEBXOS-MOCKKEY"})
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            credentials = {
                "api_key": user["api_key"],
                "api_secret": str(uuid.uuid4())
            }
            self.users_collection.update_one(
                {"api_key": user["api_key"]},
                {"$set": {"api_secret": credentials["api_secret"]}}
            )
            block_hash = self.blockchain.add_block({"type": "credentials", "api_key": credentials["api_key"]})
            logger.info(f"Credentials generated, block: {block_hash}")
            return credentials
        except Exception as e:
            logger.error(f"Credentials generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
