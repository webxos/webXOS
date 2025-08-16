from motor.motor_asyncio import AsyncIOMotorClient
from ...config.settings import settings
from ...utils.logging import log_error, log_info

class DatabaseResources:
    client: AsyncIOMotorClient = None
    db = None

    async def connect(self):
        """Connect to MongoDB with connection pooling."""
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI, maxPoolSize=50)
            self.db = self.client.get_database()
            await self.db.command("ping")
            log_info("MongoDB connected successfully")
        except Exception as e:
            log_error(f"MongoDB connection failed: {str(e)}")
            raise

    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            log_info("MongoDB disconnected")

    async def get_user_wallet(self, user_id: str):
        """Retrieve user wallet from MongoDB."""
        try:
            wallet = await self.db.wallets.find_one({"user_id": user_id})
            if not wallet:
                wallet = {
                    "user_id": user_id,
                    "balance": 0.0,
                    "wallet_key": f"wk_{user_id}_{np.random.bytes(16).hex()}",
                    "address": f"addr_{user_id}_{np.random.bytes(8).hex()}",
                    "reputation": 0
                }
                await self.db.wallets.insert_one(wallet)
            return wallet
        except Exception as e:
            log_error(f"Wallet fetch failed for {user_id}: {str(e)}")
            raise

db_resources = DatabaseResources()
