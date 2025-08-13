from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionError
import logging
import asyncio

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.uri = "mongodb://localhost:27017"
        self.db_name = "vial_mcp"

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.db_name]
            await self.client.admin.command('ping')
            logger.info("Connected to MongoDB")
        except ConnectionError as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            raise

    async def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def check_health(self):
        try:
            await self.client.admin.command('ping')
            return "healthy"
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return "unhealthy"

    async def store_prompt(self, vial_id: str, prompt: str):
        try:
            await self.db.prompts.insert_one({"vial_id": vial_id, "prompt": prompt, "timestamp": datetime.now()})
        except Exception as e:
            logger.error(f"Failed to store prompt: {str(e)}")
            raise

    async def store_task(self, vial_id: str, task: str):
        try:
            await self.db.tasks.insert_one({"vial_id": vial_id, "task": task, "timestamp": datetime.now()})
        except Exception as e:
            logger.error(f"Failed to store task: {str(e)}")
            raise

    async def store_config(self, vial_id: str, key: str, value: str):
        try:
            await self.db.configs.update_one(
                {"vial_id": vial_id, "key": key},
                {"$set": {"value": value, "timestamp": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to store config: {str(e)}")
            raise

    async def store_quantum_state(self, vial_id: str, quantum_state: dict):
        try:
            await self.db.quantum_states.update_one(
                {"vial_id": vial_id},
                {"$set": {"state": quantum_state, "timestamp": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to store quantum state: {str(e)}")
            raise

    async def store_error(self, error: str):
        try:
            await self.db.errors.insert_one({"error": error, "timestamp": datetime.now()})
        except Exception as e:
            logger.error(f"Failed to store error: {str(e)}")
            raise
