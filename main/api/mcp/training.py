import dspy
from datetime import datetime
from main.api.utils.logging import logger
from main.api.mcp.wallet import WalletManager

class VialTrainer:
    def __init__(self, mongo_url="mongodb://localhost:27017", openai_api_key="your-openai-key"):
        self.mongo_client = MongoClient(mongo_url)
        self.db = self.mongo_client["neon_mcp"]
        self.agents_collection = self.db["agents"]
        self.lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
        dspy.settings.configure(lm=self.lm)
        self.wallet_manager = WalletManager(mongo_url)

    def train_vial(self, vial_id: str, user_id: str, dataset: dict):
        """Train a vial and reward $WEBXOS via PoW."""
        try:
            if vial_id not in [f"vial{i+1}" for i in range(4)]:
                raise ValueError(f"Invalid vial_id: {vial_id}")
            response = self.lm(prompt=f"Train {vial_id} with dataset: {dataset}")
            reward = 10.0  # PoW reward
            self.agents_collection.update_one(
                {"vial_id": vial_id, "user_id": user_id},
                {
                    "$set": {"status": "trained", "balance": reward},
                    "$push": {
                        "training_data": {
                            "dataset": dataset,
                            "response": response,
                            "timestamp": datetime.now()
                        }
                    }
                },
                upsert=True
            )
            self.wallet_manager.update_balance(user_id, reward, f"train_{vial_id}")
            logger.info(f"Vial {vial_id} trained for user {user_id}, rewarded {reward} $WEBXOS")
            return {"vial_id": vial_id, "status": "trained", "reward": reward}
        except Exception as e:
            logger.error(f"Training failed for {vial_id}: {str(e)}")
            raise
