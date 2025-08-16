import tensorflow as tf
from ...utils.logging import log_error, log_info
from ...config.redis_config import get_redis
import numpy as np

class TensorFlowServe:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Mock model load (replace with actual TensorFlow Serving model)
        log_info("TensorFlow model initialized")

    async def detect_fraud(self, transaction_data: dict, user_id: str, redis=Depends(get_redis)):
        """Detect fraud in transaction using TensorFlow."""
        try:
            amount = transaction_data.get("amount", 0.0)
            recipient_len = len(transaction_data.get("recipient", ""))
            input_data = np.array([[amount, recipient_len, np.random.random()]])
            fraud_score = self.model.predict(input_data)[0][0]
            
            result = {"fraud_score": float(fraud_score), "is_fraudulent": fraud_score > 0.7}
            await redis.set(f"fraud:{user_id}:{hash(str(transaction_data))}", json.dumps(result), ex=3600)
            log_info(f"Fraud detection for user {user_id}: score {fraud_score}")
            return result
        except Exception as e:
            log_error(f"Fraud detection failed for {user_id}: {str(e)}")
            raise
