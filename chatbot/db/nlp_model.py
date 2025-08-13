import logging
import datetime
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPModel:
    def __init__(self):
        try:
            self.model = pipeline("text-classification", model="distilbert-base-uncased")
            logger.info("Initialized NLP model")
        except Exception as e:
            logger.error(f"NLP model initialization error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** NLP model initialization error: {str(e)}\n")
            raise

    def classify_query(self, query: str) -> dict:
        try:
            result = self.model(query)
            logger.info(f"Classified query: {query} -> {result}")
            return result[0]
        except Exception as e:
            logger.error(f"Query classification error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Query classification error: {str(e)}\n")
            raise
