import os
from pymongo import MongoClient
from pymongo.errors import ConnectionError
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(filename='/db/errorlog.md', level=logging.INFO, format='## [%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')

def init_db():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client['vial_mcp']
        client.admin.command('ping')
        collections = ['errors', 'gateway_logs']
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                logger.info(f"Created collection: {collection}")
        return db
    except ConnectionError as e:
        logger.error(f"Database initialization failed: MongoDB connection error: {str(e)}")
        raise Exception(f"Database initialization failed: {str(e)}")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise Exception(f"Database initialization failed: {str(e)}")

if __name__ == "__main__":
    init_db()
