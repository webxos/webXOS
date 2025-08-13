import logging
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    try:
        client = MongoClient('mongodb://mongo:27017', serverSelectionTimeoutMS=5000)
        db = client['mcp_db']
        
        # Create collections
        db.create_collection("users")
        db.create_collection("wallet")
        db.create_collection("vials")
        db.create_collection("queries")
        db.create_collection("errors")
        db.create_collection("modules")
        
        # Create indexes
        db.users.create_index("userId", unique=True)
        db.users.create_index("apiKey")
        db.wallet.create_index("userId")
        db.vials.create_index("id")
        db.queries.create_index("userId")
        db.errors.create_index("timestamp")
        db.modules.create_index("userId")
        
        # Initialize default data
        default_user = {
            "userId": "default_user",
            "apiKey": "default_key",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        db.users.update_one({"userId": "default_user"}, {"$set": default_user}, upsert=True)
        
        logger.info("MongoDB initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        with open("vial/errorlog.md", "a") as f:
            f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Database initialization error: {str(e)}\n")
        return False

if __name__ == "__main__":
    init_db()
