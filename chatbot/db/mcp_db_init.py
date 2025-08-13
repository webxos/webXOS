from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_mcp_db():
    try:
        client = MongoClient('mongodb://mongo:27017')
        db = client['mcp_db']
        
        # Create collections
        collections = ['users', 'queries', 'errors', 'wallet', 'vials', 'modules']
        existing_collections = db.list_collection_names()
        
        for collection in collections:
            if collection not in existing_collections:
                db.create_collection(collection)
                logger.info(f"Created collection: {collection}")
                
        # Create indexes
        db.users.create_index("userId", unique=True)
        db.queries.create_index("timestamp")
        db.errors.create_index("timestamp")
        db.wallet.create_index("userId", unique=True)
        db.vials.create_index("id", unique=True)
        db.modules.create_index("name", unique=True)
        
        logger.info("MCP database initialized successfully")
        client.close()
    except Exception as e:
        logger.error(f"Failed to initialize MCP database: {str(e)}")
        raise

if __name__ == "__main__":
    init_mcp_db()
