import pymongo
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_db():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017")
        db = client["mcp_db"]
        
        # Create collections
        collections = ["wallet", "sync_logs", "auth_logs", "monitor_logs", "metrics_logs", "config_logs", "health_logs", "audit_logs"]
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                logger.info(f"Created collection: {collection}")
        
        # Create indexes
        db.collection("wallet").create_index("user_id", unique=True)
        db.collection("sync_logs").create_index("timestamp")
        db.collection("auth_logs").create_index("user_id")
        db.collection("monitor_logs").create_index("timestamp")
        db.collection("metrics_logs").create_index("timestamp")
        db.collection("config_logs").create_index("timestamp")
        db.collection("health_logs").create_index("timestamp")
        db.collection("audit_logs").create_index("timestamp")
        
        logger.info("MongoDB initialization completed")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"DB initialization error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** DB initialization error: {str(e)}\n")
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    initialize_db()
