from pymongo import MongoClient
import os

def init_db():
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["webxos"]
        
        # Create collections if they don't exist
        collections = ["wallets", "logs", "blockchain"]
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                print(f"Created collection: {collection}")
        
        # Create indexes
        db.wallets.create_index("user_id", unique=True)
        db.blockchain.create_index("hash", unique=True)
        db.logs.create_index("timestamp")
        
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {str(e)}")

if __name__ == "__main__":
    init_db()
