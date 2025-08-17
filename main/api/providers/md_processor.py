from pymongo import MongoClient
from config.mcp_config import MCP_CONFIG

client = MongoClient(MCP_CONFIG["MONGO_URL"])
db = client["webxos_mcp"]

def store_data(collection: str, data: dict):
    db[collection].insert_one(data)

def retrieve_data(collection: str, query: dict):
    return db[collection].find_one(query)
