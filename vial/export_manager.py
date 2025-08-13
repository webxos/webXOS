from fastapi import HTTPException
import pymongo
import logging
import datetime
import os
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["mcp_db"]

class ExportManager:
    def __init__(self):
        self.output_dir = "vial"

    async def export_to_markdown(self, user_id: str, data: Dict[str, Any], wallet: Dict[str, Any], filename: str) -> str:
        try:
            timestamp = datetime.datetime.utcnow().isoformat()
            markdown_content = f"# Export for {user_id}\n\n**Timestamp**: {timestamp}\n\n**Data**:\n{json.dumps(data, indent=2)}\n"
            export_path = os.path.join(self.output_dir, f"vial_wallet_export_{user_id}_{timestamp}.md")
            
            with open(export_path, "w") as f:
                f.write(markdown_content)
            
            # Update wallet
            wallet["transactions"].append({
                "type": "export",
                "filename": export_path,
                "timestamp": timestamp
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + float(os.getenv("WALLET_INCREMENT", 0.0001))
            db.collection("wallet").update_one(
                {"user_id": user_id},
                {"$set": {"wallet": wallet}, "$push": {"transactions": wallet["transactions"][-1]}},
                upsert=True
            )
            
            # Log export
            db.collection("export_logs").insert_one({
                "user_id": user_id,
                "filename": export_path,
                "timestamp": timestamp,
                "wallet": wallet
            })
            
            return export_path
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{timestamp}]** Export error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))
