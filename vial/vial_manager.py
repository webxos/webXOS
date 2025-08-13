import logging
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VialManager:
    def __init__(self):
        self.db = MongoClient('mongodb://mongo:27017')['mcp_db']

    def validate_vials(self, vials: dict) -> bool:
        try:
            if len(vials) != 4:
                logger.error(f"Invalid number of vials: expected 4, found {len(vials)}")
                return False
            for vial_id, vial_data in vials.items():
                if not vial_data.get('status') or not vial_data.get('script'):
                    logger.error(f"Invalid vial data for {vial_id}")
                    return False
                if not vial_data.get('wallet_hash') or not re.match(r'^[0-9a-f]{64}$', vial_data['wallet_hash']):
                    logger.error(f"Invalid wallet hash for {vial_id}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Vial validation error: {str(e)}")
            return False

    def update_vial(self, vial_id: str, data: dict) -> bool:
        try:
            self.db.vials.update_one({"id": vial_id}, {"$set": data}, upsert=True)
            logger.info(f"Updated vial: {vial_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update vial {vial_id}: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Vial update error: {str(e)}\n")
            return False
