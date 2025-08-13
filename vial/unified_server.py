from fastapi import FastAPI
import logging
import datetime
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
async def health_check():
    try:
        mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        mongo_client.server_info()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Unified server health check error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Unified server health check error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize database
        from db.mcp_db_init import initialize_db
        initialize_db()
        
        # Validate configuration
        from db.config_validator import ConfigValidator
        validator = ConfigValidator()
        await validator.validate_config("system", {"webxos": 0.0, "transactions": []})
        
        logger.info("Unified server started")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Startup error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
