import asyncio
import json
import logging
import asyncpg
from config.config import DatabaseConfig
from lib.security import SecurityHandler

logger = logging.getLogger(__name__)

class ComputeCtl:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security = SecurityHandler(db)
        self.project_id = db.project_id
        self.connection_string = "postgresql://cloud_admin@localhost/postgres"

    async def initialize_compute(self, user_id: str, spec: dict) -> dict:
        try:
            if spec.get("project_id") != self.project_id:
                error_message = f"Invalid project ID: {spec.get('project_id')} [compute_ctl.py:15] [ID:project_error]"
                logger.error(error_message)
                return {"error": error_message}
            
            # Initialize data directory
            compute_id = spec.get("compute_id", "compute1")
            await self.db.query(
                "INSERT INTO computes (compute_id, user_id, project_id, state, spec) VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (compute_id) UPDATE SET state = $4, spec = $5",
                [compute_id, user_id, self.project_id, "ConfigurationPending", json.dumps(spec)]
            )

            # Sync safekeepers and get LSN
            lsn = await self.sync_safekeepers(compute_id)
            
            # Get basebackup
            await self.get_basebackup(compute_id, lsn)
            
            # Start Postgres
            async with asyncpg.create_pool(self.connection_string) as pool:
                async with pool.acquire() as conn:
                    await conn.execute("SELECT 1;")  # Check Postgres readiness
            
            # Configure roles and databases
            await self.configure_roles(compute_id, spec.get("roles", []))
            
            await self.db.query(
                "UPDATE computes SET state = $1 WHERE compute_id = $2 AND project_id = $3",
                ["Running", compute_id, self.project_id]
            )
            await self.security.log_action(user_id, "compute_init", {"compute_id": compute_id, "state": "Running"})
            logger.info(f"Compute initialized: {compute_id} [compute_ctl.py:30] [ID:compute_init_success]")
            return {"status": "success", "compute_id": compute_id, "state": "Running"}
        except Exception as e:
            await self.db.query(
                "UPDATE computes SET state = $1 WHERE compute_id = $2 AND project_id = $3",
                ["Failed", compute_id, self.project_id]
            )
            error_message = f"Compute initialization failed: {str(e)} [compute_ctl.py:35] [ID:compute_init_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, "compute_init", error_message)
            return {"error": error_message}

    async def sync_safekeepers(self, compute_id: str) -> str:
        # Simulated safekeeper sync
        return "0/0"

    async def get_basebackup(self, compute_id: str, lsn: str):
        # Simulated basebackup from pageserver
        pass

    async def configure_roles(self, compute_id: str, roles: list):
        async with asyncpg.create_pool(self.connection_string) as pool:
            async with pool.acquire() as conn:
                for role in roles:
                    await conn.execute(f"CREATE ROLE {role['name']} WITH LOGIN PASSWORD '{role['password']}';")
