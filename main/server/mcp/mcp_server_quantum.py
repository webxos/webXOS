import logging
from fastapi import HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from .auth_manager import AuthManager
from .quantum_simulator import QuantumSimulator
from .db.db_manager import DatabaseManager
import os

logger = logging.getLogger(__name__)

class QuantumRequest(BaseModel):
    vial_id: str
    wallet_id: str
    prompt: str
    db_type: str = "postgres"

class MCPQuantumHandler:
    """Handles quantum processing requests for Vial MCP."""
    def __init__(self):
        """Initialize MCPQuantumHandler with database and simulator."""
        self.auth_manager = AuthManager()
        self.quantum_simulator = QuantumSimulator()
        postgres_config = {
            "host": os.getenv("POSTGRES_HOST", "postgresdb"),
            "port": int(os.getenv("POSTGRES_DOCKER_PORT", 5432)),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "database": os.getenv("POSTGRES_DB", "vial_mcp")
        }
        mysql_config = {
            "host": os.getenv("MYSQL_HOST", "mysqldb"),
            "port": int(os.getenv("MYSQL_DOCKER_PORT", 3306)),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_ROOT_PASSWORD", "mysql"),
            "database": os.getenv("MYSQL_DB", "vial_mcp")
        }
        mongo_config = {
            "host": os.getenv("MONGO_HOST", "mongodb"),
            "port": int(os.getenv("MONGO_DOCKER_PORT", 27017)),
            "username": os.getenv("MONGO_USER", "mongo"),
            "password": os.getenv("MONGO_PASSWORD", "mongo")
        }
        self.db_manager = DatabaseManager(postgres_config, mysql_config, mongo_config)
        self.db_manager.connect()
        logger.info("MCPQuantumHandler initialized")

    async def process_quantum(self, request: QuantumRequest, access_token: str) -> dict:
        """Process a quantum task and store the result.

        Args:
            request (QuantumRequest): Request with vial_id, wallet_id, prompt, and db_type.
            access_token (str): OAuth access token.

        Returns:
            dict: Quantum processing result and state ID.

        Raises:
            HTTPException: If authentication or processing fails.
        """
        try:
            payload = self.auth_manager.verify_token(access_token)
            if payload["wallet_id"] != request.wallet_id:
                logger.warning(f"Unauthorized wallet access: {request.wallet_id}")
                raise HTTPException(status_code=401, detail="Unauthorized wallet access")

            # Simulate quantum processing
            quantum_state = self.quantum_simulator.simulate(request.prompt)

            # Store quantum state in the specified database
            if request.db_type == "postgres":
                with self.db_manager.postgres_conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO quantum_states (vial_id, state, timestamp, wallet_id) VALUES (%s, %s, %s, %s) RETURNING id",
                        (request.vial_id, quantum_state, datetime.now(), request.wallet_id)
                    )
                    state_id = cursor.fetchone()[0]
                    self.db_manager.postgres_conn.commit()
            elif request.db_type == "mysql":
                with self.db_manager.mysql_conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO quantum_states (vial_id, state, timestamp, wallet_id) VALUES (%s, %s, %s, %s)",
                        (request.vial_id, quantum_state, datetime.now(), request.wallet_id)
                    )
                    state_id = cursor.lastrowid
                    self.db_manager.mysql_conn.commit()
            elif request.db_type == "mongo":
                state = {
                    "vial_id": request.vial_id,
                    "state": quantum_state,
                    "timestamp": datetime.now(),
                    "wallet_id": request.wallet_id
                }
                state_id = str(self.db_manager.mongo_client.vial_mcp.quantum_states.insert_one(state).inserted_id)
            else:
                raise ValueError("Invalid database type")

            logger.info(f"Quantum state {state_id} stored for vial {request.vial_id}, wallet {request.wallet_id}")
            return {"status": "success", "state_id": state_id, "quantum_state": quantum_state}
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Quantum processing failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [MCPQuantumHandler] Quantum processing failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Quantum processing failed: {str(e)}")
