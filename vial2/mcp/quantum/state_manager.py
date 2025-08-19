from ..quantum.quantum_engine import quantum_engine
from ..database.neon_connection import neon_db
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class QuantumStateManager:
    async def save_state(self, vial_id: str):
        try:
            state = await quantum_engine.measure_qubits(vial_id)
            query = "INSERT INTO vial_logs (vial_id, event_type, event_data) VALUES ($1, $2, $3)"
            await neon_db.execute(query, vial_id, "quantum_state", {"state": state})
            logger.info(f"Saved quantum state for vial {vial_id}")
            return state
        except Exception as e:
            error_logger.log_error("quantum_state_save", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params={vial_id})
            logger.error(f"Quantum state save failed: {str(e)}")
            raise

quantum_state_manager = QuantumStateManager()

# xAI Artifact Tags: #vial2 #mcp #quantum #state #manager #neon_mcp
