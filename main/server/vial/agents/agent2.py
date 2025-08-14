import logging,sqlite3,json
from datetime import datetime
from .quantum_simulator import QuantumSimulator

logger=logging.getLogger(__name__)

class CogniTALLMwareAgent:
    """CogniTALLMwareAgent handles quantum processing tasks with wallet verification."""
    def __init__(self):
        """Initialize CogniTALLMwareAgent with QuantumSimulator."""
        self.quantum_simulator=QuantumSimulator()
        logger.info("CogniTALLMwareAgent initialized")

    async def process_quantum_task(self,vial_id:str,prompt:str,wallet_id:str)->dict:
        """Process a quantum task for a vial.

        Args:
            vial_id (str): ID of the vial.
            prompt (str): Input prompt for quantum processing.
            wallet_id (str): Wallet ID for verification.

        Returns:
            dict: Quantum state result.

        Raises:
            Exception: If quantum processing fails.
        """
        try:
            result=await self.quantum_simulator.process_quantum_link(vial_id,prompt)
            with sqlite3.connect("/app/vial_mcp.db") as conn:
                cursor=conn.cursor()
                cursor.execute("INSERT OR REPLACE INTO quantum_states (vial_id,state,wallet_id,timestamp) VALUES (?,?,?,?)",
                              (vial_id,json.dumps(result),wallet_id,datetime.now().isoformat()))
                conn.commit()
            logger.info(f"Quantum task processed for vial {vial_id} by wallet {wallet_id}")
            return {"status":"success","quantum_state":result,"wallet_id":wallet_id}
        except Exception as e:
            logger.error(f"Quantum task failed for vial {vial_id} by wallet {wallet_id}: {str(e)}")
            raise Exception(f"Quantum task failed: {str(e)}")