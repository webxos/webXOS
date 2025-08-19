from langchain.memory import ConversationBufferMemory
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.memories = {}

    def get_memory(self, vial_id: str):
        try:
            if vial_id not in self.memories:
                self.memories[vial_id] = ConversationBufferMemory()
            logger.info(f"Retrieved memory for vial {vial_id}")
            return self.memories[vial_id]
        except Exception as e:
            error_logger.log_error("memory_get", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
            logger.error(f"Memory retrieval failed: {str(e)}")
            raise

memory_manager = MemoryManager()

# xAI Artifact Tags: #vial2 #mcp #langchain #memory #manager #neon_mcp
