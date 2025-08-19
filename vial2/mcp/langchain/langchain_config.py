import os
from langchain.adapters.base import BaseAdapter
from ..error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class LangChainConfig:
    def __init__(self):
        self.adapters = {}
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", "")
        }

    def register_adapter(self, adapter_type: str, adapter: BaseAdapter):
        try:
            if not self.api_keys.get(adapter_type):
                raise ValueError(f"No API key for {adapter_type}")
            self.adapters[adapter_type] = adapter
            logger.info(f"Registered LangChain adapter for {adapter_type}")
        except Exception as e:
            error_logger.log_error("langchain_config_register", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"LangChain adapter registration failed: {str(e)}")
            raise

    def get_adapter(self, adapter_type: str):
        return self.adapters.get(adapter_type)

langchain_config = LangChainConfig()

# xAI Artifact Tags: #vial2 #mcp #langchain #config #neon #neon_mcp
