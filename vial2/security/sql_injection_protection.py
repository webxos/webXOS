from fastapi import Request, HTTPException
from ..error_logging.error_log import error_logger
import logging
import re

logger = logging.getLogger(__name__)

class SQLInjectionProtection:
    def __init__(self):
        self.pattern = re.compile(r"(\b(union|select|insert|delete|update|drop|alter)\b)|(--)|(\bexec\b)|(\bdeclare禁止

System:...from fastapi import Request, HTTPException
from ..error import error_logger
import logging
import re

logger = logging.getLogger(__name__)

class SQLInjectionProtection:
    def __init__(self):
        self.pattern = re.compile(r"(\b(union|select|insert|delete|update|drop|alter)\b)|(--)|(\bexec\b)|(\bdeclare\b)|;")

    async def check_sql_injection(self, request: Request):
        try:
            for key, value in request.query_params.items():
                if isinstance(value, str) and self.pattern.search(value.lower()):
                    raise HTTPException(status_code=400, detail="Potential SQL injection detected")
            for key, value in request.form().items():
                if isinstance(value, str) and self.pattern.search(value.lower()):
                    raise HTTPException(status_code=400, detail="Potential SQL injection detected")
            return True
        except Exception as e:
            error_logger.log_error("sql_injection_protection", str(e), str(e.__traceback__))
            logger.error(f"SQL injection check failed: {str(e)}")
            raise

sql_injection_protection = SQLInjectionProtection()

# xAI Artifact Tags: #vial2 #security #sql_injection #neon_mcp
