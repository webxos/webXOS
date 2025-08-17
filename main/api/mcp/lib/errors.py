from fastapi import HTTPException

class MCPError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

class DatabaseError(MCPError):
    def __init__(self, message: str, query: str = None):
        super().__init__("DATABASE_ERROR", message)
        self.query = query

class AuthenticationError(MCPError):
    def __init__(self, message: str):
        super().__init__("AUTHENTICATION_ERROR", message)

class ValidationError(MCPError):
    def __init__(self, message: str):
        super().__init__("VALIDATION_ERROR", message)

def raise_http_exception(error: MCPError) -> None:
    raise HTTPException(status_code=400, detail={
        "code": error.code,
        "message": error.message
    })
