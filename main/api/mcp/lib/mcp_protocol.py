from pydantic import BaseModel, validator
from typing import Dict, Any, Optional
from lib.errors import ValidationError

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: Optional[int]

    @validator("jsonrpc")
    def check_jsonrpc_version(cls, value):
        if value != "2.0":
            raise ValidationError("Invalid JSON-RPC version, must be 2.0")
        return value

    @validator("method")
    def check_method_format(cls, value):
        if not value or "." not in value:
            raise ValidationError("Method must be in format 'tool.method'")
        return value

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any = None
    error: Any = None
    id: Optional[int]

    @validator("error", always=True)
    def check_result_or_error(cls, error, values):
        if error is None and values.get("result") is None:
            raise ValidationError("Either result or error must be provided")
        if error is not None and values.get("result") is not None:
            raise ValidationError("Cannot provide both result and error")
        return error

class MCPNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
