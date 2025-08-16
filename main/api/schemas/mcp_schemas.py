from pydantic import BaseModel
from typing import List, Dict, Any

class CompletionRequest(BaseModel):
    provider: str
    prompt: str
    params: Dict[str, Any] = {}

class CompletionResponse(BaseModel):
    provider: str
    model: str
    response: str
    usage: Dict[str, Any]
