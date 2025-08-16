from typing import Dict, Any
from .settings import settings

def get_provider_configs() -> Dict[str, Any]:
    return {
        "anthropic": settings.anthropic.dict() if settings.anthropic else {},
        "openai": settings.openai.dict() if settings.openai else {},
        "xai": settings.xai.dict() if settings.xai else {},
        "google": settings.google.dict() if settings.google else {},
    }
