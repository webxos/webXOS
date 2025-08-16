from typing import Dict, Any
from .settings import settings
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.openai_provider import OpenAIProvider
from ..providers.xai_provider import XAIProvider
from ..providers.google_provider import GoogleProvider

def get_provider_configs() -> Dict[str, Any]:
    return {
        "anthropic": settings.anthropic.dict() if settings.anthropic else {},
        "openai": settings.openai.dict() if settings.openai else {},
        "xai": settings.xai.dict() if settings.xai else {},
        "google": settings.google.dict() if settings.google else {},
    }

def get_provider(provider_name: str):
    configs = get_provider_configs()
    config = configs.get(provider_name)
    if not config:
        return None
    if provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name == "openai":
        return OpenAIProvider(config)
    elif provider_name == "xai":
        return XAIProvider(config)
    elif provider_name == "google":
        return GoogleProvider(config)
    return None
