import asyncio
from abc import ABC, abstractmethod
from ..utils.logging import log_error, log_info

class BaseProvider(ABC):
    def __init__(self, api_key):
        self.api_key = api_key
        self.is_available = False

    async def check_availability(self):
        try:
            # Placeholder for real availability check
            await asyncio.sleep(0.1)
            self.is_available = True
            log_info(f"Provider availability checked: {self.__class__.__name__} is available")
        except Exception as e:
            self.is_available = False
            log_error(f"Traceback: Provider {self.__class__.__name__} unavailable: {str(e)}")

    @abstractmethod
    async def generate_response(self, prompt):
        pass

    async def handle_request(self, prompt, fallback_provider=None):
        try:
            if not self.is_available:
                raise Exception(f"Provider {self.__class__.__name__} is unavailable")
            response = await self.generate_response(prompt)
            return response
        except Exception as e:
            log_error(f"Traceback: Request failed for {self.__class__.__name__}: {str(e)}")
            if fallback_provider:
                log_info(f"Falling back to {fallback_provider.__class__.__name__}")
                return await fallback_provider.handle_request(prompt)
            raise
