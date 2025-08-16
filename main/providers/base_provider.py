from abc import ABC, abstractmethod
from ...utils.logging import log_error, log_info
import asyncio

class BaseProvider(ABC):
    def __init__(self, api_key):
        self.api_key = api_key
        self.is_available = True

    @abstractmethod
    async def generate_response(self, prompt):
        pass

    async def check_availability(self):
        try:
            # Mock availability check
            await asyncio.sleep(0.1)
            self.is_available = True
            log_info(f"Provider availability checked: {self.__class__.__name__}")
        except Exception as e:
            self.is_available = False
            log_error(f"Provider {self.__class__.__name__} unavailable: {str(e)}")
            raise
