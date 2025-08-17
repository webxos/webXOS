from ...utils.logging import log_error, log_info
from ...utils.rag import SmartRAG

class PromptHandler:
    def __init__(self):
        self.rag = SmartRAG()
        self.prompts = {
            "analyze_markdown": self.handle_analyze_markdown
        }

    async def list_prompts(self) -> list:
        return list(self.prompts.keys())

    async def handle_analyze_markdown(self, query: str, args: list) -> str:
        try:
            context = await self.rag.retrieve(query, top_k=5)
            response = await self.rag.generate(query, context)
            log_info(f"Prompt analyzed: {query}")
            return response
        except Exception as e:
            log_error(f"Prompt analysis failed: {str(e)}")
            raise

    async def get_prompt(self, name: str, args: list) -> str:
        if name not in self.prompts:
            log_error(f"Unknown prompt: {name}")
            raise ValueError(f"Unknown prompt: {name}")
        return await self.prompts[name](*args)
