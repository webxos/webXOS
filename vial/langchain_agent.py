from langchain.agents import AgentExecutor, Tool
from langchain.llms.base import LLM
from typing import Any, List, Optional
import importlib
import pkgutil

class NanoGPTLLM(LLM):
    def _call(self, prompt: str, **kwargs) -> str:
        return f"Simulated NanoGPT response to: {prompt}"
    
    @property
    def _llm_type(self) -> str:
        return "nanogpt"

def create_langchain_agent() -> AgentExecutor:
    """Create a LangChain agent with NanoGPT and dynamic prompts."""
    llm = NanoGPTLLM()
    tools = []
    for _, name, _ in pkgutil.iter_modules(['prompts']):
        module = importlib.import_module(f'prompts.{name}')
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, '_is_mcp_prompt'):
                tools.append(Tool(name=attr_name, func=lambda x: attr(x), description=f"Prompt: {attr_name}"))
    return AgentExecutor.from_agent_and_tools(agent=llm, tools=tools, verbose=True)