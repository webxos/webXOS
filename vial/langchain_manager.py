from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.llms import NanoGPT
import logging

logger = logging.getLogger(__name__)

def create_langchain_agent():
    try:
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template="Process the following command for the Vial MCP: {input}"
        )
        llm = NanoGPT(model_name="nano-gpt", api_key="mock-api-key")
        agent = AgentExecutor.from_llm_and_tools(llm=llm, tools=[])
        logger.info("LangChain agent created")
        return agent
    except Exception as e:
        logger.error(f"LangChain agent creation error: {str(e)}")
        with open("errorlog.md", "a") as f:
            f.write(f"- **[2025-08-10T20:23:00Z]** LangChain agent creation error: {str(e)}\n")
        raise
