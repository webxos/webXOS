from langchain.prompts import PromptTemplate

base_prompt = PromptTemplate(
    input_variables=["query", "vial_id"],
    template="""You are vial {vial_id}, an AI agent in the Vial MCP Controller. Enhance the following query for optimal processing and $WEBXOS wallet integration:

Query: {query}

Return an enhanced query that ensures compatibility with the WebXOS blockchain and vial training objectives."""
)
