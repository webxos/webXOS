from langchain.prompts import PromptTemplate

class BasePrompt:
    def __init__(self):
        self.template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are an AI assistant for the Vial MCP Controller. Use the provided context to answer the query concisely and accurately.
            Query: {query}
            Context: {context}
            Response:
            """
        )

    def format_prompt(self, query: str, context: str = "") -> str:
        try:
            return self.template.format(query=query, context=context)
        except Exception as e:
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Prompt formatting error: {str(e)}\n")
            raise ValueError(f"Prompt formatting failed: {str(e)}")

    def get_supported_models(self) -> list:
        return ["llama3.3", "mistral", "gemma2", "qwen", "phi"]
