from langchain import LLMChain
from langchain.prompts import PromptTemplate

class MCPChain(LLMChain):
    def __init__(self, llm, prompt=None):
        prompt = prompt or PromptTemplate(input_variables=["input"], template="{input}")
        super().__init__(llm=llm, prompt=prompt)
        self.cache = {}

    def get_from_cache(self, key):
        return self.cache.get(key)

    def store_vector(self, key, vector):
        self.cache[key] = vector