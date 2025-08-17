import dspy
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from ..config.mcp_config import mcp_config
from ..utils.logging import log_error, log_info

class SmartRAG:
    def __init__(self):
        self.transformer = SentenceTransformer(mcp_config.SENTENCE_TRANSFORMER_MODEL)
        self.mongo_client = MongoClient(mcp_config.MONGODB_CONNECTION_STRING)
        self.db = self.mongo_client[mcp_config.MONGODB_DATABASE]
        self.lm = dspy.LM('gpt-3.5-turbo')
        dspy.settings.configure(lm=self.lm)

    async def retrieve(self, query: str, top_k: int = 5) -> list:
        try:
            query_embedding = self.transformer.encode([query])[0]
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "markdown_embeddings",
                        "path": "analysis.embeddings",
                        "queryVector": query_embedding.tolist(),
                        "numCandidates": 100,
                        "limit": top_k
                    }
                },
                {"$project": {"content": 1, "analysis": 1, "score": {"$meta": "vectorSearchScore"}}}
            ]
            results = await self.db.markdowns.aggregate(pipeline).to_list(top_k)
            log_info(f"RAG retrieval for query: {query}")
            return results
        except Exception as e:
            log_error(f"RAG retrieval failed: {str(e)}")
            return []

    async def generate(self, query: str, context: list) -> str:
        try:
            prompt = f"Query: {query}\nContext: {json.dumps(context, indent=2)}\nAnswer:"
            response = self.lm(prompt)
            log_info(f"RAG generation for query: {query}")
            return response
        except Exception as e:
            log_error(f"RAG generation failed: {str(e)}")
            return f"Error: {str(e)}"
