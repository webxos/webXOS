import dspy
import spacy
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from ...config.mcp_config import config
from ...utils.logging import log_error, log_info
import re
import json
from typing import Dict, List, Any

class MarkdownProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.mongo_client = MongoClient(config.MONGODB_CONNECTION_STRING)
        self.db = self.mongo_client[config.MONGODB_DATABASE]
        self.lm = dspy.LM('gpt-3.5-turbo')
        dspy.settings.configure(lm=self.lm)

    async def process_markdown(self, content: str) -> Dict[str, Any]:
        try:
            doc = self.nlp(content)
            embeddings = self.transformer.encode([content])[0]
            code_blocks = self.extract_code_blocks(content)
            analysis = {
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "keywords": [token.text for token in doc if token.is_alpha and not token.is_stop],
                "embeddings": embeddings.tolist(),
                "code_blocks": code_blocks,
                "complexity": self.calculate_complexity(code_blocks)
            }
            await self.db.markdowns.insert_one({
                "content": content,
                "analysis": analysis,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            log_info("Markdown processed and stored")
            return analysis
        except Exception as e:
            log_error(f"Markdown processing failed: {str(e)}")
            return {"error": str(e)}

    def extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        code_blocks = []
        pattern = r"```(\w+)?\n([\s\S]*?)```"
        for match in re.finditer(pattern, content):
            language = match.group(1) or "unknown"
            code = match.group(2).strip()
            code_blocks.append({"language": language, "code": code})
        return code_blocks

    def calculate_complexity(self, code_blocks: List[Dict[str, str]]) -> Dict[str, Any]:
        complexity = {"total_lines": 0, "functions": 0, "loops": 0}
        for block in code_blocks:
            code = block["code"]
            complexity["total_lines"] += len(code.splitlines())
            complexity["functions"] += len(re.findall(r"def\s+\w+\(", code)) if block["language"] == "python" else 0
            complexity["loops"] += len(re.findall(r"\b(for|while)\b", code))
        return complexity

    async def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
            log_info(f"Vector search completed for query: {query}")
            return results
        except Exception as e:
            log_error(f"Vector search failed: {str(e)}")
            return [{"error": str(e)}]
