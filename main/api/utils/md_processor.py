import spacy
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from ...config.mcp_config import mcp_config
from ...utils.logging import log_error, log_info

class MongoDBMarkdownProcessor:
    def __init__(self):
        self.nlp = spacy.load(mcp_config.SPACY_MODEL)
        self.transformer = SentenceTransformer(mcp_config.SENTENCE_TRANSFORMER_MODEL)
        self.client = MongoClient(mcp_config.MONGODB_CONNECTION_STRING)
        self.db = self.client[mcp_config.MONGODB_DATABASE]

    async def process_markdown(self, content: str, doc_id: str) -> dict:
        try:
            doc = self.nlp(content)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            embeddings = self.transformer.encode([content])[0].tolist()
            analysis = {
                "doc_id": doc_id,
                "entities": entities,
                "embeddings": embeddings
            }
            await self.db.markdowns.update_one(
                {"doc_id": doc_id},
                {"$set": {"content": content, "analysis": analysis}},
                upsert=True
            )
            log_info(f"Markdown processed for doc_id: {doc_id}")
            return analysis
        except Exception as e:
            log_error(f"Markdown processing failed for doc_id {doc_id}: {str(e)}")
            raise
