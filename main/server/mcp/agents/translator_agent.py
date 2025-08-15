# main/server/mcp/agents/translator_agent.py
from typing import Dict, Any, Optional
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
from ..agents.global_mcp_agents import GlobalMCPAgents
import os
import requests

class TranslatorAgent:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.collection = self.db["translations"]
        self.global_agents = GlobalMCPAgents()
        self.translation_api_url = os.getenv("TRANSLATION_API_URL", "https://api.example.com/translate")
        self.translation_api_key = os.getenv("TRANSLATION_API_KEY", "default_key")

    async def translate_text(self, agent_id: str, text: str, source_lang: str, target_lang: str, user_id: str) -> Dict[str, Any]:
        try:
            # Validate agent and user
            agents = await self.global_agents.list_agents(user_id)
            agent = next((a for a in agents if a["agent_id"] == agent_id), None)
            if not agent:
                raise MCPError(code=-32003, message="Agent not found or access denied")
            
            # Validate input
            if not text or len(text) > 1000:
                raise MCPError(code=-32602, message="Text is required and must be under 1000 characters")
            if source_lang not in ["en", "es", "fr", "de", "zh"] or target_lang not in ["en", "es", "fr", "de", "zh"]:
                raise MCPError(code=-32602, message="Unsupported language code")

            # Call external translation API (mocked for simplicity)
            response = requests.post(
                self.translation_api_url,
                headers={"Authorization": f"Bearer {self.translation_api_key}"},
                json={"text": text, "source_lang": source_lang, "target_lang": target_lang}
            )
            if response.status_code != 200:
                raise MCPError(code=-32603, message="Translation API error")

            translated_text = response.json().get("translated_text", text)  # Fallback to original text
            translation_record = {
                "agent_id": agent_id,
                "user_id": user_id,
                "source_text": text,
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            result = self.collection.insert_one(translation_record)

            return {
                "status": "success",
                "translation_id": str(result.inserted_id),
                "translated_text": translated_text
            }
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Translation failed: {str(e)}")

    async def get_translation_history(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        try:
            agents = await self.global_agents.list_agents(user_id)
            if not any(a["agent_id"] == agent_id for a in agents):
                raise MCPError(code=-32003, message="Agent not found or access denied")
            
            translations = self.collection.find({"agent_id": agent_id, "user_id": user_id}).limit(100)
            return [
                {
                    "translation_id": str(t["_id"]),
                    "source_text": t["source_text"],
                    "translated_text": t["translated_text"],
                    "source_lang": t["source_lang"],
                    "target_lang": t["target_lang"]
                }
                for t in translations
            ]
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to retrieve translation history: {str(e)}")

    def close(self):
        self.client.close()
