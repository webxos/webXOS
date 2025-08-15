# main/server/mcp/notes/mcp_server_notes.py
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pydantic import BaseModel
from bson import ObjectId
from ..utils.mcp_error_handler import MCPError
import os

class Note(BaseModel):
    note_id: str
    title: str
    content: str
    tags: List[str]
    user_id: str

class NotesService:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.collection = self.db["notes"]

    async def create_note(self, title: str, content: str, tags: List[str], user_id: str) -> Dict[str, Any]:
        try:
            if not title or not content:
                raise MCPError(code=-32602, message="Title and content are required")
            if len(tags) > 10:
                raise MCPError(code=-32602, message="Maximum 10 tags allowed")
            note = {
                "title": title,
                "content": content,
                "tags": tags,
                "user_id": user_id
            }
            result = self.collection.insert_one(note)
            return {
                "status": "success",
                "note_id": str(result.inserted_id)
            }
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to create note: {str(e)}")

    async def get_note(self, note_id: str, user_id: str) -> Dict[str, Any]:
        try:
            note = self.collection.find_one({"_id": ObjectId(note_id), "user_id": user_id})
            if not note:
                raise MCPError(code=-32003, message="Note not found or access denied")
            return {
                "note_id": str(note["_id"]),
                "title": note["title"],
                "content": note["content"],
                "tags": note["tags"],
                "user_id": note["user_id"]
            }
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to retrieve note: {str(e)}")

    async def search_notes(self, user_id: str, tags: Optional[List[str]] = None, query: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            search_filter = {"user_id": user_id}
            if tags:
                search_filter["tags"] = {"$all": tags}
            if query:
                search_filter["$text"] = {"$search": query}
            notes = self.collection.find(search_filter).limit(100)
            return [
                {
                    "note_id": str(note["_id"]),
                    "title": note["title"],
                    "content": note["content"],
                    "tags": note["tags"],
                    "user_id": note["user_id"]
                }
                for note in notes
            ]
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to search notes: {str(e)}")

    def close(self):
        self.client.close()
