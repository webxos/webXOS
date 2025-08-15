# main/server/mcp/agents/translator_agent.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..db.db_manager import DBManager
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
import os
import requests
from datetime import datetime

app = FastAPI(title="Vial MCP Translator Agent")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
db_manager = DBManager()

class TranslationRequest(BaseModel):
    user_id: str
    text: str
    source_lang: Optional[str] = None
    target_lang: str

class TranslationResponse(BaseModel):
    translation_id: str
    translated_text: str
    source_lang: Optional[str]
    target_lang: str
    timestamp: str

@app.post("/agents/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest, token: str = Depends(oauth2_scheme)):
    with metrics.track_span("translate_text", {"user_id": request.user_id, "target_lang": request.target_lang}):
        try:
            metrics.verify_token(token)
            ai_service_url = os.getenv("AI_SERVICE_URL", "http://localhost:8000/ai")
            response = requests.post(
                f"{ai_service_url}/translate",
                json={
                    "text": request.text,
                    "source_lang": request.source_lang,
                    "target_lang": request.target_lang
                },
                timeout=10
            )
            response.raise_for_status()
            translated_text = response.json().get("translated_text", "")

            translation_data = {
                "user_id": request.user_id,
                "text": request.text,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "translated_text": translated_text,
                "timestamp": datetime.utcnow()
            }
            translation_id = db_manager.insert_one("translations", translation_data)

            return TranslationResponse(
                translation_id=translation_id,
                translated_text=translated_text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                timestamp=str(translation_data["timestamp"])
            )
        except Exception as e:
            handle_generic_error(e, context="translate_text")
            raise HTTPException(status_code=500, detail=f"Failed to translate text: {str(e)}")
