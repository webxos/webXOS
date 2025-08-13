from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import MarianMTModel, MarianTokenizer
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain
import logging
import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslatorRequest(BaseModel):
    text: str
    target_language: str
    wallet: dict

class TranslatorAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.llm_chain = LLMChain(
            llm=Ollama(model="llama3"),
            prompt=PromptTemplate.from_template("Translate {text} to {language}")
        )

    def load_model(self, target_language: str):
        model_name = f"Helsinki-NLP/opus-mt-en-{target_language.lower()}"
        if target_language not in self.models:
            try:
                self.models[target_language] = MarianMTModel.from_pretrained(model_name).to(self.device)
                self.tokenizers[target_language] = MarianTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"MarianMT model not found for {target_language}, falling back to LLMChain: {str(e)}")
                return False
        return True

    async def translate(self, text: str, target_language: str, wallet: dict) -> dict:
        try:
            if self.load_model(target_language):
                inputs = self.tokenizers[target_language](text, return_tensors="pt", padding=True).to(self.device)
                translated = self.models[target_language].generate(**inputs)
                result = self.tokenizers[target_language].decode(translated[0], skip_special_tokens=True)
            else:
                result = await self.llm_chain.run(text=text, language=target_language)
            wallet["transactions"].append({
                "type": "translation",
                "text": text,
                "target_language": target_language,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            wallet["webxos"] = wallet.get("webxos", 0.0) + 0.0001
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Translation completed: {text} to {target_language}\n")
            return {"translated_text": result, "wallet": wallet}
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            with open("vial/errorlog.md", "a") as f:
                f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Translation error: {str(e)}\n")
            raise HTTPException(status_code=500, detail=str(e))

translator = TranslatorAgent()

@app.post("/api/translate")
async def translate_text(request: TranslatorRequest):
    return await translator.translate(request.text, request.target_language, request.wallet)
