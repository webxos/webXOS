from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nomic
import llmware
from jina import Client
from transformers import pipeline
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import metaflow
import ollama
import logging
import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LibraryRequest(BaseModel):
    query: str
    vial_id: str
    wallet: dict

class TranslatorRequest(BaseModel):
    text: str
    target_language: str

# Inception Gateway for library data parsing and transfer
class InceptionGateway:
    def __init__(self):
        self.nomic_client = nomic
        self.jina_client = Client()
        self.hf_pipeline = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
        self.llm_chain = LLMChain(
            llm=Ollama(model="llama3"),
            prompt=PromptTemplate.from_template("Translate {text} to {language}")
        )

    async def process_nomic(self, query, wallet):
        try:
            embeddings = self.nomic_client.embed.text([query], model="nomic-embed-text-v1")
            wallet["transactions"].append({
                "type": "nomic_query",
                "query": query,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            return {"response": embeddings, "wallet": wallet}
        except Exception as e:
            logger.error(f"Nomic error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_cognitallmware(self, query, wallet):
        try:
            # Placeholder for CogniTALLMware (no official package, simulated)
            response = {"message": f"CogniTALLMware processed: {query}"}
            wallet["transactions"].append({
                "type": "cognitallmware_query",
                "query": query,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            return {"response": response, "wallet": wallet}
        except Exception as e:
            logger.error(f"CogniTALLMware error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_llmware(self, query, wallet):
        try:
            response = llmware.process_query(query)
            wallet["transactions"].append({
                "type": "llmware_query",
                "query": query,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            return {"response": response, "wallet": wallet}
        except Exception as e:
            logger.error(f"LLMware error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_jina(self, query, wallet):
        try:
            response = self.jina_client.search(query, modality="text")
            wallet["transactions"].append({
                "type": "jina_query",
                "query": query,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            return {"response": response, "wallet": wallet}
        except Exception as e:
            logger.error(f"Jina AI error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def translate(self, text, target_language):
        try:
            result = self.llm_chain.run(text=text, language=target_language)
            return {"translated_text": result}
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

gateway = InceptionGateway()

@app.post("/api/library/{vial_id}")
async def process_library_request(request: LibraryRequest):
    try:
        if request.vial_id == "1":
            return await gateway.process_nomic(request.query, request.wallet)
        elif request.vial_id == "2":
            return await gateway.process_cognitallmware(request.query, request.wallet)
        elif request.vial_id == "3":
            return await gateway.process_llmware(request.query, request.wallet)
        elif request.vial_id == "4":
            return await gateway.process_jina(request.query, request.wallet)
        else:
            raise HTTPException(status_code=400, detail="Invalid vial ID")
    except Exception as e:
        logger.error(f"Library request error: {str(e)}")
        with open("vial/errorlog.md", "a") as f:
            f.write(f"- **[{(datetime.datetime.utcnow().isoformat())}]** Library request error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate")
async def translate_text(request: TranslatorRequest):
    return await gateway.translate(request.text, request.target_language)
