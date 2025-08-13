from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import BaseLLM
import requests
import grpc
import json
import xml.etree.ElementTree as ET
import logging
import datetime
import os
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VialLLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.api_key = os.getenv("LLM_API_KEY")
        self.supported_models = ["llama3.3", "mistral", "gemma2", "qwen", "phi"]

    async def call_llm(self, prompt: str, model: str, format: str = "json") -> Dict[str, Any]:
        try:
            if model not in self.supported_models:
                raise ValueError(f"Unsupported model: {model}")
            
            # REST API call
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"prompt": prompt, "model": model}
            response = requests.post("https://api.huggingface.co/v1/inference", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Protocol conversion for legacy systems (e.g., gRPC)
            try:
                # Example gRPC conversion (placeholder)
                channel = grpc.insecure_channel("legacy-system:50051")
                stub = legacy_service_pb2_grpc.LegacyServiceStub(channel)
                grpc_response = stub.ProcessPrompt(legacy_service_pb2.PromptRequest(prompt=prompt))
                result["grpc_response"] = grpc_response.result
            except Exception as e:
                logger.warning(f"gRPC conversion failed: {str(e)}")
            
            # Transform response
            if format == "xml":
                root = ET.Element("response")
                for key, value in result.items():
                    child = ET.SubElement(root, key)
                    child.text = str(value)
                return {"status": "success", "response": ET.tostring(root, encoding="unicode")}
            return {"status": "success", "response": result}
        except Exception as e:
            logger.error(f"LLM call error: {str(e)}")
            with open("db/errorlog.md", "a") as f:
                f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** LLM call error: {str(e)}\n")
            raise Exception(str(e))

    def _generate(self, prompts: List[str], stop: List[str] = None) -> Dict[str, Any]:
        # Synchronous wrapper for LangChain compatibility
        import asyncio
        return asyncio.run(self.call_llm(prompts[0], self.supported_models[0]))
