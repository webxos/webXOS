from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any
import os

class DatabaseConfig(BaseSettings):
    url: str = Field(..., env="MONGO_URL")
    db_name: str = Field("vial_mcp", env="MONGO_DB_NAME")

class LLMProviderConfig(BaseSettings):
    name: str
    api_key: str
    base_url: Optional[str] = None
    model: str
    max_tokens: int = 1000
    temperature: float = 0.7
    rate_limit: int = 60

class QuantumConfig(BaseSettings):
    backends: list[str] = ["qiskit", "cirq"]
    default_backend: str = "qiskit"
    max_qubits: int = 20
    max_shots: int = 8192

class MLConfig(BaseSettings):
    pytorch_device: str = "auto"
    tensorflow_device: str = "auto"
    model_cache_size: int = 100
    enable_gpu: bool = True

class VialMCPConfig(BaseSettings):
    host: str = Field("0.0.0.0", env="VIAL_HOST")
    port: int = Field(8000, env="VIAL_PORT")
    debug: bool = Field(False, env="VIAL_DEBUG")
    anthropic: Optional[LLMProviderConfig] = None
    openai: Optional[LLMProviderConfig] = None
    xai: Optional[LLMProviderConfig] = None
    google: Optional[LLMProviderConfig] = None
    database: DatabaseConfig = DatabaseConfig()
    quantum: QuantumConfig = QuantumConfig()
    ml: MLConfig = MLConfig()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = VialMCPConfig()
