from pydantic_settings import BaseSettings

class MCPConfig(BaseSettings):
    MCP_SERVER_NAME: str = "webxos-mcp-gateway"
    MCP_SERVER_VERSION: str = "2.7.8"
    MCP_PROTOCOL_VERSION: str = "2024-11-05"
    MONGODB_CONNECTION_STRING: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "webxos_mcp"
    REDIS_URL: str = "redis://localhost:6379"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    SPACY_MODEL: str = "en_core_web_sm"
    JWT_SECRET_KEY: str
    ALGORITHM: str = "HS256"
    REPO_URL: str = "https://github.com/webxos/webxos-mcp-gateway.git"
    REPO_TOKEN: str
    ANTHROPIC_API_KEY: str
    OPENAI_API_KEY: str
    XAI_API_KEY: str
    GOOGLE_API_KEY: str

    class Config:
        env_file = "../../.env"
        env_file_encoding = "utf-8"

mcp_config = MCPConfig()
