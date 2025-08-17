import redis
from config.mcp_config import MCP_CONFIG

redis_client = redis.from_url(MCP_CONFIG["REDIS_URL"])
