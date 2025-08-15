# main/server/mcp/utils/api_docs.py
from typing import Dict, Any
from ..utils.api_config import APIConfig
from ..utils.mcp_error_handler import MCPError
import logging
import json

logger = logging.getLogger("mcp")

class APIDocs:
    def __init__(self):
        self.api_config = APIConfig()
        self.endpoints_metadata = {
            "mcp.createSession": {
                "description": "Creates a new user session with optional MFA.",
                "params": {
                    "user_id": {"type": "string", "required": True},
                    "mfa_verified": {"type": "boolean", "required": False}
                },
                "returns": {"session_id": "string", "access_token": "string", "expires_at": "string"}
            },
            "mcp.initiateMFA": {
                "description": "Initiates MFA for a user session.",
                "params": {
                    "user_id": {"type": "string", "required": True},
                    "mfa_method": {"type": "string", "required": True}
                },
                "returns": {"mfa_token": "string", "method": "string", "expires_at": "string"}
            },
            "mcp.createNote": {
                "description": "Creates a new note.",
                "params": {
                    "user_id": {"type": "string", "required": True},
                    "title": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                    "tags": {"type": "array", "required": False}
                },
                "returns": {"note_id": "string", "status": "string"}
            },
            "mcp.addSubIssue": {
                "description": "Adds a sub-issue to a parent issue.",
                "params": {
                    "parent_issue_id": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                    "user_id": {"type": "string", "required": True}
                },
                "returns": {"sub_issue_id": "string", "status": "string"}
            },
            "mcp.subscribe": {
                "description": "Subscribes to a channel for real-time updates.",
                "params": {
                    "user_id": {"type": "string", "required": True},
                    "channel": {"type": "string", "required": True},
                    "event_types": {"type": "array", "required": True}
                },
                "returns": {"subscription_id": "string"}
            }
        }

    def generate_docs(self) -> str:
        try:
            config = self.api_config.load_config()
            markdown = "# Vial MCP Controller API Documentation\n\n"
            markdown += "## Endpoints\n\n"
            
            for endpoint, metadata in self.endpoints_metadata.items():
                if not config["endpoints"].get(endpoint, {}).get("enabled", False):
                    continue
                
                markdown += f"### {endpoint}\n"
                markdown += f"{metadata['description']}\n\n"
                markdown += "- **Method**: POST\n"
                markdown += "- **Auth Required**: {}\n\n".format(
                    config["endpoints"][endpoint].get("auth_required", False)
                )
                
                markdown += "**Params**:\n"
                for param, details in metadata["params"].items():
                    markdown += f"- `{param}` ({details['type']}, {'required' if details['required'] else 'optional'})\n"
                
                markdown += "\n**Returns**:\n"
                for key, value in metadata["returns"].items():
                    markdown += f"- `{key}`: {value}\n"
                
                markdown += "\n**Example**:\n```json\n"
                markdown += json.dumps({
                    "jsonrpc": "2.0",
                    "method": endpoint,
                    "params": {k: f"example_{k}" for k in metadata["params"].keys()},
                    "id": 1
                }, indent=2)
                markdown += "\n```\n\n"
            
            logger.info("Generated API documentation")
            return markdown
        except Exception as e:
            logger.error(f"Failed to generate API docs: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to generate API docs: {str(e)}")
