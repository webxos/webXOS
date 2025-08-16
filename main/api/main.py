from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .routes import wallet, quantum_link, troubleshoot, oauth, mcp_endpoints
from ..utils.logging import setup_logging
from ..security.authentication import verify_token

app = FastAPI(title="Vial MCP Gateway", version="2.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://webxos.netlify.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wallet.router, prefix="/v1")
app.include_router(quantum_link.router, prefix="/v1")
app.include_router(troubleshoot.router, prefix="/v1")
app.include_router(oauth.router, prefix="/v1")
app.include_router(mcp_endpoints.router, prefix="/v1")

setup_logging()
