from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main.api.routes import health, oauth

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://webxos.netlify.app", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1")
app.include_router(oauth.router, prefix="/v1")
