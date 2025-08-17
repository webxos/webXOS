from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import health, oauth

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://webxos.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1")
app.include_router(oauth.router, prefix="/v1")
