from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..utils.logging import log_error

def add_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://webxos.netlify.app"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log_info("CORS middleware added for https://webxos.netlify.app")
