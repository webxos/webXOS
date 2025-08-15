# main/server/mcp/utils/api_docs.py
from fastapi import FastAPI, Depends
from fastapi.openapi.utils import get_openapi
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.error_handler import handle_generic_error
from fastapi.security import OAuth2PasswordBearer
from ..utils.api_config import APIConfig
import yaml
import os

app = FastAPI(title="Vial MCP API Docs")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
metrics = PerformanceMetrics()
api_config = APIConfig()

class APIDocs:
    def __init__(self):
        self.metrics = PerformanceMetrics()

    def generate_openapi(self, app: FastAPI) -> dict:
        with self.metrics.track_span("generate_openapi"):
            try:
                openapi_schema = get_openapi(
                    title="Vial MCP API",
                    version="1.0.0",
                    description="API documentation for Vial MCP Controller services",
                    routes=app.routes,
                )
                endpoints = api_config.list_endpoints()
                for endpoint in endpoints:
                    openapi_schema["paths"][f"/{endpoint.name}"] = {
                        "get": {
                            "summary": f"Access {endpoint.name} service",
                            "responses": {
                                "200": {"description": "Successful response"},
                                "401": {"description": "Unauthorized"}
                            }
                        }
                    }
                return openapi_schema
            except Exception as e:
                handle_generic_error(e, context="generate_openapi")
                raise

    def save_openapi_yaml(self, schema: dict, filename: str = "openapi.yaml") -> None:
        with self.metrics.track_span("save_openapi_yaml"):
            try:
                with open(filename, "w") as f:
                    yaml.dump(schema, f, sort_keys=False)
            except Exception as e:
                handle_generic_error(e, context="save_openapi_yaml")
                raise

@app.get("/docs/openapi")
async def get_openapi_docs(token: str = Depends(oauth2_scheme)):
    with metrics.track_span("get_openapi_docs"):
        try:
            metrics.verify_token(token)
            docs = APIDocs()
            openapi_schema = docs.generate_openapi(app)
            return openapi_schema
        except Exception as e:
            handle_generic_error(e, context="get_openapi_docs")
            raise

@app.post("/docs/save")
async def save_openapi_docs(token: str = Depends(oauth2_scheme)):
    with metrics.track_span("save_openapi_docs"):
        try:
            metrics.verify_token(token)
            docs = APIDocs()
            openapi_schema = docs.generate_openapi(app)
            docs.save_openapi_yaml(openapi_schema)
            return {"status": "success", "message": "OpenAPI docs saved to openapi.yaml"}
        except Exception as e:
            handle_generic_error(e, context="save_openapi_docs")
            raise
