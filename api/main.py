"""
FastAPI application entry point.

Registers routes, configures CORS, pre-warms model cache on startup,
and exposes health and root endpoints.

Start the server:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /             — root info
    GET  /health       — model readiness status
    GET  /docs         — Swagger UI
    POST /api/v1/predict/credit-risk
    POST /api/v1/predict/network-intrusion
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.predict import router as predict_router
from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = get_project_root()
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm model cache on startup; log shutdown on exit."""
    logger.info("Starting Intelligent Risk Analytics API v{version}", version=APP_VERSION)

    loaded_models = []
    for domain in ["credit_risk", "network_intrusion"]:
        model_path = PROJECT_ROOT / "models" / domain / "xgboost_model.joblib"
        if model_path.exists():
            try:
                from src.models.predict import load_model_and_pipeline
                load_model_and_pipeline(domain)
                loaded_models.append(domain)
                logger.info("Pre-loaded model for domain: {domain}", domain=domain)
            except Exception as e:
                logger.warning(
                    "Could not pre-load model for {domain}: {err}",
                    domain=domain,
                    err=str(e),
                )
        else:
            logger.warning(
                "Model not found for domain '{domain}' — run training first.",
                domain=domain,
            )

    app.state.loaded_models = loaded_models

    yield

    logger.info("Shutting down API.")


app = FastAPI(
    title="Intelligent Risk Analytics API",
    description=(
        "End-to-end ML platform for risk prediction.\n\n"
        "Supports two domains:\n"
        "- **Credit Risk**: Predict loan default probability\n"
        "- **Network Intrusion**: Detect malicious network traffic"
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/", summary="API root")
async def root():
    return {
        "name": "Intelligent Risk Analytics API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", summary="Health check")
async def health():
    """Return service status and which models are loaded."""
    loaded_models = getattr(app.state, "loaded_models", [])

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "models_loaded": loaded_models,
        "models_available": ["credit_risk", "network_intrusion"],
        "models_ready": len(loaded_models) == 2,
    }
