"""
FastAPI application entry point.

This file:
  1. Creates the FastAPI app instance
  2. Configures CORS (Cross-Origin Resource Sharing)
  3. Registers all routers
  4. Defines the health check endpoint
  5. Defines startup/shutdown event hooks

Why FastAPI?
  FastAPI is the modern Python standard for ML APIs. It gives you:
    - Automatic /docs Swagger UI (no manual documentation)
    - Automatic /redoc ReDoc UI
    - Async support (Starlette underneath)
    - Pydantic integration for request/response validation
    - High performance (comparable to Go and Node.js for I/O-bound workloads)
    - OpenAPI schema generation

  Compared to Flask:
    - FastAPI has built-in async, data validation, and auto-docs
    - Flask requires separate libraries (marshmallow, flask-swagger) to get the same features
    - FastAPI is ~3x faster than Flask on typical workloads

To start the server:
    uvicorn api.main:app --reload --port 8000

  --reload  : Automatically restart on file changes (development only)
  --port    : Port to listen on

The API will be available at:
  - http://localhost:8000/         (root)
  - http://localhost:8000/health   (health check)
  - http://localhost:8000/docs     (Swagger UI)
  - http://localhost:8000/redoc    (ReDoc UI)
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
    """
    Application lifespan manager.

    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.

    We pre-warm the model cache here so the first request doesn't have
    a loading delay. If models aren't trained yet, we log a warning but
    don't crash — the health endpoint will report which models are unavailable.
    """
    logger.info("Starting Intelligent Risk Analytics API v{version}", version=APP_VERSION)

    # Try to pre-load models into cache
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

    # Store loaded model list in app state for the health endpoint
    app.state.loaded_models = loaded_models

    yield  # Application runs here

    logger.info("Shutting down API.")


# Create the FastAPI application
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
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
    openapi_url="/openapi.json",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) controls which domains can call this API.
# In development we allow all origins ("*"). In production you would restrict
# this to the specific frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (development only)
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(predict_router)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="API root")
async def root():
    """API root endpoint — returns basic info."""
    return {
        "name": "Intelligent Risk Analytics API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", summary="Health check")
async def health():
    """
    Health check endpoint.

    Why a health endpoint?
      Docker HEALTHCHECK, Kubernetes liveness probes, and load balancers all
      call /health to determine if the service is running correctly.
      Returning 200 with model status lets operators know if models are ready.
    """
    loaded_models = getattr(app.state, "loaded_models", [])

    return {
        "status": "healthy",
        "version": APP_VERSION,
        "models_loaded": loaded_models,
        "models_available": ["credit_risk", "network_intrusion"],
        "models_ready": len(loaded_models) == 2,
    }
