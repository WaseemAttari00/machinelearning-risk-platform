"""
Prediction API routes.

Two endpoints:
  POST /api/v1/predict/credit-risk       → credit risk prediction
  POST /api/v1/predict/network-intrusion → intrusion detection prediction

Both follow the same pattern:
  1. Receive validated Pydantic input
  2. Convert to a plain dict
  3. Call predict_single() from the prediction module
  4. Return a structured response

Why a separate routes file instead of putting everything in main.py?
  As the API grows, main.py would become huge. Splitting routes into separate
  files follows the "Single Responsibility Principle" — main.py handles app
  setup; routes/predict.py handles prediction logic.
  This also mirrors real-world FastAPI project structure.
"""

from fastapi import APIRouter, HTTPException, status

from api.schemas.prediction import (
    CreditRiskInput,
    NetworkIntrusionInput,
    PredictionResponse,
)
from src.models.predict import predict_single
from src.utils.logger import get_logger

logger = get_logger(__name__)

# APIRouter is like a "mini app" that gets mounted onto the main FastAPI app.
# The prefix and tags appear in the /docs Swagger UI.
router = APIRouter(
    prefix="/api/v1/predict",
    tags=["predictions"],
)


@router.post(
    "/credit-risk",
    response_model=PredictionResponse,
    summary="Predict credit default risk",
    description="Given financial features of a loan applicant, predict the probability "
                "of defaulting within 2 years.",
)
async def predict_credit_risk(input_data: CreditRiskInput) -> PredictionResponse:
    """
    Credit risk prediction endpoint.

    FastAPI automatically:
      - Parses the JSON body into a CreditRiskInput object
      - Validates all fields (types, ranges) via Pydantic
      - Returns 422 with helpful error messages if validation fails
      - Serializes the response using PredictionResponse schema

    The 'async' keyword marks this as an async handler.
    For CPU-bound work like model inference, sync functions work fine too,
    but async is standard in FastAPI and supports I/O-bound operations.
    """
    try:
        # model_dump() converts the Pydantic model to a plain Python dict.
        # by_alias=True uses the original column names with hyphens (needed by the pipeline).
        input_dict = input_data.model_dump(by_alias=True)

        result = predict_single(domain="credit_risk", input_data=input_dict)

        logger.info(
            "Credit risk prediction: probability={prob:.4f}, label={label}",
            prob=result["probability"],
            label=result["risk_label"],
        )

        return PredictionResponse(**result)

    except FileNotFoundError as e:
        # This happens if the model hasn't been trained yet.
        # We return 503 (Service Unavailable) rather than 500 (Internal Server Error)
        # because the server is running fine — the model just isn't ready.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {str(e)}",
        )
    except Exception as e:
        logger.error("Prediction error: {err}", err=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/network-intrusion",
    response_model=PredictionResponse,
    summary="Detect network intrusion",
    description="Given network traffic features, predict whether the connection "
                "is normal (benign) or an attack.",
)
async def predict_network_intrusion(input_data: NetworkIntrusionInput) -> PredictionResponse:
    """Network intrusion detection endpoint."""
    try:
        input_dict = input_data.model_dump()

        result = predict_single(domain="network_intrusion", input_data=input_dict)

        logger.info(
            "Network intrusion prediction: probability={prob:.4f}, label={label}",
            prob=result["probability"],
            label=result["risk_label"],
        )

        return PredictionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {str(e)}",
        )
    except Exception as e:
        logger.error("Prediction error: {err}", err=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )
