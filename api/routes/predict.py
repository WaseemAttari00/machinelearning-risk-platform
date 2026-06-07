"""Prediction endpoints for credit risk and network intrusion domains."""

from fastapi import APIRouter, HTTPException, status

from api.schemas.prediction import (
    CreditRiskInput,
    NetworkIntrusionInput,
    PredictionResponse,
)
from src.models.predict import predict_single
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    """Credit risk prediction endpoint."""
    try:
        input_dict = input_data.model_dump(by_alias=True)
        result = predict_single(domain="credit_risk", input_data=input_dict)

        logger.info(
            "Credit risk prediction: probability={prob:.4f}, label={label}",
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
