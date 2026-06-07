"""
Pydantic schemas for API request and response validation.

What is Pydantic?
  Pydantic is a data validation library that uses Python type annotations.
  When a request comes into a FastAPI endpoint, Pydantic automatically:
    1. Parses the JSON body
    2. Validates each field against its type annotation
    3. Raises a 422 Unprocessable Entity error with a clear message if validation fails
    4. Returns a clean Python object if validation passes

Why does this matter?
  Without validation, your model could receive strings where it expects numbers,
  or missing fields that cause a KeyError deep in the prediction code.
  Pydantic makes the API "self-documenting" and "self-protecting" at the boundary.

FastAPI uses these schemas to:
  - Auto-generate the /docs Swagger UI (so users know exactly what to send)
  - Validate incoming requests before they reach your business logic
  - Serialize outgoing responses to JSON

Interview note:
  "Input validation at the system boundary" is a best practice in software
  engineering. Pydantic + FastAPI is the modern Python way to implement it.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CreditRiskInput(BaseModel):
    """
    Input schema for credit risk prediction.

    All fields correspond to columns in the Give Me Some Credit dataset.
    Field descriptions, constraints, and examples appear automatically in
    the FastAPI /docs Swagger UI.
    """

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ...,
        ge=0.0,
        description="Total balance on credit cards and personal lines of credit "
                    "divided by the sum of credit limits. Range: [0, ∞]",
        example=0.766,
    )
    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Age of borrower in years.",
        example=45,
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ...,
        ge=0,
        description="Number of times borrower has been 30-59 days past due.",
        example=2,
        alias="NumberOfTime30-59DaysPastDueNotWorse",
    )
    DebtRatio: float = Field(
        ...,
        ge=0.0,
        description="Monthly debt payments / monthly gross income.",
        example=0.803,
    )
    MonthlyIncome: Optional[float] = Field(
        None,
        ge=0.0,
        description="Monthly income in USD. Can be null (handled by imputer).",
        example=9120.0,
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ...,
        ge=0,
        description="Number of open loans and lines of credit.",
        example=13,
    )
    NumberOfTimes90DaysLate: int = Field(
        ...,
        ge=0,
        description="Number of times borrower has been 90+ days past due.",
        example=0,
    )
    NumberRealEstateLoansOrLines: int = Field(
        ...,
        ge=0,
        description="Number of mortgage and real estate loans.",
        example=6,
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ...,
        ge=0,
        description="Number of times borrower has been 60-89 days past due.",
        example=0,
        alias="NumberOfTime60-89DaysPastDueNotWorse",
    )
    NumberOfDependents: Optional[float] = Field(
        None,
        ge=0,
        description="Number of dependents in family (spouse, children, etc.).",
        example=2.0,
    )

    model_config = {"populate_by_name": True}


class NetworkIntrusionInput(BaseModel):
    """
    Input schema for network intrusion detection.

    Contains the core NSL-KDD features. Only the most impactful features
    are required; others are optional (the model pipeline imputes missing ones).
    """
    duration: float = Field(..., ge=0.0, description="Duration of connection in seconds.", example=0.0)
    protocol_type: str = Field(..., description="Network protocol: tcp, udp, or icmp.", example="tcp")
    service: str = Field(..., description="Network service (http, ftp, smtp, etc.)", example="http")
    flag: str = Field(..., description="Connection status flag (SF, S0, REJ, etc.)", example="SF")
    src_bytes: float = Field(..., ge=0.0, description="Bytes sent from source to destination.", example=215.0)
    dst_bytes: float = Field(..., ge=0.0, description="Bytes sent from destination to source.", example=45076.0)
    logged_in: int = Field(..., ge=0, le=1, description="1 if successfully logged in; 0 otherwise.", example=1)
    count: float = Field(..., ge=0.0, description="Connections to same host in past 2 seconds.", example=1.0)
    srv_count: float = Field(..., ge=0.0, description="Connections to same service in past 2 seconds.", example=1.0)
    same_srv_rate: float = Field(..., ge=0.0, le=1.0, description="% connections to same service.", example=1.0)
    diff_srv_rate: float = Field(..., ge=0.0, le=1.0, description="% connections to different services.", example=0.0)
    dst_host_count: float = Field(..., ge=0.0, description="Connections to same destination host.", example=9.0)
    dst_host_srv_count: float = Field(..., ge=0.0, description="Connections to same dest host+service.", example=9.0)
    serror_rate: float = Field(0.0, ge=0.0, le=1.0, description="% connections with SYN errors.", example=0.0)
    rerror_rate: float = Field(0.0, ge=0.0, le=1.0, description="% connections with REJ errors.", example=0.0)

    # Optional fields — will be imputed if missing
    land: Optional[int] = Field(0, ge=0, le=1)
    wrong_fragment: Optional[float] = Field(0.0, ge=0.0)
    urgent: Optional[float] = Field(0.0, ge=0.0)
    hot: Optional[float] = Field(0.0, ge=0.0)
    num_failed_logins: Optional[float] = Field(0.0, ge=0.0)
    num_compromised: Optional[float] = Field(0.0, ge=0.0)
    root_shell: Optional[int] = Field(0, ge=0, le=1)
    su_attempted: Optional[int] = Field(0, ge=0, le=1)
    num_root: Optional[float] = Field(0.0, ge=0.0)
    num_file_creations: Optional[float] = Field(0.0, ge=0.0)
    num_shells: Optional[float] = Field(0.0, ge=0.0)
    num_access_files: Optional[float] = Field(0.0, ge=0.0)
    num_outbound_cmds: Optional[float] = Field(0.0, ge=0.0)
    is_host_login: Optional[int] = Field(0, ge=0, le=1)
    is_guest_login: Optional[int] = Field(0, ge=0, le=1)
    srv_serror_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    srv_rerror_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    srv_diff_host_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_same_srv_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_diff_srv_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_same_src_port_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_srv_diff_host_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_serror_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_srv_serror_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_rerror_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    dst_host_srv_rerror_rate: Optional[float] = Field(0.0, ge=0.0, le=1.0)

    @field_validator("protocol_type")
    @classmethod
    def validate_protocol(cls, v: str) -> str:
        allowed = {"tcp", "udp", "icmp"}
        if v.lower() not in allowed:
            raise ValueError(f"protocol_type must be one of {allowed}")
        return v.lower()


class PredictionResponse(BaseModel):
    """
    Response schema for all prediction endpoints.

    Consistent response structure regardless of domain — the frontend
    doesn't need domain-specific parsing logic.
    """
    prediction: int = Field(..., description="Binary prediction: 0 (safe) or 1 (risky)")
    probability: float = Field(..., description="Probability of being class 1 (risky)")
    risk_label: str = Field(..., description="Human-readable label for the prediction")
    domain: str = Field(..., description="Which domain produced this prediction")
    threshold_used: float = Field(..., description="Decision threshold applied")


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    models_loaded: list[str]
    version: str
