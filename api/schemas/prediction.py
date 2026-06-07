"""
Pydantic request/response schemas for the prediction API.

CreditRiskInput  — UCI Default of Credit Card Clients (30,000 records, Sept 2005)
NetworkIntrusionInput — NSL-KDD network traffic features
PredictionResponse — shared response schema for all prediction endpoints
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CreditRiskInput(BaseModel):
    """
    Input schema for credit risk prediction.
    All monetary values are in New Taiwan Dollar (NTD).
    """

    LIMIT_BAL: float = Field(
        ..., ge=0,
        description="Credit limit amount (NTD). Typical range: 10,000 – 800,000.",
        example=200000.0,
    )
    SEX: int = Field(
        ..., ge=1, le=2,
        description="Gender: 1=male, 2=female.",
        example=2,
    )
    EDUCATION: int = Field(
        ..., ge=1, le=4,
        description="Education level: 1=graduate school, 2=university, 3=high school, 4=other.",
        example=2,
    )
    MARRIAGE: int = Field(
        ..., ge=1, le=3,
        description="Marital status: 1=married, 2=single, 3=other.",
        example=1,
    )
    AGE: int = Field(
        ..., ge=18, le=100,
        description="Age in years.",
        example=35,
    )
    PAY_0: int = Field(
        ..., ge=-2, le=9,
        description="Repayment status September 2005. -2=no consumption, -1=paid duly, "
                    "1=one month delay, 2=two month delay, ..., 9=nine+ month delay.",
        example=-1,
    )
    PAY_2: int = Field(..., ge=-2, le=9, description="Repayment status August 2005.", example=-1)
    PAY_3: int = Field(..., ge=-2, le=9, description="Repayment status July 2005.", example=-1)
    PAY_4: int = Field(..., ge=-2, le=9, description="Repayment status June 2005.", example=-1)
    PAY_5: int = Field(..., ge=-2, le=9, description="Repayment status May 2005.", example=-1)
    PAY_6: int = Field(..., ge=-2, le=9, description="Repayment status April 2005.", example=-1)
    BILL_AMT1: float = Field(..., description="Bill statement September 2005 (NTD).", example=3913.0)
    BILL_AMT2: float = Field(..., description="Bill statement August 2005 (NTD).", example=3102.0)
    BILL_AMT3: float = Field(..., description="Bill statement July 2005 (NTD).", example=689.0)
    BILL_AMT4: float = Field(..., description="Bill statement June 2005 (NTD).", example=0.0)
    BILL_AMT5: float = Field(..., description="Bill statement May 2005 (NTD).", example=0.0)
    BILL_AMT6: float = Field(..., description="Bill statement April 2005 (NTD).", example=0.0)
    PAY_AMT1: float = Field(..., ge=0, description="Payment amount September 2005 (NTD).", example=0.0)
    PAY_AMT2: float = Field(..., ge=0, description="Payment amount August 2005 (NTD).", example=689.0)
    PAY_AMT3: float = Field(..., ge=0, description="Payment amount July 2005 (NTD).", example=0.0)
    PAY_AMT4: float = Field(..., ge=0, description="Payment amount June 2005 (NTD).", example=0.0)
    PAY_AMT5: float = Field(..., ge=0, description="Payment amount May 2005 (NTD).", example=0.0)
    PAY_AMT6: float = Field(..., ge=0, description="Payment amount April 2005 (NTD).", example=0.0)


class NetworkIntrusionInput(BaseModel):
    """
    Input schema for network intrusion detection.
    Contains core NSL-KDD features; optional fields default to 0.
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
    """Response schema for all prediction endpoints."""
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
