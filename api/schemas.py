"""Strict Pydantic schemas for NEXUS-SLM FastAPI inference service."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class SupportCategory(str, Enum):
    PAYMENT_FAILURE = "payment_failure"
    REFUND_REQUEST = "refund_request"
    ACCOUNT_ACCESS = "account_access"
    PASSWORD_RESET = "password_reset"
    ORDER_CANCEL = "order_cancel"
    ORDER_TRACKING = "order_tracking"
    SUBSCRIPTION = "subscription"
    BILLING = "billing"
    TECHNICAL_ISSUE = "technical_issue"
    FRAUD_REPORT = "fraud_report"
    DELIVERY_ISSUE = "delivery_issue"
    ACCOUNT_CLOSURE = "account_closure"
    PRODUCT_COMPLAINT = "product_complaint"
    GENERAL_QUERY = "general_query"


class SupportPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportLanguage(str, Enum):
    ENGLISH = "english"
    HINDI = "hindi"
    HINGLISH = "hinglish"


class PredictRequest(BaseModel):
    """Customer support ticket input request matching strict inference contract."""
    ticket: Optional[str] = Field(
        default=None,
        description="Raw text of the customer support ticket (primary contract field)",
        examples=["Mera payment kat gaya but order confirm nahi hua."],
    )
    text: Optional[str] = Field(
        default=None,
        description="Alternative field name accepted for backward compatibility",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_ticket_content(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON dictionary object.")

        # Accept either 'ticket' or 'text'
        content = data.get("ticket") if data.get("ticket") is not None else data.get("text")
        if content is None:
            raise ValueError("Missing required input field: 'ticket'")

        if not isinstance(content, str):
            raise ValueError("Ticket content must be a valid string.")

        cleaned = content.strip()
        if len(cleaned) == 0:
            raise ValueError("Ticket text cannot be empty or whitespace only.")

        if len(cleaned) > 4000:
            raise ValueError(f"Ticket text length ({len(cleaned)}) exceeds maximum allowed limit (4000 characters).")

        data["ticket"] = cleaned
        data["text"] = cleaned
        return data


class PredictResponse(BaseModel):
    """Strict structured triage output response meeting production inference contract."""
    category: SupportCategory = Field(description="Predicted ticket category from 14-class taxonomy")
    priority: SupportPriority = Field(description="Predicted urgency priority level")
    language: SupportLanguage = Field(description="Detected language modality")
    requires_human: bool = Field(description="Whether escalation to a human agent is recommended")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Heuristic model confidence score in [0.0, 1.0], or None when log-probabilities are unavailable")
    
    # Operational & observability metadata fields
    latency_ms: Optional[float] = Field(default=None, description="Inference execution latency in milliseconds")
    model_version: Optional[str] = Field(default="nexus-qlora-v1", description="Model version executing prediction")
    inference_mode: Optional[str] = Field(default="neural", description="Inference engine type: 'neural' or 'heuristic_fallback'")
    request_id: Optional[str] = Field(default=None, description="Unique trace identifier for observability")
    calibrated_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Post-hoc calibrated confidence score if available")
    confidence_status: Optional[str] = Field(default="uncalibrated_heuristic", description="Confidence calibration reliability status")
    escalation_reason: Optional[str] = Field(default=None, description="Safety policy trigger reasoning")
    raw_output: Optional[str] = Field(default=None, description="Raw generation string for debugging")


class StructuredErrorResponse(BaseModel):
    """Deterministic structured error payload for API failures."""
    error: str = Field(description="High-level error classification")
    error_code: str = Field(description="Machine-readable error identifier")
    detail: Optional[str] = Field(default=None, description="Detailed failure diagnostics")
    request_id: Optional[str] = Field(default=None, description="Unique trace request ID")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of error event")


class HealthResponse(BaseModel):
    """System health check and diagnostic status."""
    status: str
    model_loaded: bool
    model_version: str
    inference_mode: str = Field(description="Active inference mode: 'neural' or 'heuristic_fallback'")
    hardware: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Active model metadata and deployment parameters."""
    active_version: str
    base_model: str
    quantization: str
    model_hash: str
    deployment_status: str
    dataset_version: str
    metrics: Dict[str, Any]
