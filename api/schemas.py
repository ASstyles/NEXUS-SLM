"""Pydantic schemas for NEXUS-SLM FastAPI inference service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Customer support ticket input request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Raw text of the customer support ticket",
        examples=["Mera payment kat gaya but order confirm nahi hua."],
    )


class PredictResponse(BaseModel):
    """Structured triage output response."""
    category: str = Field(description="Predicted ticket category")
    priority: str = Field(description="Predicted urgency priority level")
    language: str = Field(description="Detected language modality")
    requires_human: bool = Field(description="Whether escalation to a human agent is recommended")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from model token probabilities")
    latency_ms: float = Field(description="Inference execution latency in milliseconds")
    model_version: str = Field(description="Model version tag executing the prediction")
    raw_output: Optional[str] = Field(default=None, description="Raw generation string")


class HealthResponse(BaseModel):
    """System health check and diagnostic status."""
    status: str
    model_loaded: bool
    model_version: str
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
