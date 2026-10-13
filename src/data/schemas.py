"""Pydantic data schemas for NEXUS-SLM."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class TicketRecord(BaseModel):
    """Schema for customer support ticket dataset records."""
    id: str = Field(description="Unique identifier for the ticket")
    text: str = Field(min_length=1, description="Raw ticket text from customer")
    language: Literal["english", "hindi", "hinglish"] = Field(
        description="Language modality of the ticket"
    )
    category: Literal[
        "payment_failure",
        "refund_request",
        "account_access",
        "password_reset",
        "order_cancel",
        "order_tracking",
        "subscription",
        "billing",
        "technical_issue",
        "fraud_report",
        "delivery_issue",
        "account_closure",
        "product_complaint",
        "general_query",
    ] = Field(description="Triage category classification")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Resolution priority tier"
    )
    requires_human: bool = Field(
        description="Whether this ticket demands human escalation"
    )
    source: Literal["synthetic", "curated", "public", "human_reviewed"] = Field(
        default="synthetic",
        description="Origin of the example"
    )
    validated: bool = Field(
        default=False,
        description="Whether this sample has passed human or expert schema audit"
    )
    created_at: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of record creation"
    )

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language_field(cls, v: Any) -> Any:
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("roman_hindi", "romanhindi"):
                return "hinglish"
            return clean
        return v

    @field_validator("text")
    def text_must_not_be_blank(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Ticket text cannot be empty or purely whitespace")
        return s


class TriagePrediction(BaseModel):
    """Structured prediction returned by NEXUS-SLM inference."""
    category: str = Field(description="Predicted category")
    priority: str = Field(description="Predicted priority")
    language: str = Field(description="Predicted language")
    requires_human: bool = Field(description="Predicted human escalation necessity")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score from model token probabilities, or None if unavailable")
    latency_ms: Optional[float] = Field(default=None, description="Inference latency in milliseconds")
    model_version: Optional[str] = Field(default="nexus-v1.0", description="Model version tag")
    raw_output: Optional[str] = Field(default=None, description="Raw generation string before validation")
    is_valid: bool = Field(default=True, description="Whether raw output successfully validated against schema")
    error_message: Optional[str] = Field(default=None, description="Schema or validation error message if invalid")
    raw_parsed: Optional[Dict[str, Any]] = Field(default=None, description="Raw unnormalized parsed dictionary from generation")


class ValidationResult(BaseModel):
    """Result of data validation inspection."""
    valid: bool
    record_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DatasetQualityReport(BaseModel):
    """Comprehensive dataset statistics and quality audit."""
    dataset_version: str
    total_examples: int
    examples_per_category: Dict[str, int]
    examples_per_language: Dict[str, int]
    examples_per_priority: Dict[str, int]
    synthetic_count: int
    synthetic_percentage: float
    human_reviewed_count: int
    human_reviewed_percentage: float
    curated_count: int
    duplicate_count: int
    invalid_record_count: int
    mean_char_length: float
    min_char_length: int
    max_char_length: int
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class LeakageReport(BaseModel):
    """Leakage analysis between splits (Train, Validation, Test)."""
    dataset_version: str
    train_count: int
    val_count: int
    test_count: int
    exact_duplicates_across_splits: int
    near_duplicates_flagged: int
    max_similarity_score: float
    template_leakages: List[Dict[str, Any]] = Field(default_factory=list)
    leakage_detected: bool
    status: str
    inspected_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
