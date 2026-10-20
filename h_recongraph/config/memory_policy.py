"""Policy thresholds and weights for H-ReConGraph memory evolution."""

from datetime import timedelta

from pydantic import BaseModel, Field


class MemoryPolicy(BaseModel):
    """Rule-based policy for promotion, consolidation, and forgetting."""

    m0_ttl: timedelta = Field(default_factory=lambda: timedelta(hours=4))
    m1_ttl: timedelta = Field(default_factory=lambda: timedelta(days=7))
    promotion_salience_threshold: float = 0.58
    promotion_task_relevance_threshold: float = 0.55
    consolidation_threshold: float = 0.74
    archive_threshold: float = 0.68

    frequency_weight: float = 0.18
    recency_weight: float = 0.12
    importance_weight: float = 0.2
    evidence_strength_weight: float = 0.18
    retrieval_usage_weight: float = 0.14
    user_confirmation_weight: float = 0.16
    contradiction_risk_weight: float = 0.18

    age_forgetting_weight: float = 0.18
    low_usage_forgetting_weight: float = 0.16
    low_confidence_forgetting_weight: float = 0.16
    contradiction_forgetting_weight: float = 0.18
    privacy_forgetting_weight: float = 0.2
    importance_retention_weight: float = 0.18
    evidence_retention_weight: float = 0.18
    user_pin_retention_weight: float = 0.2

    max_retrieval_results: int = 10
    write_back_ephemeral_edges: bool = False
