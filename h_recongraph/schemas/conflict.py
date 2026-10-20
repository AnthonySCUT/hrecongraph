"""Conflict and memory update schemas."""

from typing import Literal

from pydantic import BaseModel

MemoryRelationType = Literal[
    'duplicate',
    'elaboration',
    'temporal_update',
    'contradiction',
    'exception',
    'new_fact',
]


class ConflictAssessment(BaseModel):
    """Relationship between a new memory and an existing memory."""

    relation_type: MemoryRelationType
    confidence: float
    contradiction_risk: float = 0.0
    matched_memory_id: str | None = None
    rationale: str = ''
    should_version: bool = False
    should_merge: bool = False
