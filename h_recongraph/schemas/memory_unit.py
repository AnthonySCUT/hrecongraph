"""Unified memory unit schema shared by M0, M1, and M2."""

from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from graphiti_core.utils.datetime_utils import utc_now

MemoryLayer = Literal['M0', 'M1', 'M2']
MemoryDimension = Literal['fact', 'constraint', 'path', 'causal']
ContentType = Literal['episode', 'edge', 'subgraph', 'semantic_state']
LifecycleStatus = Literal[
    'volatile',
    'active_short_term',
    'candidate',
    'consolidated',
    'revised',
    'expired',
    'archived',
    'deleted',
]


class SourceSpan(BaseModel):
    """Evidence span within a raw interaction."""

    evidence_id: str
    text: str
    start: int | None = None
    end: int | None = None
    field: str | None = None


class GraphBinding(BaseModel):
    """Graph nodes, edges, and subgraph identifiers bound to a memory unit."""

    nodes: list[str] = Field(default_factory=list)
    edges: list[str] = Field(default_factory=list)
    subgraph_id: str | None = None


class MemoryScores(BaseModel):
    """Operational scores used by evolution and retrieval policies."""

    salience: float = 0.0
    importance: float = 0.0
    recency: float = 1.0
    confidence: float = 0.0
    retrieval_utility: float = 0.0
    task_relevance: float = 0.0
    evidence_strength: float = 0.0
    contradiction_risk: float = 0.0
    privacy_risk: float = 0.0
    user_confirmation: float = 0.0


class UnifiedMemoryUnit(BaseModel):
    """A single H-ReConGraph memory unit."""

    memory_id: str = Field(default_factory=lambda: f'mem_{uuid4().hex}')
    user_id: str
    session_id: str | None = None
    layer: MemoryLayer
    content_type: ContentType = 'episode'
    raw_content: str | None = None
    graph_binding: GraphBinding = Field(default_factory=GraphBinding)
    semantic_state_id: str | None = None
    semantic_state: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    lifecycle_status: LifecycleStatus = 'volatile'
    scores: MemoryScores = Field(default_factory=MemoryScores)
    created_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime | None = None
    access_count: int = 0
    version: int = 1
    supersedes: list[str] = Field(default_factory=list)
    ttl: timedelta | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_span_updates(
        self,
        span_updates: dict[MemoryDimension, list[SourceSpan]] | None = None,
        append_spans: list[SourceSpan] | None = None,
    ) -> 'UnifiedMemoryUnit':
        """Merge changed dimensions into the current bundle, retaining other evidence."""
        updates = span_updates or {}
        spans = [
            span.model_copy(update={'field': dimension}, deep=True)
            for dimension in ('fact', 'constraint', 'path', 'causal')
            for span in updates.get(
                dimension, [span for span in self.source_spans if span.field == dimension]
            )
        ]
        spans.extend(span.model_copy(deep=True) for span in append_spans or [])
        spans = list(
            {
                (span.evidence_id, span.field, span.text, span.start, span.end): span
                for span in spans
            }.values()
        )
        revised = self.model_copy(deep=True)
        revised.raw_content = '\n\n'.join(dict.fromkeys(span.text for span in spans))
        revised.source_spans = spans
        revised.evidence_ids = list(dict.fromkeys(span.evidence_id for span in spans))
        revised.created_at = utc_now()
        return revised

    def touch(self, at: datetime | None = None) -> None:
        self.last_accessed_at = at or utc_now()
        self.access_count += 1

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.ttl is None:
            return False
        return (now or utc_now()) - self.created_at > self.ttl
