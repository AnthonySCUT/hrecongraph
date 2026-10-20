"""H-ReConGraph memory edge schema."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

EdgeLayer = Literal['M1', 'M2']
MemoryEdgeType = Literal['hard', 'soft', 'ephemeral']
MemoryEdgeStatus = Literal[
    'active', 'candidate', 'consolidated', 'superseded', 'expired', 'archived'
]


class ReConMemoryEdge(BaseModel):
    """A Graphiti-compatible relation enriched with H-ReConGraph lifecycle metadata."""

    edge_id: str = Field(default_factory=lambda: f'edge_{uuid4().hex}')
    source: str
    relation: str
    target: str
    layer: EdgeLayer
    edge_type: MemoryEdgeType = 'soft'
    confidence: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    semantic_state_id: str | None = None
    lifecycle_status: MemoryEdgeStatus = 'candidate'
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    supersedes: list[str] = Field(default_factory=list)
    write_back: bool = False
