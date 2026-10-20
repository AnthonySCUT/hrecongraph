"""Evidence-bounded semantic state schema."""

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from h_recongraph.schemas.memory_unit import SourceSpan

SemanticStateType = Literal[
    'evidence_supported',
    'soft_inference',
    'ephemeral_relink',
    'unsupported',
]


class MemorySemanticState(BaseModel):
    """Reasoning-oriented state bound to explicit evidence spans."""

    state_id: str = Field(default_factory=lambda: f'state_{uuid4().hex}')
    edge_or_subgraph_id: str
    temporal_scope: str | None = None
    cause: str | None = None
    effect: str | None = None
    intent: str | None = None
    condition: str | None = None
    state_change: str | None = None
    validity: str | None = None
    confidence: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    state_type: SemanticStateType = 'unsupported'

    @model_validator(mode='after')
    def evidence_bounds_non_empty_fields(self) -> 'MemorySemanticState':
        if self.state_type == 'unsupported':
            return self

        fields = {
            'temporal_scope': self.temporal_scope,
            'cause': self.cause,
            'effect': self.effect,
            'intent': self.intent,
            'condition': self.condition,
            'state_change': self.state_change,
            'validity': self.validity,
        }
        missing = [
            name
            for name, value in fields.items()
            if value is not None and not any(span.field == name for span in self.source_spans)
        ]
        if missing:
            raise ValueError(
                'semantic state fields must be grounded by source spans: ' + ', '.join(missing)
            )
        return self
