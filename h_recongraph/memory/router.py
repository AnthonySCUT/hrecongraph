"""Query router for H-ReConGraph retrieval."""

from typing import Literal

from pydantic import BaseModel, Field

MemoryNeed = Literal[
    'immediate_context',
    'recent_task_state',
    'long_term_preference',
    'temporal_reasoning',
    'causal_reasoning',
    'conflict_checking',
    'audit',
]


class QueryRoute(BaseModel):
    needs: list[MemoryNeed] = Field(default_factory=list)
    layers: list[Literal['M0', 'M1', 'M2']] = Field(default_factory=list)


class QueryRouter:
    """Rule-based layer selector for the first H-ReConGraph implementation."""

    RECENT_TERMS = ('刚才', '当前', '最近', '这次', 'today', 'recent', 'current')
    LONG_TERM_TERMS = ('以后', '长期', '偏好', '习惯', 'always', 'prefer', 'usually')
    TEMPORAL_TERMS = ('什么时候', '之前', '之后', '变化', 'when', 'before', 'after', 'changed')
    CAUSAL_TERMS = ('为什么', '原因', '导致', 'because', 'why', 'cause')
    AUDIT_TERMS = ('证据', '来源', '原文', 'evidence', 'source')
    CONFLICT_TERMS = ('冲突', '矛盾', '不一致', 'conflict', 'contradiction')

    def route(self, query: str) -> QueryRoute:
        lower = query.lower()
        needs: list[MemoryNeed] = []
        layers: list[Literal['M0', 'M1', 'M2']] = []

        if self._contains_any(lower, self.RECENT_TERMS):
            needs.append('recent_task_state')
            layers.extend(['M0', 'M1'])
        if self._contains_any(lower, self.LONG_TERM_TERMS):
            needs.append('long_term_preference')
            layers.append('M2')
        if self._contains_any(lower, self.TEMPORAL_TERMS):
            needs.append('temporal_reasoning')
            layers.extend(['M1', 'M2'])
        if self._contains_any(lower, self.CAUSAL_TERMS):
            needs.append('causal_reasoning')
            layers.extend(['M0', 'M1', 'M2'])
        if self._contains_any(lower, self.AUDIT_TERMS):
            needs.append('audit')
            layers.extend(['M0', 'M1', 'M2'])
        if self._contains_any(lower, self.CONFLICT_TERMS):
            needs.append('conflict_checking')
            layers.extend(['M1', 'M2'])

        if not needs:
            needs = ['immediate_context', 'long_term_preference']
            layers = ['M0', 'M1', 'M2']

        return QueryRoute(needs=list(dict.fromkeys(needs)), layers=list(dict.fromkeys(layers)))

    def _contains_any(self, text: str, terms: tuple[str, ...]) -> bool:
        return any(term.lower() in text for term in terms)
