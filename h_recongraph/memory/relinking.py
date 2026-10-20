"""Query-time cross-layer relinking and context rendering."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from graphiti_core.nodes import EpisodicNode
from h_recongraph.schemas.memory_edge import ReConMemoryEdge
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit
from h_recongraph.schemas.semantic_state import MemorySemanticState


class HierarchicalMemoryContext(BaseModel):
    """Query-specific hierarchical memory evidence graph."""

    query: str
    selected_layers: list[Literal['M0', 'M1', 'M2']]
    instant_memories: list[UnifiedMemoryUnit] = Field(default_factory=list)
    graph_edges: list[Any] = Field(default_factory=list)
    semantic_states: list[MemorySemanticState] = Field(default_factory=list)
    ephemeral_edges: list[ReConMemoryEdge] = Field(default_factory=list)
    evidence_trace: list[str] = Field(default_factory=list)
    source_episodes: list[EpisodicNode] = Field(default_factory=list)
    context_text: str = ''

    model_config = {'arbitrary_types_allowed': True}

    def render_text(self) -> str:
        """Render the selected memory evidence as compact text for a downstream reader."""
        blocks: list[str] = []

        for memory in self.instant_memories:
            if memory.raw_content:
                blocks.append(f'[M0] {memory.raw_content}')

        for state in self.semantic_states:
            fields = [
                f'{name}={value}'
                for name in (
                    'temporal_scope',
                    'cause',
                    'effect',
                    'intent',
                    'condition',
                    'state_change',
                    'validity',
                )
                if (value := getattr(state, name))
            ]
            if fields:
                evidence = ','.join(state.evidence_ids)
                suffix = f' [evidence={evidence}]' if evidence else ''
                blocks.append(f'[STATE {state.edge_or_subgraph_id}] ' + '; '.join(fields) + suffix)

        for edge in self.graph_edges:
            fact = getattr(edge, 'fact', None) or getattr(edge, 'name', None)
            if not fact:
                continue
            source = getattr(edge, 'source_node_uuid', '?')
            target = getattr(edge, 'target_node_uuid', '?')
            blocks.append(f'[EDGE] {source} -> {target}: {fact}')

        for episode in self.source_episodes:
            content = self._source_text(episode.content)
            if content:
                blocks.append(f'[SOURCE {episode.uuid}] {content}')

        return '\n'.join(dict.fromkeys(blocks))

    @staticmethod
    def _source_text(content: str) -> str:
        try:
            envelope = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return content
        if isinstance(envelope, dict) and isinstance(envelope.get('content'), str):
            return envelope['content']
        return content


class CrossLayerRelinker:
    """Construct ephemeral edges when M0 evidence can bridge retrieved graph context."""

    def build_context(
        self,
        query: str,
        selected_layers: list[Literal['M0', 'M1', 'M2']],
        instant_memories: list[UnifiedMemoryUnit],
        graph_edges: list[Any],
        semantic_states: list[MemorySemanticState],
    ) -> HierarchicalMemoryContext:
        ephemeral_edges = self._candidate_edges(query, instant_memories)
        evidence_trace = []
        for memory in instant_memories:
            evidence_trace.extend(memory.evidence_ids)
        for state in semantic_states:
            evidence_trace.extend(state.evidence_ids)
        for edge in ephemeral_edges:
            evidence_trace.extend(edge.evidence_ids)

        context = HierarchicalMemoryContext(
            query=query,
            selected_layers=selected_layers,
            instant_memories=instant_memories,
            graph_edges=graph_edges,
            semantic_states=semantic_states,
            ephemeral_edges=ephemeral_edges,
            evidence_trace=list(dict.fromkeys(evidence_trace)),
        )
        context.context_text = context.render_text()
        return context

    def _candidate_edges(
        self, query: str, instant_memories: list[UnifiedMemoryUnit]
    ) -> list[ReConMemoryEdge]:
        query_tokens = {token for token in query.lower().split() if len(token) > 1}
        edges: list[ReConMemoryEdge] = []
        for memory in instant_memories:
            text = memory.raw_content or ''
            memory_tokens = {token for token in text.lower().split() if len(token) > 1}
            overlap = query_tokens & memory_tokens
            if not overlap:
                continue
            edges.append(
                ReConMemoryEdge(
                    source='query',
                    relation='evidence_relinks_to',
                    target=memory.memory_id,
                    layer='M1',
                    edge_type='ephemeral',
                    confidence=min(0.95, 0.4 + 0.1 * len(overlap)),
                    evidence_ids=memory.evidence_ids,
                    lifecycle_status='candidate',
                    write_back=False,
                )
            )
        return edges
