"""Query-time cross-layer relinking and context rendering."""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from graphiti_core.nodes import EpisodicNode
from h_recongraph.memory.ranker import MemoryUtilityRanker
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
    source_units: dict[str, list[str]] = Field(default_factory=dict)
    source_fact_indices: dict[str, list[int]] = Field(default_factory=dict)
    source_origin_ids: dict[str, str] = Field(default_factory=dict)
    context_text: str = ''
    source_mode: Literal['full_source', 'evidence_bundle'] = 'full_source'

    model_config = {'arbitrary_types_allowed': True}

    def render_text(
        self,
        source_mode: Literal['full_source', 'evidence_bundle'] | None = None,
        *,
        window_size: int = 0,
        dependency_mode: Literal['all', 'lexical', 'none'] = 'all',
        window_scope: Literal['bundle', 'fact'] = 'bundle',
        deduplicate_edge_facts: bool = True,
    ) -> str:
        """Render the selected memory evidence as compact text for a downstream reader."""
        source_mode = source_mode or self.source_mode
        sources = []
        for episode in self.source_episodes:
            if source_mode == 'evidence_bundle':
                text = self._source_bundle(
                    episode,
                    window_size=window_size,
                    dependency_mode=dependency_mode,
                    window_scope=window_scope,
                )
            elif episode.uuid in self.source_units:
                text = self._render_source_units(episode, range(len(self.source_segments(episode))))
            else:
                text = self._source_text(episode.content)
            sources.append((episode.uuid, text))
        if source_mode == 'evidence_bundle':
            seen_sources = set()
            unique_sources = []
            for episode_id, text in sources:
                origin = self.source_origin_ids.get(episode_id, episode_id)
                key = (origin, text) if episode_id in self.source_units else text
                if text and key not in seen_sources:
                    unique_sources.append((episode_id, text))
                    seen_sources.add(key)
            sources = unique_sources
        blocks: list[str] = []

        for memory in self.instant_memories:
            if memory.raw_content:
                if source_mode == 'evidence_bundle' and any(
                    memory.raw_content == text for _, text in sources
                ):
                    continue
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

        seen_edge_facts = set()
        for edge in self.graph_edges:
            fact = getattr(edge, 'fact', None) or getattr(edge, 'name', None)
            if not fact:
                continue
            if source_mode == 'evidence_bundle' and fact in seen_edge_facts:
                continue
            seen_edge_facts.add(fact)
            source = getattr(edge, 'source_node_uuid', '?')
            target = getattr(edge, 'target_node_uuid', '?')
            covered_source = next(
                (episode_id for episode_id, text in sources if fact in text), None
            )
            edge_text = fact
            if deduplicate_edge_facts and covered_source is not None:
                relation = getattr(edge, 'name', None) or 'RELATION'
                reference = f'{relation} [source={covered_source}]'
                if len(reference) < len(fact):
                    edge_text = reference
            blocks.append(f'[EDGE] {source} -> {target}: {edge_text}')

        for episode_id, content in sources:
            if content:
                blocks.append(f'[SOURCE {episode_id}] {content}')

        return '\n'.join(dict.fromkeys(blocks))

    def source_indices(
        self,
        episode: EpisodicNode,
        window_size: int = 0,
        *,
        dependency_mode: Literal['all', 'lexical', 'none'] = 'all',
        window_scope: Literal['bundle', 'fact'] = 'bundle',
    ) -> dict[str, list[int]]:
        """Select source-unit indexes, expand dependencies, then union local windows."""
        content = self._source_text(episode.content)
        paragraphs = self.source_segments(episode)
        facts = [edge.fact for edge in self.graph_edges if episode.uuid in edge.episodes]
        spans = [
            span
            for state in self.semantic_states
            for span in state.source_spans
            if span.field != 'raw' and span.text and span.text in content
        ]
        explicit_spans = [span for span in spans if span.start is None or span.end is None]
        fact_anchors = [span.text for span in spans if span.field == 'fact']
        if not facts or (any(fact not in content for fact in facts) and not fact_anchors):
            all_indexes = list(range(len(paragraphs)))
            return {'fact': [], 'dependencies': [], 'window': [], 'fallback': all_indexes}

        fact_indexes = set(self.source_fact_indices.get(episode.uuid, []))
        for fact in facts if episode.uuid not in self.source_fact_indices else []:
            literal = {index for index, paragraph in enumerate(paragraphs) if fact in paragraph}
            if literal:
                fact_indexes.update(literal)
                continue
            fact_tokens = self._tokens(fact)
            overlaps = [
                (len(fact_tokens.intersection(self._tokens(paragraph))), index)
                for index, paragraph in enumerate(paragraphs)
            ]
            score, index = max(overlaps, default=(0, 0))
            if score:
                fact_indexes.add(index)

        if not fact_indexes:
            fact_indexes.update(
                index
                for index, paragraph in enumerate(paragraphs)
                if any(anchor in paragraph for anchor in fact_anchors)
            )

        offsets, cursor = [], 0
        for paragraph in paragraphs:
            start = content.find(paragraph, cursor)
            offsets.append((start, start + len(paragraph)))
            cursor = start + len(paragraph)
        active_fact_starts = {
            span.start
            for span in spans
            if span.field == 'fact'
            and span.start is not None
            and any(offsets[index][0] <= span.start < offsets[index][1] for index in fact_indexes)
        }
        spans = [
            span
            for span in spans
            if span.fact_starts is None or active_fact_starts.intersection(span.fact_starts)
        ]
        explicit_spans = [span for span in spans if span.start is None or span.end is None]
        dependency_indexes = {
            index
            for index, paragraph in enumerate(paragraphs)
            if any(span.text in paragraph for span in explicit_spans)
            or any(
                span.field != 'fact' and span.text in paragraph
                for span in spans
                if span.field != 'raw'
            )
        }
        if dependency_mode == 'none':
            dependency_indexes = set()
        elif dependency_mode == 'lexical':
            ranker = MemoryUtilityRanker()
            unit_tokens = [
                ranker.tokenize(re.sub(r'^(?:user|assistant|system):\s*', '', unit, flags=re.I))
                for unit in paragraphs
            ]
            anchor_tokens = ranker.tokenize(self.query)
            for index in fact_indexes:
                anchor_tokens.update(unit_tokens[index])
            dependency_indexes = {
                index
                for index in dependency_indexes
                if unit_tokens[index] & anchor_tokens
                or any(span.text in paragraphs[index] for span in explicit_spans)
            }
        base_indexes = fact_indexes | dependency_indexes
        window_anchors = fact_indexes if window_scope == 'fact' else base_indexes
        window_indexes = {
            neighbor
            for index in window_anchors
            for neighbor in range(
                max(0, index - window_size), min(len(paragraphs), index + window_size + 1)
            )
        }
        return {
            'fact': sorted(fact_indexes),
            'dependencies': sorted(dependency_indexes),
            'window': sorted(window_indexes - base_indexes),
            'fallback': [],
        }

    def source_segments(self, episode: EpisodicNode) -> list[str]:
        """Use supplied message boundaries, otherwise paragraph boundaries."""
        if episode.uuid in self.source_units:
            return self.source_units[episode.uuid]
        return re.split(r'\n\s*\n', self._source_text(episode.content))

    def _render_source_units(self, episode: EpisodicNode, selected: set[int] | range) -> str:
        return '\n\n'.join(
            f'[TURN {index}] {unit}' if episode.uuid in self.source_units else unit
            for index, unit in enumerate(self.source_segments(episode))
            if index in selected
        )

    def _source_bundle(
        self,
        episode: EpisodicNode,
        *,
        window_size: int = 0,
        dependency_mode: Literal['all', 'lexical', 'none'] = 'all',
        window_scope: Literal['bundle', 'fact'] = 'bundle',
    ) -> str:
        """Keep complete paragraphs anchored by retrieved facts and semantic evidence."""
        indexes = self.source_indices(
            episode, window_size, dependency_mode=dependency_mode, window_scope=window_scope
        )
        if indexes['fallback'] and episode.uuid not in self.source_units:
            return self._source_text(episode.content)
        selected = set().union(*indexes.values())
        return self._render_source_units(episode, selected)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in re.findall(r'[\w-]+', text) if len(token) > 1}

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
