"""Main H-ReConGraph memory controller."""

import json
import re
from typing import Any, Literal

from graphiti_core.utils.datetime_utils import ensure_utc
from h_recongraph.config.memory_policy import MemoryPolicy
from h_recongraph.memory.conflict import ConflictDetector
from h_recongraph.memory.evolution import MemoryEvolutionEngine
from h_recongraph.memory.graphiti_store import GraphitiMemoryStore
from h_recongraph.memory.instant_buffer import InstantMemoryBuffer
from h_recongraph.memory.llm_evaluator import LLMMemoryEvaluator
from h_recongraph.memory.ranker import MemoryUtilityRanker
from h_recongraph.memory.relinking import CrossLayerRelinker, HierarchicalMemoryContext
from h_recongraph.memory.router import QueryRouter
from h_recongraph.memory.semantic_state import SemanticStateConstructor
from h_recongraph.memory.versioning import MemoryVersionManager
from h_recongraph.schemas.conflict import ConflictAssessment
from h_recongraph.schemas.memory_unit import (
    GraphBinding,
    MemoryDimension,
    SourceSpan,
    UnifiedMemoryUnit,
)
from h_recongraph.schemas.semantic_state import MemorySemanticState


class HReConGraphMemory:
    """Layered memory controller that uses Graphiti as the graph engine.

    The controller implements the H-ReConGraph write path:
    Interaction -> M0 -> M1 -> optional M2, plus query-time cross-layer relinking.
    """

    def __init__(
        self,
        graphiti_client: Any | None = None,
        *,
        buffer: InstantMemoryBuffer | None = None,
        policy: MemoryPolicy | None = None,
        store: GraphitiMemoryStore | None = None,
        evolution: MemoryEvolutionEngine | None = None,
        semantic_state_constructor: SemanticStateConstructor | None = None,
        router: QueryRouter | None = None,
        ranker: MemoryUtilityRanker | None = None,
        relinker: CrossLayerRelinker | None = None,
        conflict_detector: ConflictDetector | None = None,
        llm_evaluator: LLMMemoryEvaluator | None = None,
        version_manager: MemoryVersionManager | None = None,
    ):
        self.policy = policy or MemoryPolicy()
        self.buffer = buffer or InstantMemoryBuffer(self.policy)
        self.store = store or (
            GraphitiMemoryStore(graphiti_client) if graphiti_client is not None else None
        )
        self.evolution = evolution or MemoryEvolutionEngine(self.policy)
        self.semantic_state_constructor = semantic_state_constructor or SemanticStateConstructor()
        self.router = router or QueryRouter()
        self.ranker = ranker or MemoryUtilityRanker()
        self.relinker = relinker or CrossLayerRelinker()
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.llm_evaluator = llm_evaluator or LLMMemoryEvaluator()
        self.version_manager = version_manager or MemoryVersionManager()

        self.short_term_memories: dict[str, UnifiedMemoryUnit] = {}
        self.long_term_memories: dict[str, UnifiedMemoryUnit] = {}
        self.semantic_states: dict[str, MemorySemanticState] = {}

    async def add_interaction(
        self,
        text: str,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        source_spans: list[SourceSpan] | None = None,
    ) -> UnifiedMemoryUnit:
        """Write a new interaction through M0, then promote/consolidate by policy."""

        instant = self.buffer.add(text, user_id, session_id, metadata)
        instant.source_spans.extend(source_spans or [])
        instant.evidence_ids = list(
            dict.fromkeys(
                [*instant.evidence_ids, *(span.evidence_id for span in source_spans or [])]
            )
        )
        instant.scores = self.evolution.score_m0(instant)

        if not self.evolution.should_promote_to_m1(instant):
            return instant

        short_term = self.evolution.promote_to_m1(instant)
        short_term_assessment = await self._assess_against_existing(
            short_term,
            [*self.short_term_memories.values(), *self.long_term_memories.values()],
        )
        self._apply_assessment_metadata(short_term, short_term_assessment)
        await self.add_memory(short_term)

        if self.evolution.should_consolidate_to_m2(short_term):
            await self._consolidate_memory(short_term)

        return short_term

    async def add_memory(self, memory: UnifiedMemoryUnit) -> UnifiedMemoryUnit:
        """Write one M1/M2 bundle with its fact, constraint, path and causal evidence."""
        memory.evidence_ids = list(
            dict.fromkeys(
                [*memory.evidence_ids, *(span.evidence_id for span in memory.source_spans)]
            )
        )
        state = self.semantic_state_constructor.construct(memory)
        memory.semantic_state_id = state.state_id
        memory.semantic_state = state.model_dump(mode='json')
        if not any(
            span.field in {'fact', 'constraint', 'path', 'causal'} for span in memory.source_spans
        ):
            memory.source_spans = [
                *(span for span in memory.source_spans if span.field == 'raw'),
                *state.source_spans,
            ]
        target = self.short_term_memories if memory.layer == 'M1' else self.long_term_memories
        target[memory.memory_id] = memory
        self.semantic_states[state.state_id] = state
        if self.store is not None:
            await self.store.add_memory(memory)
        return memory

    async def update_memory(
        self,
        memory_id: str,
        *,
        operation: Literal['modify', 'supersede', 'delete'],
        replacement: UnifiedMemoryUnit | None = None,
        span_updates: dict[MemoryDimension, list[SourceSpan]] | None = None,
        append_spans: list[SourceSpan] | None = None,
    ) -> UnifiedMemoryUnit | None:
        """Apply a local operation to a bundle and its M0/M1/M2 lineage."""
        memories = {**self.short_term_memories, **self.long_term_memories}
        target = memories[memory_id]
        if operation == 'modify' and replacement is None:
            replacement = target.with_span_updates(span_updates, append_spans)
        family_ids = {memory_id}
        lineage = [
            (memory.memory_id, parent)
            for memory in memories.values()
            for parent in (
                memory.metadata.get('promoted_from'),
                memory.metadata.get('consolidated_from'),
            )
            if parent
        ]
        while True:
            expanded = family_ids | {
                item
                for child, parent in lineage
                if child in family_ids or parent in family_ids
                for item in (child, parent)
            }
            if expanded == family_ids:
                break
            family_ids = expanded
        family = [memory for memory in memories.values() if memory.memory_id in family_ids]
        if self.store is not None:
            await self.store.remove_memories(family)
        self.buffer.remove_many(family_ids)
        for memory in family:
            memory.lifecycle_status = 'deleted' if operation == 'delete' else 'revised'
            self.semantic_states.pop(memory.semantic_state_id, None)
        if operation == 'delete':
            return None
        if operation == 'supersede':
            replacement.version = target.version + 1
            replacement.supersedes = list(dict.fromkeys([*target.supersedes, target.memory_id]))
            return await self.add_memory(replacement)
        updated = {}
        for memory in family:
            revised = replacement.model_copy(deep=True)
            revised.memory_id = memory.memory_id
            revised.layer = memory.layer
            revised.ttl = memory.ttl
            revised.lifecycle_status = (
                'active_short_term' if memory.layer == 'M1' else 'consolidated'
            )
            revised.version = memory.version + 1
            revised.supersedes = list(memory.supersedes)
            revised.graph_binding = GraphBinding()
            revised.metadata = {
                **{
                    key: value
                    for key, value in replacement.metadata.items()
                    if key not in {'graphiti_episode_id', 'graphiti_episode_ids'}
                },
                **{
                    key: memory.metadata[key]
                    for key in ('promoted_from', 'consolidated_from')
                    if key in memory.metadata
                },
            }
            updated[memory.memory_id] = await self.add_memory(revised)
        return updated[memory_id]

    async def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        *,
        source_mode: Literal['full_source', 'evidence_bundle'] = 'full_source',
    ) -> HierarchicalMemoryContext:
        """Retrieve and relink M0/M1/M2 memories for a query-specific evidence graph."""

        route = self.router.route(query)
        instant_memories: list[UnifiedMemoryUnit] = []
        graph_edges: list[Any] = []

        if 'M0' in route.layers:
            instant_memories = self.ranker.rank_memories(
                self._matching_m0(query, user_id, session_id), query
            )[: self.policy.max_retrieval_results]

        if self.store is not None and any(layer in route.layers for layer in ['M1', 'M2']):
            search_results = await self.store.search_memories(
                query,
                user_id,
                num_results=self.policy.max_retrieval_results,
            )
            graph_edges = self.ranker.rank_graph_edges(list(search_results or []), query)[
                : self.policy.max_retrieval_results
            ]

        source_episodes = []
        if graph_edges and self.store is not None:
            source_episodes = await self.store.get_source_episodes(graph_edges, user_id)
            self._restore_source_memories(source_episodes, graph_edges, user_id, route.layers)

        selected_evidence_ids = {
            evidence_id for memory in instant_memories for evidence_id in memory.evidence_ids
        }
        selected_edge_ids = {edge.uuid for edge in graph_edges}
        selected_episode_ids = {
            episode_id for edge in graph_edges for episode_id in getattr(edge, 'episodes', [])
        }
        query_ids = set(re.findall(r'[\w-]+', query))
        candidates = []
        linked = []
        for memory in [*self.short_term_memories.values(), *self.long_term_memories.values()]:
            if not self._is_readable_memory(memory, user_id, route.layers):
                continue
            state = self.semantic_states.get(memory.semantic_state_id)
            if state is None or state.edge_or_subgraph_id != memory.memory_id:
                continue
            candidates.append((memory, state))
            if (
                selected_evidence_ids.intersection(memory.evidence_ids)
                or selected_edge_ids.intersection(memory.graph_binding.edges)
                or selected_episode_ids.intersection(GraphitiMemoryStore.episode_ids(memory))
                or query_ids.intersection(memory.evidence_ids)
            ):
                linked.append((memory, state))
        if not linked:
            linked = sorted(
                [
                    (memory, state)
                    for memory, state in candidates
                    if self.ranker.lexical_relevance(memory.raw_content or '', query) > 0
                ],
                key=lambda item: (
                    self.ranker.lexical_relevance(item[0].raw_content or '', query),
                    self.ranker.score_memory(item[0], query),
                ),
                reverse=True,
            )
        selected = linked[: self.policy.max_retrieval_results]

        context = self.relinker.build_context(
            query=query,
            selected_layers=route.layers,
            instant_memories=instant_memories,
            graph_edges=graph_edges,
            semantic_states=[state for _, state in selected],
        )
        context.source_episodes = source_episodes
        context.source_mode = source_mode
        context.context_text = context.render_text()
        returned_state_ids = {state.state_id for state in context.semantic_states}
        accessed = {
            memory.memory_id: memory
            for memory in [
                *context.instant_memories,
                *(memory for memory, state in selected if state.state_id in returned_state_ids),
            ]
        }
        for memory in accessed.values():
            memory.touch()
        return context

    def _is_readable_memory(
        self, memory: UnifiedMemoryUnit, user_id: str, layers: list[str]
    ) -> bool:
        return (
            memory.user_id == user_id
            and memory.layer in layers
            and memory.lifecycle_status not in {'expired', 'archived', 'deleted', 'revised'}
            and not memory.is_expired()
        )

    def _restore_source_memories(
        self, episodes: list[Any], graph_edges: list[Any], user_id: str, layers: list[str]
    ) -> None:
        """Recover stored snapshots only from the sources of this query's selected edges."""
        group_id = self.store.group_id(user_id)
        snapshots = {}
        for episode in episodes:
            supporting_edges = [
                edge
                for edge in graph_edges
                if episode.uuid in getattr(edge, 'episodes', [])
                and getattr(edge, 'group_id', group_id) == group_id
            ]
            if episode.group_id != group_id or not supporting_edges:
                continue
            try:
                envelope = json.loads(episode.content)
                if not isinstance(envelope, dict):
                    continue
                data = envelope.get('h_recongraph')
                if not isinstance(data, dict) or not all(
                    key in data for key in ('memory_id', 'layer', 'semantic_state')
                ):
                    continue
                state_data = data['semantic_state']
                if not isinstance(state_data, dict) or not all(
                    key in state_data for key in ('state_id', 'edge_or_subgraph_id')
                ):
                    continue
                state = MemorySemanticState.model_validate(state_data)
                memory = UnifiedMemoryUnit.model_validate(
                    {
                        **data,
                        'user_id': data.get('user_id', user_id),
                        'raw_content': envelope.get('content'),
                        'created_at': data.get('created_at', episode.valid_at),
                        'ttl': data.get(
                            'ttl', self.policy.m1_ttl if data['layer'] == 'M1' else None
                        ),
                        'semantic_state_id': state.state_id,
                        'metadata': {
                            'graphiti_episode_ids': data.get('graphiti_episode_ids', []),
                            'graphiti_episode_id': data.get('graphiti_episode_id'),
                            **{
                                key: data[key]
                                for key in ('promoted_from', 'consolidated_from')
                                if key in data
                            },
                        },
                    }
                )
                memory.created_at = ensure_utc(memory.created_at)
                if memory.user_id != user_id or memory.layer not in layers:
                    continue
                if (
                    memory.layer not in {'M1', 'M2'}
                    or state.edge_or_subgraph_id != memory.memory_id
                ):
                    continue
                memory.metadata['graphiti_episode_ids'] = list(
                    dict.fromkeys(
                        [
                            *GraphitiMemoryStore.episode_ids(memory),
                            episode.uuid,
                        ]
                    )
                )
                memory.metadata['graphiti_episode_id'] = episode.uuid
            except (ValueError, TypeError):
                continue
            memory.graph_binding.nodes = list(
                dict.fromkeys(
                    node_id
                    for edge in supporting_edges
                    for node_id in (
                        getattr(edge, 'source_node_uuid', None),
                        getattr(edge, 'target_node_uuid', None),
                    )
                    if node_id
                )
            )
            memory.graph_binding.edges = [edge.uuid for edge in supporting_edges]
            previous = snapshots.get(memory.memory_id)
            key = (memory.version, ensure_utc(episode.created_at))
            if previous is not None:
                older, _, old_key = previous
                if old_key > key:
                    memory, state, older, key = older, previous[1], memory, old_key
                memory.graph_binding.nodes = list(
                    dict.fromkeys(
                        [
                            *memory.graph_binding.nodes,
                            *older.graph_binding.nodes,
                        ]
                    )
                )
                memory.graph_binding.edges = list(
                    dict.fromkeys(
                        [
                            *memory.graph_binding.edges,
                            *older.graph_binding.edges,
                        ]
                    )
                )
                memory.metadata['graphiti_episode_ids'] = list(
                    dict.fromkeys(
                        [
                            *GraphitiMemoryStore.episode_ids(older),
                            *GraphitiMemoryStore.episode_ids(memory),
                        ]
                    )
                )
            snapshots[memory.memory_id] = (memory, state, key)
        for memory, state, _ in snapshots.values():
            if not self._is_readable_memory(memory, user_id, layers):
                continue
            if (
                memory.memory_id in self.short_term_memories
                or memory.memory_id in self.long_term_memories
            ):
                continue
            existing_state = self.semantic_states.get(state.state_id)
            if (
                existing_state is not None
                and existing_state.edge_or_subgraph_id != memory.memory_id
            ):
                continue
            target = self.short_term_memories if memory.layer == 'M1' else self.long_term_memories
            target[memory.memory_id] = memory
            self.semantic_states[state.state_id] = state

    async def consolidate(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> list[UnifiedMemoryUnit]:
        """Promote eligible M1 memories into M2."""

        consolidated: list[UnifiedMemoryUnit] = []
        for memory in list(self.short_term_memories.values()):
            if memory.lifecycle_status in {'deleted', 'revised', 'archived', 'expired'}:
                continue
            if memory.user_id != user_id:
                continue
            if session_id is not None and memory.session_id != session_id:
                continue
            if self._is_already_consolidated(memory.memory_id):
                continue
            if not self.evolution.should_consolidate_to_m2(memory):
                continue
            consolidated.append(await self._consolidate_memory(memory))
        return consolidated

    async def maintain(self, user_id: str | None = None) -> dict[str, list[UnifiedMemoryUnit]]:
        """Run TTL pruning and archive stale short-term memories."""

        expired_m0 = self.buffer.prune_expired()
        archived_m1: list[UnifiedMemoryUnit] = []
        for memory in self.short_term_memories.values():
            if user_id is not None and memory.user_id != user_id:
                continue
            if self.evolution.archive_if_stale(memory):
                archived_m1.append(memory)
        return {'expired_m0': expired_m0, 'archived_m1': archived_m1}

    async def _consolidate_memory(self, memory: UnifiedMemoryUnit) -> UnifiedMemoryUnit:
        long_term = self.evolution.consolidate_to_m2(memory)
        assessment = await self._assess_against_existing(
            long_term,
            list(self.long_term_memories.values()),
        )
        existing = self._get_long_term_memory(assessment.matched_memory_id)
        long_term = self.version_manager.apply(long_term, existing, assessment)
        return await self.add_memory(long_term)

    async def _assess_against_existing(
        self,
        memory: UnifiedMemoryUnit,
        existing_memories: list[UnifiedMemoryUnit],
    ) -> ConflictAssessment:
        assessment = self.conflict_detector.assess(memory, existing_memories)
        existing = self._get_memory(assessment.matched_memory_id)
        return await self.llm_evaluator.evaluate_relation(memory, existing, assessment)

    def _apply_assessment_metadata(
        self,
        memory: UnifiedMemoryUnit,
        assessment: ConflictAssessment,
    ) -> None:
        memory.scores.contradiction_risk = assessment.contradiction_risk
        memory.metadata = {
            **memory.metadata,
            'update_relation': assessment.relation_type,
            'update_confidence': assessment.confidence,
            'matched_memory_id': assessment.matched_memory_id,
            'update_rationale': assessment.rationale,
        }

    def _get_memory(self, memory_id: str | None) -> UnifiedMemoryUnit | None:
        if memory_id is None:
            return None
        return (
            self.short_term_memories.get(memory_id)
            or self.long_term_memories.get(memory_id)
            or self.buffer.get(memory_id)
        )

    def _get_long_term_memory(self, memory_id: str | None) -> UnifiedMemoryUnit | None:
        if memory_id is None:
            return None
        return self.long_term_memories.get(memory_id)

    def _is_already_consolidated(self, memory_id: str) -> bool:
        return any(
            memory.metadata.get('consolidated_from') == memory_id
            for memory in self.long_term_memories.values()
        )

    def _matching_m0(
        self,
        query: str,
        user_id: str,
        session_id: str | None,
    ) -> list[UnifiedMemoryUnit]:
        query_ids = set(re.findall(r'[\w-]+', query))
        return [
            memory
            for memory in self.buffer.list(user_id=user_id, session_id=session_id)
            if self.ranker.lexical_relevance(memory.raw_content or '', query) > 0
            or query_ids.intersection(memory.evidence_ids)
        ]
