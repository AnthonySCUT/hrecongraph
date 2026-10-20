"""Adapter between H-ReConGraph and Graphiti."""

import json
from typing import Any

from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit


class GraphitiMemoryStore:
    """Store M1/M2 memory units in Graphiti using a stable compatibility layer."""

    def __init__(self, graphiti_client: Any):
        self.graphiti = graphiti_client

    async def add_memory(self, memory: UnifiedMemoryUnit) -> Any:
        source_description = self._source_description(memory)
        episode_body = self._episode_body(memory)
        result = await self.graphiti.add_episode(
            name=f'{memory.layer.lower()}_{memory.memory_id}',
            episode_body=episode_body,
            source_description=source_description,
            reference_time=memory.created_at or utc_now(),
            source=EpisodeType.text,
            group_id=self.group_id(memory.user_id),
        )
        episode = getattr(result, 'episode', None)
        memory.graph_binding.nodes = list(
            dict.fromkeys(
                [*memory.graph_binding.nodes, *(node.uuid for node in getattr(result, 'nodes', []))]
            )
        )
        # add_episode also returns old edges invalidated by this episode.
        memory.graph_binding.edges = list(
            dict.fromkeys(
                [
                    *memory.graph_binding.edges,
                    *(
                        edge.uuid
                        for edge in getattr(result, 'edges', [])
                        if episode is not None and episode.uuid in edge.episodes
                    ),
                ]
            )
        )
        if episode is not None:
            memory.metadata['graphiti_episode_ids'] = list(
                dict.fromkeys([*self.episode_ids(memory), episode.uuid])
            )
            memory.metadata['graphiti_episode_id'] = episode.uuid
        return result

    async def search_memories(
        self,
        query: str,
        user_id: str,
        *,
        num_results: int = 10,
        group_ids: list[str] | None = None,
    ) -> Any:
        return await self.graphiti.search(
            query=query,
            group_ids=group_ids or [self.group_id(user_id)],
            num_results=num_results,
        )

    async def get_source_episodes(self, graph_edges: list[Any], user_id: str) -> list[EpisodicNode]:
        """Load the sources referenced by selected edges, in first-reference order."""
        episode_ids = list(
            dict.fromkeys(
                episode_id for edge in graph_edges for episode_id in getattr(edge, 'episodes', [])
            )
        )
        if not episode_ids:
            return []

        episodes = await EpisodicNode.get_by_uuids(self.graphiti.driver, episode_ids)
        episodes_by_id = {
            episode.uuid: episode
            for episode in episodes
            if episode.group_id == self.group_id(user_id)
        }
        return [
            episodes_by_id[episode_id] for episode_id in episode_ids if episode_id in episodes_by_id
        ]

    def group_id(self, user_id: str) -> str:
        return f'user:{user_id}'

    @staticmethod
    def episode_ids(memory: UnifiedMemoryUnit) -> list[str]:
        history = memory.metadata.get('graphiti_episode_ids', [])
        if not isinstance(history, list):
            history = []
        latest = memory.metadata.get('graphiti_episode_id')
        return list(
            dict.fromkeys(
                episode_id
                for episode_id in [*history, latest]
                if isinstance(episode_id, str) and episode_id
            )
        )

    def _source_description(self, memory: UnifiedMemoryUnit) -> str:
        return (
            'H-ReConGraph memory '
            f'layer={memory.layer} '
            f'session_id={memory.session_id or ""} '
            f'lifecycle_status={memory.lifecycle_status} '
            f'memory_id={memory.memory_id}'
        )

    def _episode_body(self, memory: UnifiedMemoryUnit) -> str:
        snapshot = memory.model_dump(
            mode='json',
            include={
                'memory_id',
                'user_id',
                'layer',
                'session_id',
                'lifecycle_status',
                'evidence_ids',
                'semantic_state',
                'semantic_state_id',
                'scores',
                'created_at',
                'ttl',
                'version',
                'supersedes',
                'source_spans',
                'content_type',
            },
        )
        snapshot['graphiti_episode_ids'] = self.episode_ids(memory)
        snapshot['graphiti_episode_id'] = memory.metadata.get('graphiti_episode_id')
        envelope = {
            'h_recongraph': snapshot,
            'content': memory.raw_content,
        }
        return json.dumps(envelope, ensure_ascii=False, default=str)
