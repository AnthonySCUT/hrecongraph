import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph.memory.graphiti_store import GraphitiMemoryStore
from h_recongraph.schemas.memory_unit import GraphBinding, UnifiedMemoryUnit


class RecordingGraphiti:
    def __init__(self, result):
        self.result = result
        self.added = []
        self.searches = []

    async def add_episode(self, **kwargs):
        self.added.append(kwargs)
        return self.result

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return []


@pytest.mark.asyncio
async def test_add_memory_records_returned_graph_binding_and_episode_id():
    result = SimpleNamespace(
        episode=SimpleNamespace(uuid='episode-a'),
        nodes=[SimpleNamespace(uuid='node-a'), SimpleNamespace(uuid='node-b')],
        edges=[SimpleNamespace(uuid='edge-a', episodes=['episode-a'])],
    )
    client = RecordingGraphiti(result)
    store = GraphitiMemoryStore(client)
    memory = UnifiedMemoryUnit(
        user_id='u1', layer='M1', raw_content='Project A uses Redis.', evidence_ids=['ev-a']
    )

    returned = await store.add_memory(memory)

    assert returned is result
    assert memory.graph_binding.nodes == ['node-a', 'node-b']
    assert memory.graph_binding.edges == ['edge-a']
    assert memory.metadata['graphiti_episode_id'] == 'episode-a'
    assert memory.raw_content == 'Project A uses Redis.'
    assert memory.evidence_ids == ['ev-a']
    assert len(client.added) == 1
    assert client.added[0]['group_id'] == 'user:u1'
    assert json.loads(client.added[0]['episode_body'])['content'] == memory.raw_content


@pytest.mark.asyncio
async def test_add_memory_merges_existing_bindings_without_duplicates():
    result = SimpleNamespace(
        episode=SimpleNamespace(uuid='episode-a'),
        nodes=[SimpleNamespace(uuid='node-a'), SimpleNamespace(uuid='node-b')],
        edges=[
            SimpleNamespace(uuid='edge-a', episodes=['episode-a']),
            SimpleNamespace(uuid='edge-b', episodes=['episode-a']),
        ],
    )
    memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M2',
        graph_binding=GraphBinding(nodes=['node-a'], edges=['edge-a']),
        metadata={'consolidated_from': 'stm-a'},
    )

    await GraphitiMemoryStore(RecordingGraphiti(result)).add_memory(memory)

    assert memory.graph_binding.nodes == ['node-a', 'node-b']
    assert memory.graph_binding.edges == ['edge-a', 'edge-b']
    assert memory.metadata['consolidated_from'] == 'stm-a'


@pytest.mark.asyncio
async def test_invalidated_old_edge_is_not_bound_as_support_for_new_memory():
    old_edge = SimpleNamespace(uuid='edge-old', episodes=['episode-old'])
    result = SimpleNamespace(
        episode=SimpleNamespace(uuid='episode-new'),
        nodes=[],
        edges=[SimpleNamespace(uuid='edge-new', episodes=['episode-new']), old_edge],
    )
    memory = UnifiedMemoryUnit(user_id='u1', layer='M1')

    await GraphitiMemoryStore(RecordingGraphiti(result)).add_memory(memory)

    assert memory.graph_binding.edges == ['edge-new']
    assert old_edge.episodes == ['episode-old']


@pytest.mark.asyncio
async def test_add_memory_preserves_legacy_client_return_value():
    result = {'accepted': True}
    client = RecordingGraphiti(result)
    memory = UnifiedMemoryUnit(user_id='u1', layer='M1')

    returned = await GraphitiMemoryStore(client).add_memory(memory)

    assert returned is result
    assert memory.graph_binding == GraphBinding()
    assert 'graphiti_episode_id' not in memory.metadata


@pytest.mark.asyncio
async def test_search_keeps_original_graphiti_query_contract():
    client = RecordingGraphiti(None)

    await GraphitiMemoryStore(client).search_memories('Project A', 'u1', num_results=3)

    assert client.searches == [{'query': 'Project A', 'group_ids': ['user:u1'], 'num_results': 3}]


def make_source_episode(episode_id, *, group_id='user:u1', content='Original source text.'):
    return EpisodicNode(
        uuid=episode_id,
        name=episode_id,
        group_id=group_id,
        source=EpisodeType.text,
        source_description='Source retrieval test',
        content=content,
        valid_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_source_episodes_are_loaded_once_by_exact_ids_in_first_reference_order(monkeypatch):
    first = make_source_episode('episode-a')
    second = make_source_episode('episode-b')
    load = AsyncMock(return_value=[second, first])
    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', load)
    client = SimpleNamespace(driver=object())
    edges = [
        SimpleNamespace(episodes=['episode-a', 'episode-b']),
        SimpleNamespace(episodes=['episode-a']),
    ]

    episodes = await GraphitiMemoryStore(client).get_source_episodes(edges, 'u1')

    load.assert_awaited_once_with(client.driver, ['episode-a', 'episode-b'])
    assert episodes == [first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize('edges', [[], [SimpleNamespace(episodes=[])]])
async def test_no_source_references_does_not_read_database(monkeypatch, edges):
    load = AsyncMock()
    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', load)

    episodes = await GraphitiMemoryStore(RecordingGraphiti(None)).get_source_episodes(edges, 'u1')

    assert episodes == []
    load.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_episode_does_not_trigger_recent_history_or_extra_results(monkeypatch):
    found = make_source_episode('episode-a')
    unrelated = make_source_episode('episode-unrelated')
    load = AsyncMock(return_value=[unrelated, found])
    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', load)
    client = SimpleNamespace(driver=object(), retrieve_episodes=AsyncMock())

    episodes = await GraphitiMemoryStore(client).get_source_episodes(
        [SimpleNamespace(episodes=['episode-missing', 'episode-a'])], 'u1'
    )

    assert episodes == [found]
    load.assert_awaited_once_with(client.driver, ['episode-missing', 'episode-a'])
    client.retrieve_episodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_results_follow_the_existing_user_group(monkeypatch):
    mine = make_source_episode('episode-a')
    other = make_source_episode('episode-b', group_id='user:u2')
    load = AsyncMock(return_value=[mine, other])
    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', load)

    episodes = await GraphitiMemoryStore(SimpleNamespace(driver=object())).get_source_episodes(
        [SimpleNamespace(episodes=['episode-a', 'episode-b'])], 'u1'
    )

    assert episodes == [mine]


@pytest.mark.asyncio
async def test_source_content_is_returned_verbatim_without_unwrapping_or_generation(monkeypatch):
    content = json.dumps(
        {
            'h_recongraph': {'memory_id': 'stm-a', 'layer': 'M1'},
            'content': 'During the pilot, Project A uses PostgreSQL because of compatibility.',
        }
    )
    source = make_source_episode('episode-a', content=content)
    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', AsyncMock(return_value=[source]))

    episodes = await GraphitiMemoryStore(SimpleNamespace(driver=object())).get_source_episodes(
        [SimpleNamespace(episodes=['episode-a'])], 'u1'
    )

    assert episodes[0] is source
    assert episodes[0].content == content


@pytest.mark.asyncio
async def test_source_read_error_is_propagated_without_automatic_fallback(monkeypatch):
    load = AsyncMock(side_effect=RuntimeError('database unavailable'))
    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', load)

    with pytest.raises(RuntimeError, match='database unavailable'):
        await GraphitiMemoryStore(SimpleNamespace(driver=object())).get_source_episodes(
            [SimpleNamespace(episodes=['episode-a'])], 'u1'
        )

    assert load.await_count == 1


@pytest.mark.asyncio
async def test_repeated_writes_keep_ordered_episode_history_and_latest_legacy_id():
    client = RecordingGraphiti(None)
    store = GraphitiMemoryStore(client)
    memory = UnifiedMemoryUnit(user_id='u1', layer='M1', raw_content='Project A uses Redis.')

    for episode_id in ['episode-a', 'episode-b', 'episode-b', 'episode-c']:
        client.result = SimpleNamespace(
            episode=SimpleNamespace(uuid=episode_id), nodes=[], edges=[]
        )
        await store.add_memory(memory)

    assert memory.metadata['graphiti_episode_ids'] == ['episode-a', 'episode-b', 'episode-c']
    assert memory.metadata['graphiti_episode_id'] == 'episode-c'
    second_body = json.loads(client.added[1]['episode_body'])['h_recongraph']
    assert second_body['graphiti_episode_ids'] == ['episode-a']
    assert second_body['graphiti_episode_id'] == 'episode-a'
    fourth_body = json.loads(client.added[3]['episode_body'])['h_recongraph']
    assert fourth_body['graphiti_episode_ids'] == ['episode-a', 'episode-b']
    assert len(client.added) == 4


@pytest.mark.asyncio
async def test_legacy_single_episode_id_is_merged_without_losing_other_metadata():
    client = RecordingGraphiti(
        SimpleNamespace(episode=SimpleNamespace(uuid='episode-new'), nodes=[], edges=[])
    )
    memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M2',
        metadata={
            'graphiti_episode_ids': ['episode-old', 'episode-old'],
            'graphiti_episode_id': 'episode-legacy',
            'consolidated_from': 'stm-a',
        },
    )

    await GraphitiMemoryStore(client).add_memory(memory)

    assert memory.metadata['graphiti_episode_ids'] == [
        'episode-old',
        'episode-legacy',
        'episode-new',
    ]
    assert memory.metadata['graphiti_episode_id'] == 'episode-new'
    assert memory.metadata['consolidated_from'] == 'stm-a'
    body = json.loads(client.added[0]['episode_body'])['h_recongraph']
    assert body['graphiti_episode_ids'] == ['episode-old', 'episode-legacy']


def test_envelope_preserves_recovery_fields_without_copying_arbitrary_metadata():
    from datetime import timedelta

    memory = UnifiedMemoryUnit(
        memory_id='stm-a',
        user_id='u1',
        layer='M1',
        raw_content='Original source.',
        ttl=timedelta(days=2),
        version=3,
        supersedes=['stm-old'],
        metadata={'graphiti_episode_id': 'episode-a', 'private_note': 'not an envelope field'},
    )
    store = GraphitiMemoryStore(RecordingGraphiti(None))

    body = json.loads(store._episode_body(memory))
    snapshot = body['h_recongraph']

    assert body['content'] == memory.raw_content
    assert snapshot['user_id'] == 'u1'
    assert snapshot['created_at'] == memory.created_at.isoformat().replace('+00:00', 'Z')
    assert snapshot['ttl'] == 'P2D'
    assert snapshot['version'] == 3
    assert snapshot['supersedes'] == ['stm-old']
    assert snapshot['graphiti_episode_ids'] == ['episode-a']
    assert 'private_note' not in snapshot
    assert 'private_note' not in snapshot.get('metadata', {})


@pytest.mark.asyncio
async def test_failed_write_does_not_append_an_episode_or_change_legacy_id():
    client = RecordingGraphiti(None)
    client.add_episode = AsyncMock(side_effect=RuntimeError('write failed'))
    memory = UnifiedMemoryUnit(
        user_id='u1', layer='M1', metadata={'graphiti_episode_id': 'episode-old'}
    )
    before = memory.model_dump()

    with pytest.raises(RuntimeError, match='write failed'):
        await GraphitiMemoryStore(client).add_memory(memory)

    assert memory.model_dump() == before
    client.add_episode.assert_awaited_once()
