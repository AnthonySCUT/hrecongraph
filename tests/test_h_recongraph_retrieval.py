import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph import HReConGraphMemory, MemoryPolicy
from h_recongraph.memory.graphiti_store import GraphitiMemoryStore
from h_recongraph.memory.ranker import MemoryUtilityRanker
from h_recongraph.schemas.memory_unit import GraphBinding, UnifiedMemoryUnit


class SearchGraphiti:
    def __init__(self, edges, episodes=()):
        self.edges = edges
        self.episodes = list(episodes)
        self.searches = []
        self.episode_reads = []
        self.driver = SimpleNamespace(
            graph_operations_interface=SimpleNamespace(
                episodic_node_get_by_uuids=self.get_episodes_by_uuids
            )
        )

    async def get_episodes_by_uuids(self, node_type, driver, uuids):
        self.episode_reads.append(list(uuids))
        return [episode for episode in self.episodes if episode.uuid in uuids]

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return self.edges


def add_local_memory(
    controller,
    memory_id,
    *,
    user_id='u1',
    session_id='old-session',
    layer='M1',
    edge_ids=(),
    episode_id=None,
    evidence_ids=None,
):
    memory = UnifiedMemoryUnit(
        memory_id=memory_id,
        user_id=user_id,
        session_id=session_id,
        layer=layer,
        raw_content=f'{memory_id} because compatibility matters',
        evidence_ids=evidence_ids or [f'ev-{memory_id}'],
        graph_binding=GraphBinding(edges=list(edge_ids)),
        metadata={'graphiti_episode_id': episode_id} if episode_id else {},
    )
    state = controller.semantic_state_constructor.construct(memory)
    memory.semantic_state_id = state.state_id
    target = controller.long_term_memories if layer == 'M2' else controller.short_term_memories
    target[memory_id] = memory
    controller.semantic_states[state.state_id] = state
    return memory


def retrieved_memory_ids(context):
    return {state.edge_or_subgraph_id for state in context.semantic_states}


@pytest.mark.asyncio
async def test_m0_chinese_retrieval_selects_related_text_within_limit():
    controller = HReConGraphMemory(policy=MemoryPolicy(max_retrieval_results=1))
    related = controller.buffer.add('项目接口因为联调而延期', 'u1')
    omitted = controller.buffer.add('项目预算等待审批', 'u1')
    unrelated = controller.buffer.add('午餐吃面条', 'u1')

    context = await controller.retrieve('项目接口延期原因', user_id='u1')

    assert context.instant_memories == [related]
    assert related.access_count == 1
    assert omitted.access_count == unrelated.access_count == 0
    assert '午餐' not in context.context_text


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['volcano eruption', 'please tell me'])
async def test_m0_without_content_overlap_returns_empty_context(query):
    controller = HReConGraphMemory()
    memory = controller.buffer.add('please review project budget', 'u1')

    context = await controller.retrieve(query, user_id='u1')

    assert context.instant_memories == []
    assert context.context_text == ''
    assert memory.access_count == 0


@pytest.mark.asyncio
async def test_m0_can_be_selected_by_its_evidence_id():
    controller = HReConGraphMemory()
    memory = controller.buffer.add('项目接口因为联调而延期', 'u1')
    controller.buffer.add('午餐吃面条', 'u1')

    context = await controller.retrieve(f'source {memory.evidence_ids[0]}', user_id='u1')

    assert context.instant_memories == [memory]
    assert memory.access_count == 1


def test_repeated_ranking_preserves_memory_scores_and_order():
    ranker = MemoryUtilityRanker()
    memories = [
        UnifiedMemoryUnit(user_id='u1', layer='M0', raw_content='project deadline'),
        UnifiedMemoryUnit(user_id='u1', layer='M0', raw_content='coffee break'),
    ]
    memories[0].scores.retrieval_utility = 0.25
    original_scores = [memory.scores.model_dump() for memory in memories]
    first = ranker.rank_memories(memories, 'project deadline')
    first_scores = [ranker.score_memory(memory, 'project deadline') for memory in first]

    for _ in range(3):
        assert ranker.rank_memories(memories, 'project deadline') == first
        assert [ranker.score_memory(memory, 'project deadline') for memory in first] == first_scores

    assert [memory.scores.model_dump() for memory in memories] == original_scores


def test_memory_ranking_uses_chinese_content_overlap():
    ranker = MemoryUtilityRanker()
    partial = UnifiedMemoryUnit(user_id='u1', layer='M0', raw_content='项目预算')
    relevant = UnifiedMemoryUnit(user_id='u1', layer='M0', raw_content='项目接口延期')

    assert ranker.rank_memories([partial, relevant], '项目接口延期原因') == [relevant, partial]


@pytest.mark.asyncio
async def test_only_states_bound_to_retrieved_edges_are_selected():
    client = SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='Project A')])
    controller = HReConGraphMemory(client)
    add_local_memory(controller, 'unrelated', edge_ids=['edge-b'])
    add_local_memory(controller, 'related', edge_ids=['edge-a'])

    context = await controller.retrieve('Project A', user_id='u1', session_id='new-session')

    assert retrieved_memory_ids(context) == {'related'}
    assert len(context.graph_edges) == 1
    assert client.searches == [{'query': 'Project A', 'group_ids': ['user:u1'], 'num_results': 10}]


@pytest.mark.asyncio
async def test_one_retrieved_edge_can_select_multiple_supporting_memories():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='shared-edge', episodes=[], fact='Project A')])
    )
    add_local_memory(controller, 'first', edge_ids=['shared-edge'])
    add_local_memory(controller, 'second', edge_ids=['shared-edge'])

    context = await controller.retrieve('Project A', user_id='u1')

    assert retrieved_memory_ids(context) == {'first', 'second'}


@pytest.mark.asyncio
async def test_returned_episode_provenance_also_links_local_states():
    controller = HReConGraphMemory(
        SearchGraphiti(
            [SimpleNamespace(uuid='resolved-edge', episodes=['episode-a'], fact='Project A')]
        )
    )
    add_local_memory(controller, 'related', edge_ids=['old-edge'], episode_id='episode-a')
    add_local_memory(controller, 'unrelated', episode_id='episode-b')

    context = await controller.retrieve('Project A', user_id='u1')

    assert retrieved_memory_ids(context) == {'related'}


@pytest.mark.asyncio
async def test_matching_ids_do_not_select_another_users_local_states():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=['episode-a'], fact='Project A')])
    )
    add_local_memory(controller, 'mine', edge_ids=['edge-a'])
    add_local_memory(
        controller, 'other-user', user_id='u2', edge_ids=['edge-a'], episode_id='episode-a'
    )

    context = await controller.retrieve('Project A ev-other-user', user_id='u1')

    assert retrieved_memory_ids(context) == {'mine'}


@pytest.mark.asyncio
async def test_long_term_memory_remains_available_across_sessions():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='user prefers tea')])
    )
    add_local_memory(controller, 'preference', layer='M2', edge_ids=['edge-a'])

    context = await controller.retrieve('What do I prefer?', user_id='u1', session_id='new-session')

    assert context.selected_layers == ['M2']
    assert retrieved_memory_ids(context) == {'preference'}


@pytest.mark.asyncio
async def test_states_follow_the_existing_layer_route():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='user prefers tea')])
    )
    add_local_memory(controller, 'short', layer='M1', edge_ids=['edge-a'])
    add_local_memory(controller, 'long', layer='M2', edge_ids=['edge-a'])

    context = await controller.retrieve('What do I prefer?', user_id='u1')

    assert retrieved_memory_ids(context) == {'long'}


@pytest.mark.asyncio
async def test_no_graph_hits_does_not_return_every_local_state():
    controller = HReConGraphMemory(SearchGraphiti([]))
    add_local_memory(controller, 'unrelated', layer='M2', edge_ids=['edge-b'])

    context = await controller.retrieve('What do I prefer?', user_id='u1')

    assert context.semantic_states == []


@pytest.mark.asyncio
async def test_selected_m0_evidence_keeps_its_promoted_semantic_state_without_graphiti():
    controller = HReConGraphMemory()
    selected = await controller.add_interaction(
        'Please remember this: current project deadline because compatibility matters',
        user_id='u1',
        session_id='s1',
    )
    add_local_memory(controller, 'unrelated', evidence_ids=['other-evidence'])

    context = await controller.retrieve('why current project', user_id='u1', session_id='s1')

    assert context.instant_memories
    assert retrieved_memory_ids(context) == {selected.memory_id}


@pytest.mark.asyncio
async def test_states_are_linked_only_after_graph_results_are_limited():
    controller = HReConGraphMemory(
        SearchGraphiti(
            [
                SimpleNamespace(uuid='edge-a', episodes=[], fact='Project A'),
                SimpleNamespace(uuid='edge-b', episodes=[], fact='Project B'),
            ]
        ),
        policy=MemoryPolicy(max_retrieval_results=1),
    )
    add_local_memory(controller, 'omitted', edge_ids=['edge-b'])
    add_local_memory(controller, 'selected', edge_ids=['edge-a'])

    context = await controller.retrieve('Project A', user_id='u1')

    assert [edge.uuid for edge in context.graph_edges] == ['edge-a']
    assert retrieved_memory_ids(context) == {'selected'}


@pytest.mark.asyncio
async def test_unrelated_memories_do_not_fill_the_existing_state_limit():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='Project A')]),
        policy=MemoryPolicy(max_retrieval_results=1),
    )
    add_local_memory(controller, 'unrelated', edge_ids=['edge-b'])
    add_local_memory(controller, 'related', edge_ids=['edge-a'])

    context = await controller.retrieve('Project A', user_id='u1')

    assert retrieved_memory_ids(context) == {'related'}


@pytest.mark.asyncio
async def test_explicit_evidence_query_still_selects_its_own_state():
    controller = HReConGraphMemory(SearchGraphiti([]))
    add_local_memory(controller, 'requested', evidence_ids=['ev-requested'])
    add_local_memory(controller, 'unrelated', evidence_ids=['ev-unrelated'])

    context = await controller.retrieve('source ev-requested', user_id='u1')

    assert retrieved_memory_ids(context) == {'requested'}


@pytest.mark.asyncio
async def test_real_graphiti_result_models_connect_write_to_retrieve():
    now = utc_now()
    episode = EpisodicNode(
        uuid='episode-a',
        name='Project A',
        group_id='user:u1',
        source=EpisodeType.text,
        source_description='test source',
        content='Please remember this: Project A uses Redis because compatibility matters.',
        valid_at=now,
    )
    nodes = [
        EntityNode(uuid='node-a', name='Project A', group_id='user:u1'),
        EntityNode(uuid='node-b', name='Redis', group_id='user:u1'),
    ]
    edge = EntityEdge(
        uuid='edge-a',
        group_id='user:u1',
        source_node_uuid='node-a',
        target_node_uuid='node-b',
        created_at=now,
        name='USES',
        fact='Project A uses Redis.',
        episodes=['episode-a'],
    )
    result = AddEpisodeResults(
        episode=episode,
        episodic_edges=[],
        nodes=nodes,
        edges=[edge],
        communities=[],
        community_edges=[],
    )

    class WriteAndSearchGraphiti(SearchGraphiti):
        async def add_episode(self, **kwargs):
            return result

    client = WriteAndSearchGraphiti([edge])
    controller = HReConGraphMemory(client)
    written = await controller.add_interaction(episode.content, user_id='u1', session_id='s1')
    add_local_memory(controller, 'unrelated', edge_ids=['edge-b'])

    context = await controller.retrieve('why Project A uses Redis', user_id='u1', session_id='s2')

    assert written.graph_binding.nodes == ['node-a', 'node-b']
    assert written.graph_binding.edges == ['edge-a']
    assert written.metadata['graphiti_episode_id'] == 'episode-a'
    assert context.graph_edges == [edge]
    assert retrieved_memory_ids(context) == {written.memory_id}
    assert context.semantic_states[0].cause == 'compatibility matters.'
    assert len(client.searches) == 1


def source_episode(episode_id, content):
    return EpisodicNode(
        uuid=episode_id,
        name=episode_id,
        group_id='user:u1',
        source=EpisodeType.text,
        source_description='Source retrieval test',
        content=content,
        valid_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_sources_are_returned_without_any_local_memory_binding():
    source = source_episode(
        'episode-a',
        'During the pilot, Project A uses PostgreSQL because of compatibility.',
    )
    edge = EntityEdge(
        uuid='edge-a',
        group_id='user:u1',
        source_node_uuid='node-a',
        target_node_uuid='node-b',
        created_at=utc_now(),
        name='USES',
        fact='Project A uses PostgreSQL',
        episodes=['episode-a'],
    )
    client = SearchGraphiti([edge], [source])
    controller = HReConGraphMemory(client)

    context = await controller.retrieve('why Project A uses PostgreSQL', user_id='u1')

    assert controller.short_term_memories == {}
    assert context.semantic_states == []
    assert context.source_episodes == [source]
    assert context.source_episodes[0].content == source.content
    assert client.episode_reads == [['episode-a']]
    assert context.model_dump(mode='json')['source_episodes'][0]['content'] == source.content


@pytest.mark.asyncio
async def test_sources_are_loaded_only_for_final_returned_edges():
    first = source_episode('episode-a', 'Source for Project A.')
    omitted = source_episode('episode-b', 'Source for Project B.')
    client = SearchGraphiti(
        [
            SimpleNamespace(uuid='edge-a', episodes=['episode-a'], fact='Project A'),
            SimpleNamespace(uuid='edge-b', episodes=['episode-b'], fact='Project B'),
        ],
        [first, omitted],
    )
    controller = HReConGraphMemory(client, policy=MemoryPolicy(max_retrieval_results=1))

    context = await controller.retrieve('Project A', user_id='u1')

    assert [edge.uuid for edge in context.graph_edges] == ['edge-a']
    assert context.source_episodes == [first]
    assert client.episode_reads == [['episode-a']]


@pytest.mark.asyncio
async def test_shared_and_multiple_source_episodes_are_not_cut_by_edge_limit():
    first = source_episode('episode-a', 'First source.')
    second = source_episode('episode-b', 'Second source.')
    client = SearchGraphiti(
        [
            SimpleNamespace(
                uuid='edge-a', episodes=['episode-a', 'episode-b', 'episode-a'], fact='Project A'
            )
        ],
        [second, first],
    )
    controller = HReConGraphMemory(client, policy=MemoryPolicy(max_retrieval_results=1))

    context = await controller.retrieve('Project A', user_id='u1')

    assert len(context.graph_edges) == 1
    assert context.source_episodes == [first, second]
    assert client.episode_reads == [['episode-a', 'episode-b']]


@pytest.mark.asyncio
async def test_missing_source_keeps_the_retrieved_edge_without_inventing_content():
    edge = SimpleNamespace(uuid='edge-a', episodes=['episode-missing'], fact='Project A')
    client = SearchGraphiti([edge])

    context = await HReConGraphMemory(client).retrieve('Project A', user_id='u1')

    assert context.graph_edges == [edge]
    assert context.source_episodes == []
    assert client.episode_reads == [['episode-missing']]


@pytest.mark.asyncio
async def test_empty_stored_source_remains_empty():
    source = source_episode('episode-a', '')
    client = SearchGraphiti(
        [SimpleNamespace(uuid='edge-a', episodes=['episode-a'], fact='Project A')], [source]
    )

    context = await HReConGraphMemory(client).retrieve('Project A', user_id='u1')

    assert context.source_episodes == [source]
    assert context.source_episodes[0].content == ''


@pytest.mark.asyncio
async def test_source_free_retrieval_adds_no_database_read():
    client = SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='Project A')])

    context = await HReConGraphMemory(client).retrieve('Project A', user_id='u1')

    assert context.source_episodes == []
    assert client.episode_reads == []


@pytest.mark.asyncio
async def test_m0_only_without_store_keeps_empty_source_list():
    controller = HReConGraphMemory()
    await controller.add_interaction('okay', user_id='u1', session_id='s1')

    context = await controller.retrieve('okay', user_id='u1', session_id='s1')

    assert context.instant_memories
    assert context.source_episodes == []


@pytest.mark.asyncio
@pytest.mark.parametrize('graph_hit', [False, True])
async def test_lexical_fallback_is_ranked_bounded_and_excludes_unrelated_memories(graph_hit):
    edges = [SimpleNamespace(uuid='unbound', episodes=[], fact='Redis')] if graph_hit else []
    controller = HReConGraphMemory(
        SearchGraphiti(edges), policy=MemoryPolicy(max_retrieval_results=1)
    )
    unrelated = add_local_memory(controller, 'unrelated')
    unrelated.raw_content = 'The user likes tea.'
    partial = add_local_memory(controller, 'partial')
    partial.raw_content = 'Redis supports caching.'
    best = add_local_memory(controller, 'best')
    best.raw_content = 'Redis cluster has replicas.'

    context = await controller.retrieve('Redis cluster?', user_id='u1')

    assert retrieved_memory_ids(context) == {'best'}
    assert best.access_count == 1
    assert partial.access_count == unrelated.access_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'query', ['the and is?', 'please tell me about it', '的 了 是 吗', '的了是吗']
)
async def test_stopwords_alone_do_not_enable_local_fallback(query):
    controller = HReConGraphMemory(SearchGraphiti([]))
    memory = add_local_memory(controller, 'stopwords')
    memory.raw_content = 'the and is please tell me about it 的 了 是 吗'

    context = await controller.retrieve(query, user_id='u1')

    assert context.semantic_states == []
    assert memory.access_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['expired', 'archived', 'deleted', 'revised', 'ttl'])
async def test_inactive_local_memory_is_not_returned_or_touched_even_with_binding(status):
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='Redis')])
    )
    memory = add_local_memory(controller, 'inactive', edge_ids=['edge-a'])
    memory.raw_content = 'Redis cluster'
    if status == 'ttl':
        memory.created_at -= timedelta(days=2)
        memory.ttl = timedelta(days=1)
    else:
        memory.lifecycle_status = status

    context = await controller.retrieve('Redis cluster', user_id='u1')

    assert context.semantic_states == []
    assert memory.access_count == 0


@pytest.mark.asyncio
async def test_fallback_respects_user_and_route_without_touching_other_candidates():
    controller = HReConGraphMemory(SearchGraphiti([]))
    mine = add_local_memory(controller, 'mine', layer='M2')
    short = add_local_memory(controller, 'short', layer='M1')
    other = add_local_memory(controller, 'other', user_id='u2', layer='M2')
    for memory in [mine, short, other]:
        memory.raw_content = 'I prefer oolong tea.'

    context = await controller.retrieve('prefer oolong?', user_id='u1')

    assert context.selected_layers == ['M2']
    assert retrieved_memory_ids(context) == {'mine'}
    assert [mine.access_count, short.access_count, other.access_count] == [1, 0, 0]


@pytest.mark.asyncio
async def test_successful_binding_does_not_expand_to_lexical_only_memories():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='Redis')])
    )
    linked = add_local_memory(controller, 'linked', edge_ids=['edge-a'])
    lexical = add_local_memory(controller, 'lexical')
    lexical.raw_content = 'Redis cluster'

    context = await controller.retrieve('Redis cluster', user_id='u1')

    assert retrieved_memory_ids(context) == {'linked'}
    assert linked.access_count == 1
    assert lexical.access_count == 0


@pytest.mark.asyncio
async def test_returned_m0_and_associated_m1_m2_are_each_touched_once_without_writes():
    client = SearchGraphiti([])
    controller = HReConGraphMemory(client, policy=MemoryPolicy(consolidation_threshold=2))
    # Build the write locally; only the read path uses the recording graph client.
    controller.store = None
    short = await controller.add_interaction(
        'remember this current project Redis because compatibility matters',
        user_id='u1',
        session_id='s1',
    )
    long = add_local_memory(controller, 'ltm-copy', layer='M2', evidence_ids=short.evidence_ids)
    short.graph_binding.edges = ['edge-a']
    short.metadata['graphiti_episode_id'] = 'episode-a'
    client.edges = [SimpleNamespace(uuid='edge-a', episodes=['episode-a'], fact='Redis')]
    client.add_episode = AsyncMock()
    controller.store = GraphitiMemoryStore(client)
    instant = next(iter(controller.buffer._memories.values()))

    for expected in [1, 2]:
        context = await controller.retrieve(
            f'why current Redis {short.evidence_ids[0]}', user_id='u1', session_id='s1'
        )
        assert retrieved_memory_ids(context) == {short.memory_id, long.memory_id}
        assert [instant.access_count, short.access_count, long.access_count] == [expected] * 3
    client.add_episode.assert_not_awaited()
    assert controller.evolution.consolidation_score(
        short
    ) > controller.evolution.consolidation_score(short.model_copy(update={'access_count': 0}))


@pytest.mark.asyncio
async def test_bound_candidates_cut_by_limit_are_not_touched():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=[], fact='Redis')]),
        policy=MemoryPolicy(max_retrieval_results=1),
    )
    first = add_local_memory(controller, 'first', edge_ids=['edge-a'])
    second = add_local_memory(controller, 'second', edge_ids=['edge-a'])

    context = await controller.retrieve('Redis', user_id='u1')

    assert retrieved_memory_ids(context) == {'first'}
    assert [first.access_count, second.access_count] == [1, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize('requested', ['ev-a', 'ev-ab', 'im_a', 'im_ab'])
async def test_evidence_ids_require_complete_tokens(requested):
    controller = HReConGraphMemory(SearchGraphiti([]))
    memories = [
        add_local_memory(controller, f'm-{i}', evidence_ids=[i])
        for i in ['ev-a', 'ev-ab', 'im_a', 'im_ab']
    ]

    context = await controller.retrieve(f'source ({requested}).', user_id='u1')

    assert retrieved_memory_ids(context) == {f'm-{requested}'}
    assert [memory.access_count for memory in memories] == [
        int(memory.evidence_ids == [requested]) for memory in memories
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['source xev-a', 'source ev-a-extra', 'source ev-a_more'])
async def test_embedded_or_extended_evidence_id_does_not_match(query):
    controller = HReConGraphMemory(SearchGraphiti([]))
    memory = add_local_memory(controller, 'only-a', evidence_ids=['ev-a'])

    context = await controller.retrieve(query, user_id='u1')

    assert context.semantic_states == []
    assert memory.access_count == 0


@pytest.mark.asyncio
async def test_older_episode_in_history_still_links_local_memory():
    controller = HReConGraphMemory(
        SearchGraphiti([SimpleNamespace(uuid='edge-a', episodes=['episode-old'], fact='Redis')])
    )
    memory = add_local_memory(controller, 'linked')
    memory.metadata.update(
        graphiti_episode_id='episode-new', graphiti_episode_ids=['episode-old', 'episode-new']
    )

    context = await controller.retrieve('Redis', user_id='u1')

    assert retrieved_memory_ids(context) == {'linked'}
    assert memory.access_count == 1


def enveloped_source(
    *, memory_id='stm-recovered', layer='M1', user_id='u1', episode_id='episode-a'
):
    owner = HReConGraphMemory()
    memory = add_local_memory(owner, memory_id, layer=layer, user_id=user_id)
    memory.raw_content = 'Project A uses Redis because compatibility matters'
    memory.semantic_state = owner.semantic_states[memory.semantic_state_id].model_dump(mode='json')
    source = source_episode(episode_id, GraphitiMemoryStore(None)._episode_body(memory))
    source.group_id = f'user:{user_id}'
    return memory, source


def source_edge(episode_ids, *, group_id='user:u1', uuid='edge-a'):
    return EntityEdge(
        uuid=uuid,
        group_id=group_id,
        source_node_uuid='node-a',
        target_node_uuid='node-b',
        created_at=utc_now(),
        name='USES',
        fact='Project A uses Redis',
        episodes=episode_ids,
    )


@pytest.mark.asyncio
async def test_fresh_controller_recovers_state_binding_and_history_only_from_returned_sources():
    written, source = enveloped_source()
    body = json.loads(source.content)
    body['h_recongraph']['graphiti_episode_ids'] = ['episode-old']
    body['h_recongraph']['graphiti_episode_id'] = 'episode-old'
    source.content = json.dumps(body)
    edge = source_edge(['episode-a'])
    client = SearchGraphiti([edge], [source])
    controller = HReConGraphMemory(client)

    context = await controller.retrieve('why Project A uses Redis', user_id='u1')

    recovered = controller.short_term_memories[written.memory_id]
    assert retrieved_memory_ids(context) == {written.memory_id}
    assert context.semantic_states[0].model_dump() == written.semantic_state
    assert recovered.graph_binding.nodes == ['node-a', 'node-b']
    assert recovered.graph_binding.edges == ['edge-a']
    assert recovered.metadata['graphiti_episode_ids'] == ['episode-old', 'episode-a']
    assert recovered.metadata['graphiti_episode_id'] == 'episode-a'
    assert recovered.access_count == 1
    assert recovered.created_at == written.created_at
    assert client.episode_reads == [['episode-a']]
    assert context.source_episodes == [source]
    await controller.retrieve('why Project A uses Redis', user_id='u1')
    assert controller.short_term_memories[written.memory_id] is recovered
    assert recovered.access_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'content',
    [
        'Plain legacy source.',
        'not {json',
        '[]',
        '{}',
        '{"h_recongraph": null}',
        '{"h_recongraph": {"memory_id": "stm-old", "layer": "M1"}}',
    ],
)
async def test_legacy_or_incomplete_source_is_returned_without_fabricating_state(content):
    source = source_episode('episode-a', content)
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))

    context = await controller.retrieve('Project A', user_id='u1')

    assert context.source_episodes == [source]
    assert context.semantic_states == []
    assert controller.short_term_memories == {}
    assert controller.long_term_memories == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change', ['user', 'layer', 'state_owner', 'state_invalid', 'expired', 'ttl']
)
async def test_source_recovery_rejects_wrong_scope_or_invalid_state_without_losing_raw_source(
    change,
):
    _, source = enveloped_source()
    body = json.loads(source.content)
    data = body['h_recongraph']
    if change == 'user':
        data['user_id'] = 'u2'
    elif change == 'layer':
        data['layer'] = 'M2'
    elif change == 'state_owner':
        data['semantic_state']['edge_or_subgraph_id'] = 'someone-else'
    elif change == 'state_invalid':
        data['semantic_state']['state_type'] = 'evidence_supported'
        data['semantic_state']['source_spans'] = []
    elif change == 'expired':
        data['lifecycle_status'] = 'expired'
    else:
        data['created_at'] = (utc_now() - timedelta(days=2)).isoformat()
        data['ttl'] = 'P1D'
    source.content = json.dumps(body)
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))

    context = await controller.retrieve('current Redis', user_id='u1')

    assert context.source_episodes == [source]
    assert context.semantic_states == []
    assert controller.short_term_memories == {}
    assert controller.long_term_memories == {}


@pytest.mark.asyncio
async def test_recovery_does_not_read_unselected_sources_or_overwrite_cached_memory():
    original, selected = enveloped_source()
    _, omitted = enveloped_source(memory_id='stm-omitted', episode_id='episode-b')
    client = SearchGraphiti(
        [source_edge(['episode-a']), source_edge(['episode-b'], uuid='edge-b')], [selected, omitted]
    )
    controller = HReConGraphMemory(client, policy=MemoryPolicy(max_retrieval_results=1))
    cached = add_local_memory(controller, original.memory_id, edge_ids=['edge-a'])
    cached.raw_content = 'Newer in-process text'

    context = await controller.retrieve('Redis', user_id='u1')

    assert client.episode_reads == [['episode-a']]
    assert set(controller.short_term_memories) == {original.memory_id}
    assert controller.short_term_memories[original.memory_id] is cached
    assert cached.raw_content == 'Newer in-process text'
    assert cached.access_count == 1
    assert context.source_episodes == [selected]


@pytest.mark.asyncio
async def test_local_fallback_also_works_without_a_graph_store():
    controller = HReConGraphMemory()
    memory = add_local_memory(controller, 'redis')
    memory.raw_content = 'Redis replicas'

    context = await controller.retrieve('Redis?', user_id='u1')

    assert retrieved_memory_ids(context) == {'redis'}
    assert memory.access_count == 1


@pytest.mark.asyncio
async def test_m0_cut_by_result_limit_is_not_touched():
    controller = HReConGraphMemory(policy=MemoryPolicy(max_retrieval_results=1))
    first = controller.buffer.add('Redis', user_id='u1', session_id='s1')
    best = controller.buffer.add('Redis cluster', user_id='u1', session_id='s1')

    context = await controller.retrieve('current Redis cluster', user_id='u1', session_id='s1')

    assert context.instant_memories == [best]
    assert [first.access_count, best.access_count] == [0, 1]


@pytest.mark.asyncio
async def test_unrelated_query_without_overlap_returns_no_state_or_access():
    controller = HReConGraphMemory(SearchGraphiti([]))
    memory = add_local_memory(controller, 'only-redis')
    memory.raw_content = 'Redis cluster'

    context = await controller.retrieve('astronomy telescope', user_id='u1')

    assert context.semantic_states == []
    assert memory.access_count == 0


@pytest.mark.asyncio
async def test_fallback_uses_meaningful_chinese_overlap_without_matching_single_stopwords():
    controller = HReConGraphMemory(SearchGraphiti([]))
    related = add_local_memory(controller, 'related')
    related.raw_content = '用户喜欢喝咖啡'
    unrelated = add_local_memory(controller, 'unrelated')
    unrelated.raw_content = '用户安排旅游'

    context = await controller.retrieve('咖啡', user_id='u1')

    assert retrieved_memory_ids(context) == {'related'}
    assert [related.access_count, unrelated.access_count] == [1, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize('layer,query', [('M1', 'current Redis'), ('M2', 'prefer Redis')])
async def test_source_recovers_into_its_allowed_layer(layer, query):
    written, source = enveloped_source(layer=layer)
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))

    context = await controller.retrieve(query, user_id='u1')

    target = controller.short_term_memories if layer == 'M1' else controller.long_term_memories
    assert retrieved_memory_ids(context) == {written.memory_id}
    assert target[written.memory_id].layer == layer
    assert target[written.memory_id].access_count == 1


@pytest.mark.asyncio
async def test_complete_old_envelope_can_infer_owner_from_group_without_resetting_age():
    written, source = enveloped_source()
    body = json.loads(source.content)
    for field in [
        'user_id',
        'created_at',
        'ttl',
        'semantic_state_id',
        'graphiti_episode_ids',
        'graphiti_episode_id',
    ]:
        body['h_recongraph'].pop(field, None)
    source.valid_at = utc_now() - timedelta(days=1)
    source.content = json.dumps(body)
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))

    context = await controller.retrieve('current Redis', user_id='u1')

    restored = controller.short_term_memories[written.memory_id]
    assert retrieved_memory_ids(context) == {written.memory_id}
    assert restored.user_id == 'u1'
    assert restored.created_at == source.valid_at
    assert restored.ttl == controller.policy.m1_ttl


@pytest.mark.asyncio
async def test_old_envelope_without_age_fields_does_not_revive_expired_m1():
    _, source = enveloped_source()
    body = json.loads(source.content)
    body['h_recongraph'].pop('created_at')
    body['h_recongraph'].pop('ttl')
    source.valid_at = utc_now() - timedelta(days=30)
    source.content = json.dumps(body)
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))

    context = await controller.retrieve('current Redis', user_id='u1')

    assert context.semantic_states == []
    assert context.source_episodes == [source]
    assert controller.short_term_memories == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong_group', ['source', 'edge'])
async def test_recovery_requires_source_and_supporting_edge_user_group(wrong_group):
    _, source = enveloped_source()
    edge = source_edge(['episode-a'])
    if wrong_group == 'source':
        source.group_id = 'user:u2'
    else:
        edge.group_id = 'user:u2'
    store = GraphitiMemoryStore(SearchGraphiti([edge], [source]))
    # Exercise the controller's scope check even if an injected store returns a wrong-group source.
    store.get_source_episodes = AsyncMock(return_value=[source])
    controller = HReConGraphMemory(store=store)

    context = await controller.retrieve('current Redis', user_id='u1')

    assert context.semantic_states == []
    assert controller.short_term_memories == {}


@pytest.mark.asyncio
async def test_multiple_snapshots_merge_provenance_and_use_latest_content_once():
    written, older = enveloped_source(episode_id='episode-old')
    _, newer = enveloped_source(episode_id='episode-new')
    older.created_at = utc_now() - timedelta(hours=1)
    newer.created_at = utc_now()
    body = json.loads(newer.content)
    body['content'] = 'Newer stored content'
    newer.content = json.dumps(body)
    controller = HReConGraphMemory(
        SearchGraphiti(
            [
                source_edge(['episode-new'], uuid='edge-new'),
                source_edge(['episode-old'], uuid='edge-old'),
            ],
            [older, newer],
        )
    )

    context = await controller.retrieve('current Redis', user_id='u1')

    restored = controller.short_term_memories[written.memory_id]
    assert len(context.semantic_states) == 1
    assert restored.raw_content == 'Newer stored content'
    assert set(restored.graph_binding.edges) == {'edge-old', 'edge-new'}
    assert restored.graph_binding.nodes == ['node-a', 'node-b']
    assert restored.metadata['graphiti_episode_ids'] == ['episode-old', 'episode-new']
    assert restored.metadata['graphiti_episode_id'] == 'episode-new'
    assert restored.access_count == 1


@pytest.mark.asyncio
async def test_latest_inactive_snapshot_prevents_older_active_snapshot_resurrection():
    _, older = enveloped_source(episode_id='episode-old')
    _, newer = enveloped_source(episode_id='episode-new')
    older.created_at = utc_now() - timedelta(hours=1)
    newer.created_at = utc_now()
    body = json.loads(newer.content)
    body['h_recongraph']['lifecycle_status'] = 'deleted'
    newer.content = json.dumps(body)
    controller = HReConGraphMemory(
        SearchGraphiti([source_edge(['episode-old', 'episode-new'])], [older, newer])
    )

    context = await controller.retrieve('current Redis', user_id='u1')

    assert context.semantic_states == []
    assert controller.short_term_memories == {}
    assert context.source_episodes == [older, newer]


@pytest.mark.asyncio
async def test_colliding_memory_or_state_id_cannot_overwrite_other_users_cache():
    written, source = enveloped_source()
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))
    cached = add_local_memory(controller, written.memory_id, user_id='u2')
    old_state = controller.semantic_states[cached.semantic_state_id]

    context = await controller.retrieve('current Redis', user_id='u1')

    assert context.semantic_states == []
    assert controller.short_term_memories[written.memory_id] is cached
    assert controller.semantic_states[cached.semantic_state_id] is old_state
    assert cached.access_count == 0


@pytest.mark.asyncio
async def test_state_id_collision_does_not_replace_an_unrelated_state():
    _, source = enveloped_source()
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))
    unrelated = add_local_memory(controller, 'unrelated', user_id='u2')
    old_state = controller.semantic_states[unrelated.semantic_state_id]
    body = json.loads(source.content)
    body['h_recongraph']['semantic_state']['state_id'] = old_state.state_id
    source.content = json.dumps(body)

    context = await controller.retrieve('current Redis', user_id='u1')

    assert context.semantic_states == []
    assert set(controller.short_term_memories) == {'unrelated'}
    assert controller.semantic_states[old_state.state_id] is old_state


@pytest.mark.asyncio
async def test_write_restart_read_round_trip_preserves_actual_payload_and_no_read_writes():
    class EnvelopeGraphiti(SearchGraphiti):
        def __init__(self):
            super().__init__([])
            self.writes = []

        async def add_episode(self, **kwargs):
            self.writes.append(kwargs)
            source = source_episode(f'episode-{len(self.writes)}', kwargs['episode_body'])
            self.episodes.append(source)
            edge = source_edge([source.uuid])
            self.edges = [edge]
            return AddEpisodeResults(
                episode=source,
                episodic_edges=[],
                nodes=[],
                edges=[edge],
                communities=[],
                community_edges=[],
            )

    client = EnvelopeGraphiti()
    writer = HReConGraphMemory(client, policy=MemoryPolicy(consolidation_threshold=2))
    written = await writer.add_interaction(
        'remember this current project uses Redis because compatibility matters',
        user_id='u1',
        session_id='old-session',
    )
    await writer.store.add_memory(written)
    assert written.metadata['graphiti_episode_ids'] == ['episode-1', 'episode-2']
    reader = HReConGraphMemory(client)

    context = await reader.retrieve('current Redis', user_id='u1', session_id='new-session')

    restored = reader.short_term_memories[written.memory_id]
    assert context.semantic_states[0].model_dump(mode='json') == written.semantic_state
    assert restored.raw_content == written.raw_content
    assert restored.ttl == written.ttl
    assert restored.created_at == written.created_at
    assert restored.session_id == 'old-session'
    assert restored.metadata['graphiti_episode_ids'] == ['episode-1', 'episode-2']
    assert restored.metadata['graphiti_episode_id'] == 'episode-2'
    assert restored.graph_binding.edges == ['edge-a']
    assert restored.access_count == 1
    assert len(client.writes) == 2
    assert client.episode_reads == [['episode-2']]


@pytest.mark.asyncio
async def test_legacy_naive_timestamp_uses_existing_graphiti_utc_convention():
    _, source = enveloped_source()
    body = json.loads(source.content)
    body['h_recongraph']['created_at'] = utc_now().replace(tzinfo=None).isoformat()
    body['h_recongraph']['ttl'] = 'P1D'
    source.content = json.dumps(body)
    controller = HReConGraphMemory(SearchGraphiti([source_edge(['episode-a'])], [source]))

    context = await controller.retrieve('current Redis', user_id='u1')

    assert len(context.semantic_states) == 1
    restored = next(iter(controller.short_term_memories.values()))
    assert restored.created_at.tzinfo is not None
    assert not restored.is_expired()


@pytest.mark.asyncio
async def test_failed_source_read_does_not_count_m0_as_a_completed_retrieval():
    client = SearchGraphiti([source_edge(['episode-a'])])
    controller = HReConGraphMemory(client)
    instant = controller.buffer.add('Redis', user_id='u1')
    controller.store.get_source_episodes = AsyncMock(side_effect=RuntimeError('read failed'))

    with pytest.raises(RuntimeError, match='read failed'):
        await controller.retrieve('current Redis', user_id='u1')

    assert instant.access_count == 0


@pytest.mark.asyncio
async def test_context_text_restores_envelope_content_and_keeps_selected_graph_evidence():
    written, source = enveloped_source()
    edge = source_edge(['episode-a'])
    controller = HReConGraphMemory(SearchGraphiti([edge], [source]))

    context = await controller.retrieve('current Redis', user_id='u1')

    assert context.context_text == context.render_text()
    assert f'[SOURCE {source.uuid}] {written.raw_content}' in context.context_text
    assert 'h_recongraph' not in context.context_text
    assert '[EDGE] node-a -> node-b: Project A uses Redis' in context.context_text
    assert f'[STATE {written.memory_id}]' in context.context_text


@pytest.mark.asyncio
async def test_context_text_keeps_plain_legacy_source_verbatim():
    source = source_episode('episode-plain', 'Plain legacy source text.')
    edge = source_edge(['episode-plain'])
    controller = HReConGraphMemory(SearchGraphiti([edge], [source]))

    context = await controller.retrieve('Project A', user_id='u1')

    assert context.context_text.endswith('[SOURCE episode-plain] Plain legacy source text.')


@pytest.mark.asyncio
async def test_m0_only_context_text_is_available_without_graph_store():
    controller = HReConGraphMemory()
    await controller.add_interaction('current project deadline', user_id='u1', session_id='s1')

    context = await controller.retrieve('current project', user_id='u1', session_id='s1')

    assert context.context_text.startswith('[M0] current project deadline')
