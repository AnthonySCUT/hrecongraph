# 02 来源原文回溯

## 核心改动

从最终选中的关系出发，收集来源 episode UUID，批量取回原文，将关系证据与正文一起返回。

## 代码变更

- `graphiti_store.py / get_source_episodes()`：汇总关系的 episodes，按首次引用顺序去重，通过 EpisodicNode.get_by_uuids 批量获取。
- `relinking.py / HierarchicalMemoryContext`：增加 source_episodes。
- `manager.py / retrieve()`：将所选关系的来源装入上下文。
- 两份已有 store / retrieval 测试各补充 7 项。

数据流：

```text
selected graph edges -> unique episode IDs -> batch read -> context.source_episodes
```

例如两条关系共同引用同一 episode，正文在来源集合中出现一次，保持原始内容。

完整差异：[02 patch](../patches/02-来源原文回溯.patch)。

以下保留本批修改前后的完整代码、测试构造与输出示例。

### 1. 1. 本批解决的问题与范围

```text
问题
  -> 原有 Graphiti.search：按问题检索关系
  -> 原有排序与结果数量限制
  -> 最终返回关系的 episodes UUID
  -> 去重后调用现有 EpisodicNode.get_by_uuids
  -> HierarchicalMemoryContext.source_episodes
```

### 2. 2. graphiti_store.py：来源读取的具体实现 / 2.1 改前行为与完整文件

```python
"""Adapter between H-ReConGraph and Graphiti."""

import json
from typing import Any

from graphiti_core.nodes import EpisodeType
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

    def group_id(self, user_id: str) -> str:
        return f'user:{user_id}'

    def _source_description(self, memory: UnifiedMemoryUnit) -> str:
        return (
            'H-ReConGraph memory '
            f'layer={memory.layer} '
            f'session_id={memory.session_id or ""} '
            f'lifecycle_status={memory.lifecycle_status} '
            f'memory_id={memory.memory_id}'
        )

    def _episode_body(self, memory: UnifiedMemoryUnit) -> str:
        envelope = {
            'h_recongraph': {
                'memory_id': memory.memory_id,
                'layer': memory.layer,
                'session_id': memory.session_id,
                'lifecycle_status': memory.lifecycle_status,
                'evidence_ids': memory.evidence_ids,
                'semantic_state': memory.semantic_state,
                'scores': memory.scores.model_dump(),
            },
            'content': memory.raw_content,
        }
        return json.dumps(envelope, ensure_ascii=False, default=str)
```

### 3. 2. graphiti_store.py：来源读取的具体实现 / 2.2 改后完整文件

```python
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

    def _source_description(self, memory: UnifiedMemoryUnit) -> str:
        return (
            'H-ReConGraph memory '
            f'layer={memory.layer} '
            f'session_id={memory.session_id or ""} '
            f'lifecycle_status={memory.lifecycle_status} '
            f'memory_id={memory.memory_id}'
        )

    def _episode_body(self, memory: UnifiedMemoryUnit) -> str:
        envelope = {
            'h_recongraph': {
                'memory_id': memory.memory_id,
                'layer': memory.layer,
                'session_id': memory.session_id,
                'lifecycle_status': memory.lifecycle_status,
                'evidence_ids': memory.evidence_ids,
                'semantic_state': memory.semantic_state,
                'scores': memory.scores.model_dump(),
            },
            'content': memory.raw_content,
        }
        return json.dumps(envelope, ensure_ascii=False, default=str)
```

### 4. 2. graphiti_store.py：来源读取的具体实现 / 2.4 示例

```json
[
  {"uuid": "edge-a", "episodes": ["episode-a", "episode-b"]},
  {"uuid": "edge-b", "episodes": ["episode-a"]}
]
```

### 5. 3. relinking.py：上下文如何承载来源 / 3.1 改前完整上下文定义

```python
class HierarchicalMemoryContext(BaseModel):
    """Query-specific hierarchical memory evidence graph."""

    query: str
    selected_layers: list[Literal['M0', 'M1', 'M2']]
    instant_memories: list[UnifiedMemoryUnit] = Field(default_factory=list)
    graph_edges: list[Any] = Field(default_factory=list)
    semantic_states: list[MemorySemanticState] = Field(default_factory=list)
    ephemeral_edges: list[ReConMemoryEdge] = Field(default_factory=list)
    evidence_trace: list[str] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}
```

### 6. 3. relinking.py：上下文如何承载来源 / 3.2 改后完整上下文定义

```python
from graphiti_core.nodes import EpisodicNode
```

### 7. 3. relinking.py：上下文如何承载来源 / 3.2 改后完整上下文定义

```python
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

    model_config = {'arbitrary_types_allowed': True}
```

### 8. 3. relinking.py：上下文如何承载来源 / 3.3 逐段说明

```python
    source_episodes: list[EpisodicNode] = Field(default_factory=list)
```

### 9. 3. relinking.py：上下文如何承载来源 / 3.4 示例与兼容性

```text
graph_edges[0].episodes = ["episode-a"]
source_episodes[0].uuid = "episode-a"
source_episodes[0].content = 数据库中该来源的原样内容
```

### 10. 4. manager.py：在什么位置查询来源 / 4.1 改前完整 retrieve

```python
    async def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
    ) -> HierarchicalMemoryContext:
        """Retrieve and relink M0/M1/M2 memories for a query-specific evidence graph."""

        route = self.router.route(query)
        instant_memories: list[UnifiedMemoryUnit] = []
        graph_edges: list[Any] = []

        if 'M0' in route.layers:
            instant_memories = self.ranker.rank_memories(
                self._matching_m0(query, user_id, session_id), query
            )[: self.policy.max_retrieval_results]
            for memory in instant_memories:
                memory.touch()

        if self.store is not None and any(layer in route.layers for layer in ['M1', 'M2']):
            search_results = await self.store.search_memories(
                query,
                user_id,
                num_results=self.policy.max_retrieval_results,
            )
            graph_edges = self.ranker.rank_graph_edges(list(search_results or []), query)[
                : self.policy.max_retrieval_results
            ]

        relevant_ids = {memory.memory_id for memory in instant_memories}
        selected_evidence_ids = {
            evidence_id for memory in instant_memories for evidence_id in memory.evidence_ids
        }
        selected_edge_ids = {edge.uuid for edge in graph_edges}
        selected_episode_ids = {
            episode_id for edge in graph_edges for episode_id in getattr(edge, 'episodes', [])
        }
        for memory in [*self.short_term_memories.values(), *self.long_term_memories.values()]:
            if memory.user_id != user_id or memory.layer not in route.layers:
                continue
            if (
                selected_evidence_ids.intersection(memory.evidence_ids)
                or selected_edge_ids.intersection(memory.graph_binding.edges)
                or memory.metadata.get('graphiti_episode_id') in selected_episode_ids
                or any(evidence_id in query for evidence_id in memory.evidence_ids)
            ):
                relevant_ids.add(memory.memory_id)
        local_states = [
            state
            for state in self.semantic_states.values()
            if state.edge_or_subgraph_id in relevant_ids
        ]

        return self.relinker.build_context(
            query=query,
            selected_layers=route.layers,
            instant_memories=instant_memories,
            graph_edges=graph_edges,
            semantic_states=local_states[: self.policy.max_retrieval_results],
        )
```

### 11. 4. manager.py：在什么位置查询来源 / 4.2 改后完整 retrieve

```python
    async def retrieve(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
    ) -> HierarchicalMemoryContext:
        """Retrieve and relink M0/M1/M2 memories for a query-specific evidence graph."""

        route = self.router.route(query)
        instant_memories: list[UnifiedMemoryUnit] = []
        graph_edges: list[Any] = []

        if 'M0' in route.layers:
            instant_memories = self.ranker.rank_memories(
                self._matching_m0(query, user_id, session_id), query
            )[: self.policy.max_retrieval_results]
            for memory in instant_memories:
                memory.touch()

        if self.store is not None and any(layer in route.layers for layer in ['M1', 'M2']):
            search_results = await self.store.search_memories(
                query,
                user_id,
                num_results=self.policy.max_retrieval_results,
            )
            graph_edges = self.ranker.rank_graph_edges(list(search_results or []), query)[
                : self.policy.max_retrieval_results
            ]

        relevant_ids = {memory.memory_id for memory in instant_memories}
        selected_evidence_ids = {
            evidence_id for memory in instant_memories for evidence_id in memory.evidence_ids
        }
        selected_edge_ids = {edge.uuid for edge in graph_edges}
        selected_episode_ids = {
            episode_id for edge in graph_edges for episode_id in getattr(edge, 'episodes', [])
        }
        for memory in [*self.short_term_memories.values(), *self.long_term_memories.values()]:
            if memory.user_id != user_id or memory.layer not in route.layers:
                continue
            if (
                selected_evidence_ids.intersection(memory.evidence_ids)
                or selected_edge_ids.intersection(memory.graph_binding.edges)
                or memory.metadata.get('graphiti_episode_id') in selected_episode_ids
                or any(evidence_id in query for evidence_id in memory.evidence_ids)
            ):
                relevant_ids.add(memory.memory_id)
        local_states = [
            state
            for state in self.semantic_states.values()
            if state.edge_or_subgraph_id in relevant_ids
        ]

        context = self.relinker.build_context(
            query=query,
            selected_layers=route.layers,
            instant_memories=instant_memories,
            graph_edges=graph_edges,
            semantic_states=local_states[: self.policy.max_retrieval_results],
        )
        if graph_edges and self.store is not None:
            context.source_episodes = await self.store.get_source_episodes(graph_edges, user_id)
        return context
```

### 12. 4. manager.py：在什么位置查询来源 / 4.3 逐段解释与副作用

```python
        context = self.relinker.build_context(
            query=query,
            selected_layers=route.layers,
            instant_memories=instant_memories,
            graph_edges=graph_edges,
            semantic_states=local_states[: self.policy.max_retrieval_results],
        )
        if graph_edges and self.store is not None:
            context.source_episodes = await self.store.get_source_episodes(graph_edges, user_id)
        return context
```

### 13. 5. test_h_recongraph_graphiti_store.py：测试设计与完整新增代码

```python
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
```

### 14. 6. test_h_recongraph_retrieval.py：控制器链路测试 / 6.1 假客户端改前完整定义

```python
class SearchGraphiti:
    def __init__(self, edges):
        self.edges = edges
        self.searches = []

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return self.edges
```

### 15. 6. test_h_recongraph_retrieval.py：控制器链路测试 / 6.2 假客户端改后完整定义

```python
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
```

### 16. 6. test_h_recongraph_retrieval.py：控制器链路测试 / 6.3 完整新增测试代码

```python
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
```

## 验证结果

第二批累计 40 passed；改动文件通过 Ruff check、format --check 和差异检查。

来源回溯新增 14 项测试，覆盖来源顺序、去重和控制器到适配层的数据流。

### 1. 7. 验证过程与实际结果

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\AIR\Desktop\01_Folders\graphiti-main\graphiti-main\.venv\Scripts\python.exe' -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py -q -p no:cacheprovider
```
