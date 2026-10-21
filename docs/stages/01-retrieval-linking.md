# 01 读取关联

## 核心改动

将“全部本地状态参与返回”改为“依据实际命中结果选择支持状态”。图写入结果中的节点、关系和 episode 建立记忆绑定，读取端沿这些绑定关联本地语义状态。

## 代码变更

- `graphiti_store.py / add_memory()`：保存节点 UUID；仅绑定来源中包含当前 episode 的关系，区分当前关系和被此次写入作废的旧关系；记录当前 episode UUID。
- `manager.py / retrieve()`：先排序并限额 M0 与图关系，再根据 M0 证据、关系 UUID、episode 来源及显式证据查询选择语义状态。
- `test_h_recongraph_memory.py`：TTL 测试将 created_at 回拨一秒。
- 新增 `test_h_recongraph_graphiti_store.py`、`test_h_recongraph_retrieval.py`。

核心关系：

```python
selected_edge_ids.intersection(memory.graph_binding.edges)
selected_evidence_ids.intersection(memory.evidence_ids)
```

示例：图命中 Project A 的关系时，返回支持该关系的语义状态，其他 Project B 状态保留在本地缓存。

逐行差异：[store](../patches/01_graphiti_store.patch)、[manager](../patches/02_manager.patch)、[TTL 测试](../patches/03_test_h_recongraph_memory.patch)。

以下保留本批修改前后的完整代码、测试构造与输出示例。

### 1. 文件 01：h_recongraph/memory/graphiti_store.py / 修改前：完整 add_memory 函数

```python
    async def add_memory(self, memory: UnifiedMemoryUnit) -> Any:
        source_description = self._source_description(memory)
        episode_body = self._episode_body(memory)
        return await self.graphiti.add_episode(
            name=f'{memory.layer.lower()}_{memory.memory_id}',
            episode_body=episode_body,
            source_description=source_description,
            reference_time=memory.created_at or utc_now(),
            source=EpisodeType.text,
            group_id=self.group_id(memory.user_id),
        )
```

### 2. 文件 01：h_recongraph/memory/graphiti_store.py / 修改后：完整 add_memory 函数

```python
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
```

### 3. 文件 01：h_recongraph/memory/graphiti_store.py / 逐段解释：每个新增表达式具体做什么

```python
# 固定 ID 的集合合并示例。
old_nodes = ['node-a']
returned_nodes = ['node-a', 'node-b']
# 拼接：['node-a', 'node-a', 'node-b']
# dict.fromkeys 按首次出现去重：['node-a', 'node-b']
```

### 4. 文件 01：h_recongraph/memory/graphiti_store.py / 逐段解释：每个新增表达式具体做什么

```python
                entity_edges = resolved_edges + invalidated_edges
```

### 5. 文件 01：h_recongraph/memory/graphiti_store.py / 逐段解释：每个新增表达式具体做什么

```python
if episode is not None and episode.uuid in edge.episodes
```

### 6. 文件 02：h_recongraph/memory/manager.py / 改后行为

```text
Graphiti.search → 原排序器 → 现有数量上限 → 取得返回关系 UUID / 来源 episode UUID
选中 M0 → 现有数量上限 → 取得证据 ID
上述 ID → 对应的本地记忆 → 该记忆的语义状态
```

### 7. 文件 02：h_recongraph/memory/manager.py / 修改前：完整 retrieve 函数

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
            )
            for memory in instant_memories:
                memory.touch()

        if self.store is not None and any(layer in route.layers for layer in ['M1', 'M2']):
            search_results = await self.store.search_memories(
                query,
                user_id,
                num_results=self.policy.max_retrieval_results,
            )
            graph_edges = self.ranker.rank_graph_edges(list(search_results or []), query)

        relevant_ids = {memory.memory_id for memory in instant_memories}
        relevant_ids.update(self.short_term_memories)
        relevant_ids.update(self.long_term_memories)
        local_states = [
            state
            for state in self.semantic_states.values()
            if state.edge_or_subgraph_id in relevant_ids
            or any(evidence_id in query for evidence_id in state.evidence_ids)
        ]

        return self.relinker.build_context(
            query=query,
            selected_layers=route.layers,
            instant_memories=instant_memories[: self.policy.max_retrieval_results],
            graph_edges=graph_edges[: self.policy.max_retrieval_results],
            semantic_states=local_states[: self.policy.max_retrieval_results],
        )
```

### 8. 文件 02：h_recongraph/memory/manager.py / 修改后：完整 retrieve 函数

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

### 9. 文件 02：h_recongraph/memory/manager.py / 分为六段理解实际执行过程

```python
if memory.user_id != user_id or memory.layer not in route.layers:
    continue
```

### 10. 文件 02：h_recongraph/memory/manager.py / 一次调用的完整数据演算

```text
query = "Project A"
user_id = "u1"
route.layers = [M0, M1, M2]
选中 M0 = []
返回边 = [{uuid: edge-a, episodes: [episode-a]}]

本地记忆按插入顺序：
  B：user=u1，M1，edges=[edge-b]，episode=episode-b
  A：user=u1，M1，edges=[edge-a]，episode=episode-a
  C：user=u2，M1，edges=[edge-a]，episode=episode-a

selected_evidence_ids = {}
selected_edge_ids = {edge-a}
selected_episode_ids = {episode-a}

B：用户/层通过，四个匹配分支全不成立 → 不加入
A：用户/层通过，edge 和 episode 分支成立 → 加入 memory_id(A)
C：user 不同 → 不参与后面四个分支

local_states = [state(A)]
graph_edges = [edge-a]
```

### 11. 文件 03：tests/test_h_recongraph_memory.py / 修改前：完整变动测试

```python
@pytest.mark.asyncio
async def test_maintain_expires_m0_buffer_items():
    memory = HReConGraphMemory(policy=MemoryPolicy(m0_ttl=timedelta(microseconds=1)))
    await memory.add_interaction('好的', user_id='u1', session_id='s1')

    result = await memory.maintain(user_id='u1')

    assert len(result['expired_m0']) == 1
```

### 12. 文件 03：tests/test_h_recongraph_memory.py / 修改后：完整变动测试

```python
@pytest.mark.asyncio
async def test_maintain_expires_m0_buffer_items():
    memory = HReConGraphMemory(policy=MemoryPolicy(m0_ttl=timedelta(microseconds=1)))
    instant = await memory.add_interaction('好的', user_id='u1', session_id='s1')
    instant.created_at -= timedelta(seconds=1)

    result = await memory.maintain(user_id='u1')

    assert len(result['expired_m0']) == 1
```

### 13. 文件 04：tests/test_h_recongraph_graphiti_store.py / 新增文件完整代码

```python
import json
from types import SimpleNamespace

import pytest

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
```

### 14. 文件 05：tests/test_h_recongraph_retrieval.py / 新增文件完整代码

```python
from types import SimpleNamespace

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph import HReConGraphMemory, MemoryPolicy
from h_recongraph.schemas.memory_unit import GraphBinding, UnifiedMemoryUnit


class SearchGraphiti:
    def __init__(self, edges):
        self.edges = edges
        self.searches = []

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
```

## 验证结果

第一批：26 passed，包含原有 9 项和新增 17 项。对应 Python 文件通过 Ruff check 与格式检查。

TTL 测试通过回拨创建时间固定过期条件；原版复核为 8 passed / 1 failed，修复后原有 9 项全部通过。

### 1. 批次验证与复现

```powershell
cd 'C:\Users\AIR\Desktop\01_Folders\graphiti-main\26-10-16-逐代码审核'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\AIR\Desktop\01_Folders\graphiti-main\graphiti-main\.venv\Scripts\python.exe' -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py -q -p no:cacheprovider
```
