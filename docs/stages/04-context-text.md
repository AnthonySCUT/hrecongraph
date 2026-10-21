# 04 读取证据文本组装

## 核心改动

将选中的 M0 原文、语义字段、图关系和来源正文组织为 context_text，形成读取端可直接使用的文本输入。

## 代码变更

- `relinking.py / HierarchicalMemoryContext`：增加 context_text、render_text()、_source_text()。
- `render_text()`：依次渲染 [M0]、[STATE]、[EDGE]、[SOURCE]，相同完整文本块去重。
- `_source_text()`：读取来源 envelope 的 content 正文，普通文本沿用原文。
- `manager.py / retrieve()`：来源装入后生成最终文本。
- retrieval 测试增加 envelope 来源、普通来源和纯 M0 三项验证。

```python
context.source_episodes = source_episodes
context.context_text = context.render_text()
```

例：[EDGE] 保留关系方向与事实文本，[SOURCE] 提供相应原始叙述，[STATE] 补充时间、条件或原因等结构化字段。

完整差异：[04 patch](../patches/04-读取证据文本组装.patch)。

以下保留本批修改前后的完整代码、测试构造与输出示例。

### 1. 2. 读取链路的改后形态

```text
query
  -> 原有路由
  -> 原有 M0 / Graphiti 关系检索与排序
  -> 原有 source_episodes 按命中 episode ID 回溯
  -> 第三批按需恢复本地状态
  -> 构造 HierarchicalMemoryContext
  -> render_text()
  -> context_text
```

### 2. 3. relinking.py / 3.1 改前完整文件

```python
"""Query-time cross-layer relinking."""

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

    model_config = {'arbitrary_types_allowed': True}


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

        return HierarchicalMemoryContext(
            query=query,
            selected_layers=selected_layers,
            instant_memories=instant_memories,
            graph_edges=graph_edges,
            semantic_states=semantic_states,
            ephemeral_edges=ephemeral_edges,
            evidence_trace=list(dict.fromkeys(evidence_trace)),
        )

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
```

### 3. 3. relinking.py / 3.2 改后完整文件

```python
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
```

### 4. 3. relinking.py / 3.3 逐段说明 / A. 新增导入

```python
import json
```

### 5. 3. relinking.py / 3.3 逐段说明 / B. 新增字段

```python
context_text: str = ''
```

### 6. 3. relinking.py / 3.3 逐段说明 / C. render_text()

```python
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
```

### 7. 3. relinking.py / 3.3 逐段说明 / D. _source_text()

```python
    @staticmethod
    def _source_text(content: str) -> str:
        try:
            envelope = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return content
        if isinstance(envelope, dict) and isinstance(envelope.get('content'), str):
            return envelope['content']
        return content
```

### 8. 3. relinking.py / 3.3 逐段说明 / D. _source_text()

```json
{"h_recongraph": {...}, "content": "Project A uses Redis."}
```

### 9. 3. relinking.py / 3.3 逐段说明 / D. _source_text()

```text
Project A uses Redis.
```

### 10. 3. relinking.py / 3.3 逐段说明 / E. build_context()

```python
context.context_text = context.render_text()
```

### 11. 4. manager.py / 4.1 改前 retrieve 完整函数

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
```

### 12. 4. manager.py / 4.2 改后 retrieve 完整函数

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
```

### 13. 4. manager.py / 4.3 实际新增位置

```python
context.source_episodes = source_episodes
context.context_text = context.render_text()
```

### 14. 5. 输出示例

```text
[M0] current project uses Redis
[STATE stm-a] cause=compatibility matters [evidence=im-a]
[EDGE] node-a -> node-b: Project A uses Redis
[SOURCE episode-a] Project A uses Redis because compatibility matters.
```

### 15. 6. 测试新增代码

```python
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
```

## 验证结果

第四批累计 101 passed；改动文件通过 Ruff check、format --check 和差异检查。

新增 3 项核心测试，覆盖 envelope 来源、普通来源与纯 M0 文本组装。

### 1. 7. 验证结果

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\AIR\Desktop\01_Folders\graphiti-main\graphiti-main\.venv\Scripts\python.exe' -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py -q -p no:cacheprovider

& 'C:\Users\AIR\Desktop\01_Folders\graphiti-main\graphiti-main\.venv\Scripts\ruff.exe' check h_recongraph/memory/relinking.py h_recongraph/memory/manager.py tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py

& 'C:\Users\AIR\Desktop\01_Folders\graphiti-main\graphiti-main\.venv\Scripts\ruff.exe' format --check h_recongraph/memory/relinking.py h_recongraph/memory/manager.py tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py

git diff --check
```
