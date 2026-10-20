# 第五批：M0 相关性与稳定排序

## 核心改动

读取链路统一为：**问题 → 路由 → 有效词面/证据 ID 召回 M0 → 按本次得分排序 → 按原有限额返回 → 更新实际访问次数**。

- M0 与本地 M1/M2 回退复用已有分词：英文内容词、中文相邻双字词及停用词过滤。
- 无有效重合时返回空 M0；按完整证据 ID 查询时返回对应 M0。
- 排序读取已有基础分数，本次查询得分独立计算，消除重复读取导致的分数累加。

研究作用：让短时记忆与问题相关，减少进入回答上下文的无关原文；使不同查询与重复实验中的排序可比较。

示例：M0 同时包含“项目接口因为联调而延期”“项目预算等待审批”“午餐吃面条”，查询“项目接口延期原因”，限额为 1 时选中第一条。只有无关内容或仅共享停用词时，M0 上下文为空。

## 代码变更

本批起点：第四批完成的 101 项测试。业务修改涉及两个现有文件，测试补充于原有 retrieval 文件。

### manager.py：M0 候选召回

改前以空格切词；匹配为空时取最近三条。改后调用现有 lexical_relevance，并保留证据 ID 直接定位。

修改前完整函数：

```python
    def _matching_m0(
        self,
        query: str,
        user_id: str,
        session_id: str | None,
    ) -> list[UnifiedMemoryUnit]:
        query_tokens = {token for token in query.lower().split() if len(token) > 1}
        memories = self.buffer.list(user_id=user_id, session_id=session_id)
        if not query_tokens:
            return memories
        matching = []
        for memory in memories:
            text_tokens = {token for token in (memory.raw_content or '').lower().split()}
            if query_tokens & text_tokens:
                matching.append(memory)
        return matching or memories[:3]
```

修改后完整函数：

```python
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
```

返回的候选继续经过 retrieve 中已有的 rank_memories 和 max_retrieval_results 切片。进入最终上下文的对象由已有 touch 流程计数。

### ranker.py：统一词面得分、消除排序副作用

score_memory 的相关性项改为现有分词后的查询覆盖率：

\[
r(m,q)=\frac{|T(m)\cap T(q)|}{|T(q)|}
\]

其中 T 为已有 tokenize；查询内容词集合为空时相关性为 0。其余评分项沿用当前公式：

\[
S(m,q)=\operatorname{round}\!\left(\max\left(0,r+u+\mathrm{recency}+\mathrm{evidence}+\mathrm{importance}-\mathrm{contradiction}-\mathrm{privacy}\right),3\right).
\]

u 是已有 retrieval_utility。排序过程中保持其值，访问频次由 access_count 单独记录。

修改前完整函数：

```python
    def score_memory(self, memory: UnifiedMemoryUnit, query: str) -> float:
        semantic_relevance = self._token_overlap(memory.raw_content or '', query)
        scores = memory.scores
        score = (
            semantic_relevance
            + scores.retrieval_utility
            + scores.recency
            + scores.evidence_strength
            + scores.importance
            - scores.contradiction_risk
            - scores.privacy_risk
        )
        return round(max(score, 0.0), 3)
```

```python
    def rank_memories(
        self, memories: list[UnifiedMemoryUnit], query: str
    ) -> list[UnifiedMemoryUnit]:
        ranked = sorted(memories, key=lambda memory: self.score_memory(memory, query), reverse=True)
        for memory in ranked:
            memory.scores.retrieval_utility = self.score_memory(memory, query)
        return ranked
```

修改后完整函数：

```python
    def score_memory(self, memory: UnifiedMemoryUnit, query: str) -> float:
        semantic_relevance = self.lexical_relevance(memory.raw_content or '', query)
        scores = memory.scores
        score = (
            semantic_relevance
            + scores.retrieval_utility
            + scores.recency
            + scores.evidence_strength
            + scores.importance
            - scores.contradiction_risk
            - scores.privacy_risk
        )
        return round(max(score, 0.0), 3)
```

```python
    def rank_memories(
        self, memories: list[UnifiedMemoryUnit], query: str
    ) -> list[UnifiedMemoryUnit]:
        return sorted(memories, key=lambda memory: self.score_memory(memory, query), reverse=True)
```

### test_h_recongraph_retrieval.py：新增核心用例

新增导入：from h_recongraph.memory.ranker import MemoryUtilityRanker。

五个测试函数共六个用例：中文召回与限额、无关内容、纯停用词、证据 ID 定位、重复排序稳定性、中文排序。

```python
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
```

独立差异文件：[05-M0相关性与稳定排序.patch](../patches/05-M0相关性与稳定排序.patch)。补丁以本批开始时的三个文件为基准。

## 验证结果

- 修改前回归：101 passed。
- 先加入新测试、保留原业务实现：5 failed、1 passed，复现候选误召回与排序分数累加。
- 修改后指定三文件：**107 passed**，1 个已有 Pydantic Config 弃用提示。
- 三个改动 Python 文件：Ruff check 通过，format --check 通过。
- git diff --check 通过。

在发布目录执行：

```powershell
python -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py -q -p no:cacheprovider
```

本批验证了读取候选相关性、返回限额、访问计数及重复排序稳定性。
