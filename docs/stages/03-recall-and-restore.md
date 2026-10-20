# 03 召回回退、访问计数与状态恢复

## 核心改动

图关联状态为空时，以有界词面召回补充本地记忆；实际返回的 M0/M1/M2 统一累计访问。来源 envelope 支持新 controller 恢复语义状态及命中图绑定。

## 代码变更

- `manager.py / retrieve()`：先选直接关联状态，关联为空时按有效词面重合与记忆分数排序；应用 max_retrieval_results；完整 ID token 匹配；对最终返回对象按 memory_id 去重 touch。
- `manager.py / _restore_source_memories()`：解析本次命中来源，恢复对应的 UnifiedMemoryUnit、MemorySemanticState 和关系绑定；同一记忆的多份快照按版本与时间选择并合并绑定。
- `graphiti_store.py / episode_ids()、add_memory()、_episode_body()`：追加去重的 graphiti_episode_ids；旧 graphiti_episode_id 指向最新 ID；envelope 保存恢复所需的记忆和状态字段。
- `ranker.py / tokenize()、lexical_relevance()`：提供内容词过滤、英文词与中文双字词重合。
- 在两份已有测试中增加对应验证。

```python
memory.metadata['graphiti_episode_ids'] = list(
    dict.fromkeys([*self.episode_ids(memory), episode.uuid])
)
memory.metadata['graphiti_episode_id'] = episode.uuid
```

例如 controller 重建后，检索关系指向已存储来源；来源中的 memory_id、layer 和 semantic_state 使相应语义状态重新进入读取上下文。访问计数在本地对象上更新。

完整差异：[03 patch](../patches/03-召回回退与状态恢复.patch)。

## 验证结果

第三批累计 98 passed。改动文件通过 Ruff check、format --check 和 git diff --check。
