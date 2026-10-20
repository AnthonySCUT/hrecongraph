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

## 验证结果

第二批累计 40 passed；改动文件通过 Ruff check、format --check 和差异检查。
