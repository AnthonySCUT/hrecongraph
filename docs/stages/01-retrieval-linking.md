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

## 验证结果

第一批：26 passed，包含原有 9 项和新增 17 项。对应 Python 文件通过 Ruff check 与格式检查。
