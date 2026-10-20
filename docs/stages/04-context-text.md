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

## 验证结果

第四批累计 101 passed；改动文件通过 Ruff check、format --check 和差异检查。
