# 第六批：证据片段组装

## 核心改动

增加 `full_source` 与 `evidence_bundle` 两种读取文本模式。片段模式以命中关系事实和四维 `source_spans` 为锚点，保留对应原文段落并合并重复来源；改写关系沿用完整来源，等待原文锚点补齐。

读取数据流：

```text
命中关系 + semantic_state.source_spans
  -> 定位来源段落
  -> 合并重复来源
  -> 输出 evidence_bundle
```

## 代码变更

- `relinking.py`：上下文增加 `source_mode`，并按模式装配来源文本。
- `manager.py`：检索结果继续携带来源 episode 与语义状态，供片段装配使用。
- `test_h_recongraph_retrieval.py`：验证来源裁剪、去重与全文模式默认行为。

示例：事实段、路径段与原因段构成证据束；无关午餐段落被裁去。

[完整逐文件报告](../details/06-证据片段组装逐文件交付.md)，含代码对比、示例与实验分析。

## 验证结果

- 指定三文件回归：**111 passed**。
- 改动文件 Ruff check、format check 与 `git diff --check` 通过。
- 本批建立读取装配机制；固定样本配对实验在第七批进行。
