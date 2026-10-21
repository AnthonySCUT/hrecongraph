# 11：LongMemEval 轮次级证据束配对

日期：2026-10-21。工作副本：`26-10-16-逐代码审核`。

## 核心改动

本批把第十批的自然文本四维绑定接入 LongMemEval 读取评测：每个命中 Session 按对话轮次组织为段落，写入 `HReConGraphMemory` 后由 `SemanticStateConstructor` 自动生成 `fact / constraint / path / causal` 证据锚点，再由图关系驱动 `full_source` 与 `evidence_bundle` 配对读取。

```text
LongMemEval Session
  -> turn-level source episodes
  -> H-ReConGraph M1 memory
  -> automatic four-dimensional source_spans
  -> query-related graph fact
  -> full_source / evidence_bundle
  -> token, evidence-turn, answer-visible, latency metrics
```

## 代码变更

### `evaluate_read_context.py`

新增 `build_naturally_bound_contexts()`：写入时不传外部 `read_context_write_bindings.json`，直接调用现有 `HReConGraphMemory.add_memory()` 完成自然文本自动绑定，再从 episode envelope 恢复读取上下文。新增 `--natural-bindings` 入口，并补充 Token 统计与相对 `full_source` 的缩减率。

16 个四维机制样本的自然自动绑定结果：

| 指标 | `full_source` | `evidence_bundle` |
|---|---:|---:|
| 样本数 | 16 | 16 |
| 证据保留率 | 100.00% | 74.29% |
| 完整样本数 | 16 | 7 |
| 探针正确数 | 16 | 8 |
| 总 Token | 1,227 | 642 |
| Token 缩减率 | 0.00% | 47.68% |

四个维度的完整样本数为：fact `1/4`、constraint `4/4`、path `1/4`、causal `1/4`。该结果直接暴露自然文本自动绑定中事实、路径和因果依赖的段落召回缺口。

### `evaluate_longmemeval_read.py`

重写 LongMemEval 读取协议：

- 按对话轮次生成来源段落，每个命中 Session 保留完整轮次序列。
- 使用现有 `MemoryUtilityRanker` 选择与问题相关的 Session，所有命中 Session 进入同一对照上下文。
- 对每个命中 Session 写入 `HReConGraphMemory`，由现有 `SemanticStateConstructor` 生成自动四维 `source_spans`。
- 将最相关轮次作为图关系事实，调用真实 `context.render_text('full_source')` 与 `context.render_text('evidence_bundle')`。
- 统计答案字符串可见率、`has_answer` 证据轮次保留率、上下文字符数、估算 Token、Token 缩减率、渲染延迟，并按题型汇总。

## 验证结果

500 题 LongMemEval oracle：

| 指标 | `full_source` | `evidence_bundle` |
|---|---:|---:|
| 平均上下文字符数 | 28,789.93 | 17,734.65 |
| 平均上下文 Token | 7,197.12 | 4,433.27 |
| 平均 Token 缩减率 | 0.00% | 38.46% |
| 答案可见率 | 46.60% | 43.20% |
| 证据轮次保留率 | 98.77% | 75.64% |
| 中位渲染耗时 | 0.611 ms | 5.101 ms |

按题型的 `evidence_bundle` 结果：

| 题型 | 样本数 | Token 缩减率 | 答案可见率 | 证据轮次保留率 |
|---|---:|---:|---:|---:|
| `knowledge-update` | 78 | 40.75% | 67.95% | 78.42% |
| `multi-session` | 133 | 38.37% | 36.09% | 76.00% |
| `single-session-assistant` | 56 | 36.85% | 48.21% | 73.21% |
| `single-session-preference` | 30 | 46.63% | 0.00% | 55.56% |
| `single-session-user` | 70 | 37.08% | 77.14% | 82.14% |
| `temporal-reasoning` | 133 | 36.75% | 25.56% | 75.79% |

指定回归测试：**137 passed，1 条既有 Pydantic 弃用提示**。新增评测代码与测试通过 Ruff check 和 `ruff format --check`。
