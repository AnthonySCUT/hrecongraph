# 第十一批：LongMemEval 读取成本配对

## 核心改动

将自然文本四维绑定接入 LongMemEval Oracle 读取实验。每个命中 Session 按原始对话轮次组织来源，由 `HReConGraphMemory` 写入并生成证据锚点，再使用真实上下文渲染分别比较 `full_source` 与 `evidence_bundle`。

## 代码变更

- `evaluate_read_context.py`：加入无外部绑定的 16 个机制样本自然绑定实验。
- `evaluate_longmemeval_read.py`：按 Session 与 Turn Index 构造来源，记录上下文字符、Token、支持轮次保留、答案可见率和渲染延迟，并按题型汇总。
- 评分使用官方答案与支持轮次标注；问题和原文用于检索及裁剪。

## 验证结果

LongMemEval Oracle 500 题：

| 指标 | `full_source` | `evidence_bundle` |
|---|---:|---:|
| 平均上下文 Token | 7,197.12 | 4,433.27 |
| Token 缩减率 | 0.00% | 38.46% |
| 支持证据轮次保留率 | 98.77% | 75.64% |
| 答案字符串可见率 | 46.60% | 43.20% |

16 个机制样本自然绑定组的必要证据片段保留率为 **74.29%**，完整样本 **7/16**；该结果显示事实与路径锚定仍是自然绑定的主要改进点。

- 指定回归：**137 passed**。
- Ruff check 与 format check 通过。
