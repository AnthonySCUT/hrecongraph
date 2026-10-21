# 第十二批：证据召回、窗口与成本 Pareto 消融

## 核心改动

固定检索结果，比较 `full_source` 与 `evidence_bundle(w=0/1/2)`。装配顺序为事实锚点、四维依赖、同 Session 邻近轮次窗口、索引求并集和原序输出；分别记录各阶段新增的支持证据。

## 代码变更

- `relinking.py`：增加 `window_size`，以来源索引并集装配轮次或段落。
- `evaluate_longmemeval_read.py`：对 Oracle 500 题输出逐题索引、题型统计、双召回口径及 Pareto 分析。
- `evaluate_read_context.py`：对 16 个机制样本比较自然绑定与显式绑定。

[完整逐文件报告](../details/12-证据裁剪召回成本消融与Pareto探索逐文件交付.md)，含代码对比、示例与实验分析。

## 验证结果

LongMemEval Oracle 500 题，479 题有支持轮次标注，共 896 条：

| 配置 | 平均 Token | 整体召回 | 完整保留题数 |
|---|---:|---:|---:|
| `full_source` | 7,246.39 | 98.21% | 469/479 |
| `evidence_bundle(w=0)` | 6,727.29 | 75.45% | 301/479 |
| `evidence_bundle(w=1)` | 7,149.28 | 97.43% | 462/479 |
| `evidence_bundle(w=2)` | 7,236.29 | 97.88% | 466/479 |

事实锚点阶段保留 410 条支持轮次；展开依赖后增至 676 条（+266）；再加 `w=1` 增至 873 条（+197）。窗口依赖的证据召回增量明显，整体 Token 缩减空间则受到依赖来源文本成本影响。

- 指定回归：**142 passed**。
- Ruff check、format check、`git diff --check` 与逐题结果核对通过。
