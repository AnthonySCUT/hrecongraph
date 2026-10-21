# 读取与写入绑定实验复现

## 核心改动

实验分为 16 个机制样本与 LongMemEval Oracle 500 题。机制样本保存在 `h_recongraph/examples/`；LongMemEval 数据按下方步骤获取。

LongMemEval 的词面适配流程为：选取与问题存在有效词重合的 Session，每个 Session 取重合词最多的 Top-1 原始轮次，构造 `RELEVANT_TURN` 关系，再由实际记忆写入与上下文装配代码生成证据束。并列时取较早轮次。条件 Top-2 使用最高分的 50% 与两个不同重合实词作为阈值。

问题与原文决定命中关系和裁剪；`answer`、`has_answer`、机制样本必要片段用于装配后的评分。现有 500 题用于配置探索，后续确认实验按来源会话另行划分。

## 代码变更

安装依赖：

```powershell
python -m pip install -e ".[dev]"
```

数据来源：[LongMemEval 官方仓库](https://github.com/xiaowu0162/LongMemEval)，文件为官方 cleaned 数据中的 Oracle 版本。

```powershell
New-Item -ItemType Directory -Path data -Force
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json' -OutFile data/longmemeval_oracle.json
Get-FileHash data/longmemeval_oracle.json -Algorithm SHA256
```

本轮复核下载文件与实验输入一致，SHA256 为：

```text
821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c
```

机制样本自然绑定与完整显式绑定：

```powershell
python -m h_recongraph.examples.evaluate_read_context --natural-bindings --repeats 3 --output-json results/natural-bindings.json
python -m h_recongraph.examples.evaluate_read_context --write-bindings h_recongraph/examples/read_context_write_bindings.json --repeats 3 --output-json results/explicit-bindings.json
```

LongMemEval 读取与条件双锚点消融：

```powershell
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --dependency-ablation --output-json results/read-ablation.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --dependency-ablation --max-fact-anchors 2 --output-json results/top2-ablation.json
```

## 验证结果

- 第 13 批指定回归：149 passed；官方 `graphiti-core==0.29.2` 包下复核通过。
- Token 统一采用 `len(text) // 4` 估算；渲染预热后计时三次，汇总中位数。
- 整体召回：保留支持轮次 / 全部支持轮次；裁剪条件召回：保留支持轮次 / 命中全文中的支持轮次。
- Micro 按支持轮次汇总；Macro 对有相应支持标注的题目平均。第 12–13 批整体召回有效题数为 479，共 896 条支持轮次；裁剪条件召回有效题数为 475，命中全文含 880 条支持轮次。
- 第 11 批历史结果对全部 500 题平均，无支持标注题按 1 计入；该批的 75.64% 为此历史口径。第 12 批起主表采用整体 Micro。
- [阶段结果与附件索引](results/README.md)汇集逐题记录、题型统计、装配索引和补丁。
