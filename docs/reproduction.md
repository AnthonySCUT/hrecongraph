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

第 14 批写入绑定校准：

```powershell
python -m h_recongraph.examples.evaluate_read_context --binding-calibration --slot-mode first --repeats 3 --output-json results/binding-mechanism.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-calibration --slot-mode first --output-json results/binding-longmemeval.json
```

该入口比较 `substring`、`word_boundary`、`fact_scope` 三组写入规则，每组运行全文与 `w=0/1/2`。`fact_scope` 在来源文本上建立同段或有效实词重合的依赖归属，将原文事实段落偏移保存为 `SourceSpan.fact_starts`；读取以命中事实偏移选择相应依赖。JSON 中的 `fixed_retrieval_equal` 检查 Session 与事实锚点一致性，`comparison_w1` 记录直接支持损失、窗口覆盖变化和远端证据保留，`joint_pareto` 汇总 12 组配置。

第 15 批句子锚点与词形消融：

```powershell
python -m h_recongraph.examples.evaluate_read_context --anchor-calibration --slot-mode first --repeats 3 --output-json results/anchor-mechanism.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --anchor-calibration --slot-mode first --output-json results/anchor-longmemeval.json
```

该入口比较 `substring/fact_scope/word_form_scope` 与 `paragraph/sentence` 的六组组合。每组共用一次写入与检索结果，运行四种装配配置。句子仅在固定的 Top-1 原始轮次内选择；`fact_anchors` 记录原始 `turn_index`、`start/end` 与文本，`source_fact_spans` 驱动依赖展开。句子索引按中英文句末标点和换行建立，英文句末标点后要求空白或文本结束。全文、窗口和支持评分使用原始消息轮次。

公开的第 15 批结果按六组拆分保存，[结果总览](results/15-LongMemEval句子锚点消融.json)的 `result_files` 指向各组完整逐题文件。脚本输出保留六组完整记录，分组文件与脚本输出共享配置、数据 SHA256 和逐题记录。

第 16 批多片段与按需定位：

```powershell
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-rule fact_scope --slot-mode first --output-json results/first-fact-scope.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-rule word_form_scope --slot-mode first --output-json results/first-word-forms.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-rule fact_scope --slot-mode all --output-json results/all-fact-scope.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-rule word_form_scope --slot-mode all --output-json results/all-word-forms.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-rule word_form_scope --slot-mode all --fact-granularity sentence --output-json results/all-dense-sentence.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-rule word_form_scope --slot-mode all --sentence-locator --output-json results/all-on-demand.json
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --repeats 3 --binding-rule word_form_scope --slot-mode all --sentence-locator --max-fact-anchors 2 --output-json results/all-on-demand-top2.json
```

对应机制样本使用 `evaluate_read_context --natural-bindings`，添加上述 `binding-rule/slot-mode/fact-granularity/sentence-locator` 参数。完整显式绑定继续使用 `--write-bindings`。完整结果的配置映射见[第 16 批索引](results/16-连续证据与按需定位结果索引.json)。

`all` 将各次触发加入来源证据；摘要字段保持原有触发词优先级。按需定位仅切分已命中的原始轮次，定位结果用于简洁图事实与来源坐标，段落索引继续确定依赖。`sentence_locator_median_ms` 独立记录预热后三次定位的中位耗时，`render_median_ms` 继续记录上下文渲染耗时。

第 17 批冻结配置与端到端 QA：

```powershell
# 在当前终端配置 HRECONGRAPH_QA_API_KEY，下面的接口填写该 Key 对应的地址。
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --qa-dir results/qa17 --qa-prepare-only
python -m h_recongraph.examples.evaluate_longmemeval_read --data data/longmemeval_oracle.json --limit 500 --qa-dir results/qa17 --model glm-5.1 --base-url <API_BASE_URL> --concurrency 150
```

`--qa-dir` 固定运行去重全文、词形作用域 Top-1／`w=1`、条件 Top-2／`w=1`。每题只构造一次写入状态，保存原始消息边界和来源日期。数据与 Prompt 保存于 `manifest.json`；`contexts.jsonl` 保存在本地运行目录，公开结果记录其 SHA256。重复执行同一命令复用成功调用，补跑缺失回答与评分。

GLM-5.1 使用温度 0、关闭 thinking；回答上限 512 Token、更新标定 1,536 Token、评分 2,048 Token，单次等待上限 1,200 秒。三个匿名答案合并为一次评分调用，匿名顺序按题目 ID 轮换。更新标注在生成答案前完成，并核对新旧值原文引用；答案将旧值作为问题时点的有效值时计为过期信息误用。

`calls.jsonl` 保存每次调用的响应、实际 usage、延迟和状态；授权切换接口时，`execution_segments.jsonl` 记录地址、模型、并发与执行代码摘要，冻结的配置和 Prompt 沿用原清单。回答成本包括输入与输出 Token，更新标定和评分费用另行统计。`results.json` 记录逐题 QA、更新误用、题型汇总、Wilson 区间与配对正确率差异。

`update_annotation_reviews.json` 保存按问题时间范围复核的标注，本批将三道明确询问历史值的问题归为历史问答。54 道更新题的新旧原文引用核对通过，作为过期误用的分母。重新运行先核对评分完整性和标注变化，再复用回答补评；响应中的 JSON 围栏与 `stale_muse` 字段拼写统一解析，全部原始响应保留。

## 验证结果

- 第 17 批指定回归：165 passed；官方 `graphiti-core==0.29.2` 包下复核通过。
- 第 06–16 批上下文 Token 统一采用 `len(text) // 4` 估算；渲染预热后计时三次，汇总中位数。第 17 批回答与评价成本采用接口实际 usage，回答延迟采用请求墙钟耗时，包含服务排队与 SDK 重试。
- 整体召回：保留支持轮次 / 全部支持轮次；裁剪条件召回：保留支持轮次 / 命中全文中的支持轮次。
- Micro 按支持轮次汇总；Macro 对有相应支持标注的题目平均。第 12–16 批整体召回有效题数为 479，共 896 条支持轮次；裁剪条件召回有效题数为 475，命中全文含 880 条支持轮次。
- 第 11 批历史结果对全部 500 题平均，无支持标注题按 1 计入；该批的 75.64% 为此历史口径。第 12 批起主表采用整体 Micro。
- [阶段结果与附件索引](results/README.md)汇集逐题记录、题型统计、装配索引和补丁。
- 第 17 批完成全量 500 题、1,500 份回答和全部评分；三组 QA 为 80.2%、80.8%、78.8%，过期误用为 4/54、6/54、5/54。原凭据完成 578 份回答，授权备用凭据在同一接口与模型下完成剩余 922 份；最终续跑并发 150。执行核对见[第 17 批复核](results/17-QA执行复核.json)。
