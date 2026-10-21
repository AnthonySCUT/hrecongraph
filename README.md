# H-ReConGraph

基于 Graphiti 的三层图记忆研究原型。本仓库代码已同步至 **17 阶段**，包含记忆束更新、四维证据绑定、多片段来源保留、证据裁剪、按需句子定位及冻结配置的端到端 QA 评测。阶段机制与执行进度见下方阶段页。

## 研究框架

- **M0 瞬时记忆**：暂存当前交互，以相关词面或证据 ID 参与召回。
- **M1 短期记忆**：保存当前会话和任务相关记忆，支持记忆束局部更新。
- **M2 长期记忆**：通过巩固形成，保留版本和来源关联。
- **四维证据状态**：围绕事实、约束、路径、因果组织语义信息，并绑定原文证据。

总体流程：

```text
写入：对话 -> M0 -> M1 升格 -> 四维证据绑定 -> 记忆束局部更新 -> M2 巩固
读取：query -> 三层路由 -> M0 相关性召回 + Graphiti 关系检索
            -> 来源回溯 -> 语义状态关联 -> 上下文装配
```

`evidence_bundle` 读取从命中事实出发，展开绑定的限定、路径和因果证据，按来源轮次窗口裁剪文本；来源正文已包含图事实时复用正文，减少重复渲染。配对实验分别研究记忆更新、证据保留与上下文成本，并在固定检索下比较绑定规则和事实粒度。

## 阶段进展

| 阶段 | 主要机制 | 回归或实验结果 | 阶段记录 |
|---|---|---|---|
| 原版 | M0/M1/M2 写入、规则语义状态、Graphiti 检索 | 回归：8 passed；TTL 测试：1 failed | [00 原版起点](docs/stages/00-baseline.md) |
| 01 | 入图绑定与按命中结果关联状态 | 回归：26 passed | [01 读取关联](docs/stages/01-retrieval-linking.md) |
| 02 | 关系来源原文回溯 | 回归：40 passed | [02 来源回溯](docs/stages/02-source-episodes.md) |
| 03 | 召回回退、访问计数、来源状态恢复 | 回归：98 passed | [03 召回与恢复](docs/stages/03-recall-and-restore.md) |
| 04 | 读取证据文本组装 | 回归：101 passed | [04 文本组装](docs/stages/04-context-text.md) |
| 05 | M0 有效词面召回、稳定排序 | 回归：107 passed | [05 相关性与排序](docs/stages/05-stable-retrieval.md) |
| 06 | 证据片段组装与来源裁剪 | 回归：111 passed；证据束机制用例 | [06 证据片段组装](docs/stages/06-evidence-bundle.md) |
| 07 | 忠实性与读取成本对照 | 回归：113 passed；16 样本保留 88.57%，减字 36.45% | [07 忠实性与成本](docs/stages/07-faithfulness-cost.md) |
| 08 | 记忆束局部更新与改写裁剪 | 回归：121 passed；完整绑定证据保留 100%，减字 43.00% | [08 局部更新](docs/stages/08-local-updates.md) |
| 09 | 连续更新与增量记忆束 | 回归：129 passed；changed、kept、四维证据均 100% | [09 连续更新](docs/stages/09-continuous-updates.md) |
| 10 | 自然对话四维证据绑定 | 回归：133 passed；自然锚定、恢复与同源去重通过 | [10 自然证据绑定](docs/stages/10-natural-evidence.md) |
| 11 | LongMemEval 读取成本配对 | 回归：137 passed；Token 减少 38.46%，历史按题平均召回 75.64% | [11 LongMemEval 配对](docs/stages/11-longmemeval-pairing.md) |
| 12 | 证据召回、窗口与成本 Pareto 消融 | 回归：142 passed；Oracle 500 题，`w=1` 召回 97.43% | [12 召回成本消融](docs/stages/12-recall-cost-pareto.md) |
| 13 | 重复渲染修复、依赖作用域消融 | 回归：149 passed；Oracle 500 题，去重 `w=1` 召回 97.43% | [13 重复渲染与依赖消融](docs/stages/13-rendering-dependency-ablation.md) |
| 14 | 写入词界与事实作用域校准 | 回归：153 passed；Oracle 500 题，作用域 `w=1` 召回 93.75%，较原证据束省 Token 10.99% | [14 写入绑定校准](docs/stages/14-write-binding-calibration.md) |
| 15 | 句子锚点与词形因子消融 | 回归：157 passed；Oracle 500 题，词形原粒度 `w=1` 召回 95.20%、5,986.68 Token | [15 句子锚点与词形](docs/stages/15-sentence-anchor-ablation.md) |
| 16 | 多片段证据、按需句子定位与条件双锚点 | 回归：161 passed；Oracle 500 题，候选 `w=1` 召回 97.10%、6,245.99 Token | [16 多片段与按需定位](docs/stages/16-multispan-on-demand-anchors.md) |
| 17 | 冻结配置、真实 QA、更新误用与实际成本 | 回归：165 passed；六题试跑完成；全量已存 578 份回答，接口 401 待续跑 | [17 端到端 QA](docs/stages/17-frozen-end-to-end-qa.md) |

01–17 阶段说明均可从本仓库 `docs/stages/` 访问。06–17 批完整报告、逐题结果和补丁见[实验附件索引](docs/results/README.md)，运行方法与评价口径见[复现说明](docs/reproduction.md)。第 17 批已发布 QA 实现、试跑结果与全量续跑记录，完整评分等待指定接口凭据恢复。

## 代表性结果

选取证据绑定、连续更新、去重读取、规则与粒度消融四类实验，展示研究主线。各项结果对应表中样本与设置。

| 研究重点 | 代表阶段与设置 | 主要结果 | 详细记录 |
|---|---|---|---|
| 四维证据绑定 | 08：16 个机制样本，完整显式绑定 | 必要证据保留 100%，读取字符减少 43.00% | [记忆束与改写裁剪](docs/stages/08-local-updates.md) |
| 连续记忆更新 | 09：新增、补充、局部修改、替代、删除与读取回放 | 更新正确率 36.36% → 100%；无关记忆保持 100%；四维证据完整率 17.65% → 100% | [连续更新配对实验](docs/stages/09-continuous-updates.md) |
| 读取去重与成本 | 13：Oracle 500 题，原子串证据束 `w=1` | 支持召回 97.43%，平均 6,606.28 Token；较重复渲染配置减少 7.60% Token | [去重与依赖消融](docs/stages/13-rendering-dependency-ablation.md) |
| 证据保留与锚点成本 | 16：Oracle 500 题，词形作用域，`w=1` | 第 15 批首匹配：95.20%、5,986.68 Token；完整触发 + 按需 Top-2：97.10%、6,245.99 Token | [多片段与按需定位](docs/stages/16-multispan-on-demand-anchors.md) |

LongMemEval 表中召回采用整体 Micro：保留支持轮次 / 全部标注支持轮次；479 题共 896 条支持。Token 采用 `len(text) // 4` 估算，平均值覆盖 500 题。完整分组、配置与逐题结果见[实验附件索引](docs/results/README.md)。

当前默认采用原子串规则、原粒度与完整触发收集；`first` 保留历史首匹配对照，词形、句子作用域和条件 Top-2 作为可选消融配置。按需句子定位保留段落依赖索引。完整阶段演进见“阶段进展”表，各批次详情保存在对应阶段页。

## 安装与调用

```powershell
python -m pip install -e ".[dev]"
```

```python
from h_recongraph import HReConGraphMemory

memory = HReConGraphMemory(graphiti_client)
await memory.add_interaction(
    '项目接口因为联调而延期',
    user_id='research_user',
    session_id='session_1',
)
context = await memory.retrieve(
    '项目接口延期原因',
    user_id='research_user',
    session_id='session_1',
)
print(context.context_text)
```

## 验证

当前代码快照（17 阶段）指定回归：**165 passed**。环境、检查范围与复核结果见[发布验证](docs/verification.md)。

```powershell
python -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py tests/test_h_recongraph_longmemeval.py -q -p no:cacheprovider
```

Graphiti 引擎代码基于 [getzep/graphiti](https://github.com/getzep/graphiti)，版本为 `0.29.2`。许可证为 Apache-2.0。
