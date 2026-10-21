# H-ReConGraph

基于 Graphiti 的三层图记忆研究原型。本仓库的可安装代码快照对应 **05 阶段**；后续 **06–13 阶段**的研究迭代与实验进展汇总在本 README 中，逐阶段实现记录保存在逐代码审核工作副本。

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

06–13 阶段加入 `evidence_bundle` 读取：从命中事实出发，展开绑定的限定、路径和因果证据，按来源轮次窗口裁剪文本；来源正文已包含图事实时复用正文，减少重复渲染。LongMemEval 配对实验比较支持证据保留与上下文 Token 成本。

## 阶段进展

| 阶段 | 主要机制 | 回归或实验结果 | 阶段记录 |
|---|---|---|---|
| 原版 | M0/M1/M2 写入、规则语义状态、Graphiti 检索 | 8 passed / 1 TTL 测试 failed | [00 原版起点](docs/stages/00-baseline.md) |
| 01 | 入图绑定与按命中结果关联状态 | 26 passed | [01 读取关联](docs/stages/01-retrieval-linking.md) |
| 02 | 关系来源原文回溯 | 40 passed | [02 来源回溯](docs/stages/02-source-episodes.md) |
| 03 | 召回回退、访问计数、来源状态恢复 | 98 passed | [03 召回与恢复](docs/stages/03-recall-and-restore.md) |
| 04 | 读取证据文本组装 | 101 passed | [04 文本组装](docs/stages/04-context-text.md) |
| 05 | M0 有效词面召回、稳定排序 | 107 passed | [05 相关性与排序](docs/stages/05-stable-retrieval.md) |
| 06 | 证据片段组装与来源裁剪 | 证据束机制样本配对实验 | 证据片段组装 |
| 07 | 忠实性与读取成本对照 | 证据保留与上下文成本配对实验 | 忠实性与读取成本 |
| 08 | 记忆束局部更新与改写裁剪 | 局部变更与来源锚点配对实验 | 局部更新 |
| 09 | 连续更新与增量记忆束 | 新增、补充、局部修改、替代、删除与读取 | 连续更新 |
| 10 | 自然对话四维证据绑定 | 自然文本自动绑定与读取裁剪对照 | 自然证据绑定 |
| 11 | LongMemEval 读取成本配对 | Oracle 500 题，按轮次比较全文与证据束 | LongMemEval 配对 |
| 12 | 证据召回、窗口与成本 Pareto 消融 | Oracle 500 题，比较 `w=0/1/2` | 召回成本消融 |
| 13 | 重复渲染修复、依赖作用域消融 | Oracle 500 题，比较去重、窗口与依赖筛选 | 重复渲染与依赖消融 |

01–05 阶段说明和 patch 位于本仓库 `docs/`。06–13 阶段的逐文件说明、代码差异和实验 JSON 保存在逐代码审核工作副本。

## 第 13 阶段读取结果

固定检索结果，对 LongMemEval Oracle 500 题比较去重全文和证据束配置。479 题包含支持轮次标注，共 896 条；Token 估算采用 `len(text) // 4`。

| 配置 | 平均 Token | 支持轮次保留 | 整体证据召回 |
|---|---:|---:|---:|
| 去重全文 | 6,703.39 | 880 / 896 | 98.21% |
| 证据束 `w=1` | 6,606.28 | 873 / 896 | 97.43% |
| 证据束 `w=2` | 6,693.30 | 877 / 896 | 97.88% |

`w=1` 相较历史重复渲染配置平均减少 543.01 Token（7.60%），保留 873 条支持轮次；相较去重全文减少 1.45%。完整逐题记录、题型统计和 Pareto 分析见第 13 阶段逐文件交付记录。

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

本仓库 05 阶段指定回归测试：**107 passed**。后续代码迭代第 13 阶段指定回归测试：**149 passed**。

```powershell
python -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py -q -p no:cacheprovider
```

Graphiti 引擎代码基于 [getzep/graphiti](https://github.com/getzep/graphiti)，版本为 `0.29.2`。许可证为 Apache-2.0。
