# H-ReConGraph Lightweight

基于 Graphiti 的三层图记忆科研原型。本仓库对应从原版备份逐文件审核至 **05 版**的代码快照，主要完善按问题读取、来源回溯、语义状态恢复与稳定排序。

## 核心改动

- **M0**：保存当前原文，以有效词面或证据 ID 召回。
- **M1 / M2**：依据命中关系、来源 episode 和证据关联返回语义状态。
- **来源回溯**：批量读取命中关系引用的原文，恢复相应记忆与图绑定。
- **上下文组装**：通过 `context.context_text` 返回 M0、语义状态、关系和来源正文。
- **排序与访问**：查询得分独立计算，实际返回记忆按次更新访问计数。

读取流程：

```text
query
  -> 三层路由
  -> M0 相关性召回 + Graphiti 关系检索
  -> 命中关系的来源回溯
  -> 本地语义状态关联 / 词面回退 / 来源恢复
  -> 上下文文本 + 访问计数
```

## 代码变更

```text
h_recongraph/
  config/              三层记忆策略
  memory/              控制器、Graphiti 适配、排序、上下文与演化
  schemas/             记忆单元、语义状态及关系表示
  examples/            原有调用示例
tests/                 本轮指定的三份测试
docs/stages/           原版至 05 版的逐阶段说明
docs/patches/          各批次代码差异
```

Graphiti 核心通过 `graphiti-core==0.29.2` 依赖提供。仓库保留 H-ReConGraph 模块的内部依赖，使当前读取与三层演化逻辑可以独立阅读。

| 阶段 | 主要机制 | 阶段测试结果 | 说明 |
|---|---|---|---|
| 原版 | M0/M1/M2 写入、规则语义状态、Graphiti 检索 | 8 passed / 1 TTL 测试 failed | [00 原版起点](docs/stages/00-baseline.md) |
| 01 | 入图绑定与按命中结果关联状态 | 26 passed | [01 读取关联](docs/stages/01-retrieval-linking.md) |
| 02 | 关系来源原文回溯 | 40 passed | [02 来源回溯](docs/stages/02-source-episodes.md) |
| 03 | 召回回退、访问计数、来源状态恢复 | 98 passed | [03 召回与恢复](docs/stages/03-recall-and-restore.md) |
| 04 | 读取证据文本组装 | 101 passed | [04 文本组装](docs/stages/04-context-text.md) |
| 05 | M0 有效词面召回、稳定排序 | 107 passed | [05 相关性与排序](docs/stages/05-stable-retrieval.md) |

表中各阶段数量来自对应阶段的本地回归记录；当前发布快照对应 05 版。

安装：

```powershell
python -m pip install -e ".[dev]"
```

调用：

```python
from h_recongraph import HReConGraphMemory

memory = HReConGraphMemory(graphiti_client)
await memory.add_interaction(
    "项目接口因为联调而延期",
    user_id="research_user",
    session_id="session_1",
)
context = await memory.retrieve(
    "项目接口延期原因",
    user_id="research_user",
    session_id="session_1",
)
print(context.context_text)
```

## 验证结果

05 版指定测试：**107 passed**。发布副本复核结果见 [发布验证](docs/verification.md)。

```powershell
python -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py -q -p no:cacheprovider
ruff check h_recongraph/memory/manager.py h_recongraph/memory/graphiti_store.py h_recongraph/memory/ranker.py h_recongraph/memory/relinking.py tests
ruff format --check h_recongraph/memory/manager.py h_recongraph/memory/graphiti_store.py h_recongraph/memory/ranker.py h_recongraph/memory/relinking.py tests
```

Graphiti 项目：[getzep/graphiti](https://github.com/getzep/graphiti)。许可证随仓库保留为 Apache-2.0。
