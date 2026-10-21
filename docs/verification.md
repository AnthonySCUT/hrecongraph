# 13 阶段发布验证

## 核心改动

复核日期：2026-10-21。公开仓库代码同步至第 13 阶段，包含三层记忆、四维证据绑定、记忆束局部更新、读取装配及配对实验脚本。代码同步提交：`043dcfc`。

## 代码变更

- 业务代码与四份回归测试取自逐代码审核工作副本；连续更新、机制样本和 LongMemEval 评测脚本及配套输入一并发布。
- 安装依赖声明为 `graphiti-core==0.29.2`；阶段详情位于 `docs/stages/`，01–05 历史补丁位于 `docs/patches/`。
- 验证直接在发布目录执行，复用原项目 `.venv/Scripts/python.exe`，通过 `PYTHONPATH` 加载 Graphiti 0.29.2 源码。H-ReConGraph 与测试从发布目录加载。

## 验证结果

- pytest：**149 passed**，1 个已有 Pydantic Config 弃用提示。
- 改动范围的 13 个 Python 文件：Ruff check、`ruff format --check` 通过。
- 同步的业务、评测、测试和样本文件与审核工作副本逐一核对一致。
- `git diff --check` 通过。
- 发布文档与本地阶段稿共 32 份完成文风检查，阶段报告统一为“核心改动、代码变更、验证结果”；相对链接目标核对通过。

回归命令：

```powershell
python -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py tests/test_h_recongraph_longmemeval.py -q -p no:cacheprovider
```
