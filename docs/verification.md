# 15 阶段发布验证

## 核心改动

复核日期：2026-10-21。公开仓库代码同步至第 15 阶段，包含三层记忆、四维证据绑定、记忆束局部更新、读取装配及句子锚点与词形消融。第 06–13 批代码同步提交为 `043dcfc`；第 14 批提交为 `b1d0361`。

## 代码变更

- 业务代码与四份回归测试取自逐代码审核工作副本；连续更新、机制样本和 LongMemEval 评测脚本及配套输入一并发布。
- 安装依赖声明为 `graphiti-core==0.29.2`；阶段摘要位于 `docs/stages/`，完整报告、结果和补丁由[附件索引](results/README.md)统一关联。
- 验证直接在发布目录执行，复用原项目 `.venv/Scripts/python.exe`，通过 `PYTHONPATH` 加载官方 Graphiti 0.29.2 wheel 解包源码。H-ReConGraph 与测试从发布目录加载。

## 验证结果

- pytest：**157 passed**，1 个已有 Pydantic Config 弃用提示。
- 发布范围的 28 个 Python 文件：Ruff check、`ruff format --check` 通过。
- 同步的业务、评测、测试和样本文件与审核工作副本逐一核对一致。
- `git diff --check` 通过。
- 发布文档完成文风检查，阶段报告统一为“核心改动、代码变更、验证结果”；相对链接目标与全部 JSON 解析核对通过。
- 第 15 批代码补丁反向应用检查通过；原子串与词界作用域的原粒度配置精确复现第 14 批成本与召回，六组 Session / Turn 相同。
- 2,793 个句子锚点的原文偏移、Turn 归属与图事实核对通过；16 个机制样本六组事实段落一致。

回归命令：

```powershell
python -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py tests/test_h_recongraph_longmemeval.py -q -p no:cacheprovider
```
