# 17 阶段发布验证

## 核心改动

复核日期：2026-10-21。公开仓库代码同步至第 17 阶段，加入冻结配置的端到端 QA、更新误用与实际调用成本评测。第 06–13 批代码同步提交为 `043dcfc`；第 14、15、16 批分别为 `b1d0361`、`66f31c8`、`cf14c7c`。

## 代码变更

- 业务代码与四份回归测试取自逐代码审核工作副本；连续更新、机制样本和 LongMemEval 评测脚本及配套输入一并发布。
- 安装依赖声明为 `graphiti-core==0.29.2`；阶段摘要位于 `docs/stages/`，完整报告、结果和补丁由[附件索引](results/README.md)统一关联。
- 验证直接在发布目录执行，复用原项目 `.venv/Scripts/python.exe`，通过 `PYTHONPATH` 加载官方 Graphiti 0.29.2 wheel 解包源码。H-ReConGraph 与测试从发布目录加载。

## 验证结果

- pytest：**165 passed**，1 个已有 Pydantic Config 弃用提示。
- 发布范围的 28 个 Python 文件：Ruff check、`ruff format --check` 通过。
- 同步的业务、评测、测试和样本文件与审核工作副本逐一核对一致。
- `git diff --check` 通过。
- 发布文档完成文风检查，阶段报告统一为“核心改动、代码变更、验证结果”；相对链接目标与全部 JSON 解析核对通过。
- 第 17 批代码补丁反向应用检查通过；QA 三组共享 Session，Top-1 与 Top-2 各 500 题的上下文估算 Token 逐题复现第 16 批。
- 按需定位与段落方案的写入证据摘要、装配和支持集合相同；各组句子锚点原文坐标与 Turn 归属逐项核对通过。
- 16 个机制样本六组自然绑定在 `w=1` 保留 35/35 必要片段，完整与探针 16/16；完整显式绑定在 `w=0` 达到相同结果。
- 第 17 批六题试跑完成 18 次回答、6 次评分及 1 次更新标注；全量 500 题准备 1,500 份上下文，保存 578 份回答和 78 份更新标注。后续调用因指定凭据返回 HTTP 401 而中断，完整批次评分保留待续跑状态。

回归命令：

```powershell
python -m pytest tests/test_h_recongraph_memory.py tests/test_h_recongraph_graphiti_store.py tests/test_h_recongraph_retrieval.py tests/test_h_recongraph_longmemeval.py -q -p no:cacheprovider
```
