# 05 版发布验证

## 核心改动

发布时间：2026-10-20。发布副本包含当前逐代码审核版的 H-ReConGraph Python 源码、三份指定测试、原版至 05 版的阶段说明与七份代码补丁。

## 代码变更

- 业务与测试代码直接取自 05 版工作副本。
- 发布配置声明 `graphiti-core==0.29.2`，保留原有 Ruff 规则及模块导入分组。
- 发布副本测试复用已有 Python 环境，并通过 PYTHONPATH 加载审核副本中的 Graphiti 0.29.2 源码；H-ReConGraph 与测试从发布目录加载。
- 七份补丁对应第一批三份逐文件差异及第二至第五批的四份批次差异。

## 验证结果

- pytest：**107 passed**，1 个已有 Pydantic Config 弃用提示。
- 四个改动业务文件与三份测试：Ruff check、format --check 通过。
- 发布文件与源工作副本的 Python 文件逐一 SHA256 对比一致。
- Git 差异空白检查通过。
