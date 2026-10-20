# 00 原版起点

## 核心改动

原版以 Graphiti 为底座，建立 M0 瞬时原文、M1 短期记忆和 M2 长期记忆。交互经规则打分升格，语义状态记录时间、因果、意图、条件和变化；读取端根据问题选择层级。

本轮审核从原版备份的完整副本开始：558 个文件、23,930,740 字节，初始逐文件 SHA256 相同。

## 代码变更

原版主要环节：

- `manager.py`：串联 add_interaction、retrieve、consolidate、maintain。
- `graphiti_store.py`：把记忆序列化为 episode，并调用 Graphiti 搜索。
- `ranker.py`：按词面重合和记忆分数组合排序。
- `relinking.py`：组织查询上下文及原有临时关联边。
- `evolution.py`、`versioning.py`：三层升格、巩固和版本演化。

原版 retrieve 在选取本地语义状态时，将全部本地 M1/M2 ID 加入相关集合；图检索结果与本地状态关联因此成为第一批修复重点。

## 验证结果

原版 9 项测试：8 passed、1 failed。失败来自 1 微秒 TTL 测试对执行时机的依赖，第一批通过回拨测试对象创建时间使测试确定化。
