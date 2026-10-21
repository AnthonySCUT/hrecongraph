# 第八批：记忆束局部更新与改写裁剪

## 核心改动

以“事实 + 限定 + 路径 + 因果”的原文证据束作为写入、更新与读取单位。新增 `add_memory()` 和 `update_memory()`，支持 `modify`、`supersede`、`delete`；改写关系通过事实原文锚点定位段落，再合并其余维度依赖段落。

## 代码变更

- `manager.py`：对目标记忆束及其 M1/M2 关联执行局部操作。
- `memory_unit.py`、`semantic_state.py`：保存完整四维证据及其变更状态。
- `graphiti_store.py`：按记忆束来源维护 episode 与关系边。
- `relinking.py`：通过原文锚点裁剪改写后的事实关系。
- 测试覆盖新增、修改、替代、删除和来源恢复。

## 验证结果

16 个机制样本在完整写入绑定条件下，四维必要证据均 **4/4 完整保留**；证据束读取字符数减少 **43.00%**。4 个关系改写样本均通过原文锚点裁剪。

- 指定三文件回归：**121 passed**。
- Ruff check、format check 与 `git diff --check` 通过。
