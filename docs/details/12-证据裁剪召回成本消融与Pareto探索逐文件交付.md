# 第十二批：证据裁剪的召回—成本消融与 Pareto 探索

## 核心改动

研究问题：在固定检索结果下，分别度量四维依赖绑定与邻近窗口增加的必要证据，以及相应的上下文成本。

装配流程：**事实锚点 → 已绑定的事实/限定/路径/因果依赖 → 同一 Session 内邻近窗口 → 索引并集 → 原序输出**。比较 `full_source`、`evidence_bundle(w=0/1/2)`。业务默认保持 `full_source`，原段落输入的 `w=0` 延续既有选择逻辑。

- LongMemEval 直接保存原始消息列表，以 `(Session Index, Turn Index)` 标识证据。消息内部空行与换行均保留在所属轮次；正文打印 `[TURN i]` 保留位置。机制样本使用原段落编号。
- 同源 M1/M2 副本共享来源标识后去重；不同原始轮次中的重复陈述分别保留。
- 每题构建一次 `HReConGraphMemory` 写入状态、基础图边及来源上下文，四组配置共享这些对象。输出固定检索记录、图事实轮次、绑定字段计数与绑定摘要。
- 词面实验适配保持一致：选择与问题有正词面相关性的全部 Session，每个 Session 以最大词面相关轮次形成一条基础图关系；并列时选最早轮次。自然绑定沿用 `SemanticStateConstructor`。
- 检索与装配读取问题、来源文本和已有绑定；`answer`、`has_answer`、机制样本必要证据片段用于装配后的评分。
- 原有全文回退单独记录为 `fallback_added`，其保留证据单列统计。主要实验的自然绑定与 LongMemEval 均为零回退贡献。

评价口径：令第 i 题全部支持证据为 S_i，命中来源所含支持证据为 F_i，最终保留为 R_i，则整体召回为 |R_i|/|S_i|，裁剪条件召回为 |R_i|/|F_i|。分别报告逐题平均（macro）和汇总计数（micro）；零分母题从相应召回指标中排除并明确计数。完整保留率统计全部 S_i 均被保留的有效题比例。

上下文 Token 使用现有 `estimate_tokens(text)=len(text)//4`；所有配置在同一 Windows / Python 3.11.15 环境运行。每次渲染先预热一次，再计时三次，报告逐题中位数的中位数。字符与 Token 包含 `[STATE]`、`[EDGE]`、`[SOURCE]`、轮次标签等实际输出文本；阶段增量成本另按来源正文统计。

## 代码变更

1. `h_recongraph/memory/relinking.py`
   - `render_text(..., window_size=0)` 接入读取窗口。
   - `source_units` 保存原始消息边界；`source_fact_indices` 保存命中轮次；`source_origin_ids` 关联同源副本。
   - `source_indices()` 返回事实、依赖、窗口及全文回退索引；`_source_bundle()` 装配这些索引对应的完整消息或段落。

```python
base_indexes = fact_indexes | dependency_indexes
window_indexes = {
    neighbor
    for index in base_indexes
    for neighbor in range(
        max(0, index - window_size),
        min(len(paragraphs), index + window_size + 1),
    )
}
# 求并集后，按来源顺序渲染；同一索引只输出一次。
```

2. `h_recongraph/examples/evaluate_read_context.py`
   - 对机制样本运行四组配置，覆盖自然自动绑定与显式完整绑定。
   - 共用 `assembly_stages()`、`summarize_rows()` 和 `pareto_analysis()`，统一阶段、召回分母、Token 和 Pareto 口径。
   - 阶段索引记录 `fact`、`fact_plus_dependencies`、`dependencies_plus_window` 及互斥增量；`final` 同时记录旧全文回退。

3. `h_recongraph/examples/evaluate_longmemeval_read.py`
   - 按原始消息轮次建立固定上下文；四组渲染完成后，再读取官方支持轮次标注评分。
   - 输出 500 题 × 4 配置逐题记录、六类题型分组、阶段支持证据索引、缺失原因及距离、双召回口径下的 Pareto 前沿与 90%/95% 参考线。
   - `rendered_source_indices` 表示该配置实际输出的位置；`assembly_indexes` 表示对应证据束的逐阶段装配。

4. `tests/test_h_recongraph_retrieval.py` 与 `tests/test_h_recongraph_longmemeval.py`
   - 新增四个核心测试函数，首尾窗口参数化为两例：`w=0` 选择一致、首尾截断、重叠窗口并集与原序、消息内换行/重复轮次/同源副本去重。
   - 保留并补齐显式绑定 16/16 完整证据与 16/16 探针断言。
   - 现有 LongMemEval 测试增加四组配置、索引评分及去掉答案/支持标签后仍可构建相同上下文的检查。

产物：

- `12-LongMemEval证据召回成本Pareto消融.json`：500 题、2,000 条逐配置结果、固定检索记录、分组和 Pareto。
- `12-机制样本自然绑定召回成本消融.json`：16 个无外部标注样本、64 条结果。
- `12-机制样本显式绑定召回成本消融.json`：原始绑定与完整显式绑定各 64 条结果。
- `12-证据裁剪召回成本消融与Pareto探索.patch`：相对第十一批修订提交 `a163879` 的五个代码/测试文件差异。

复现命令（使用既有 Python 环境，三次计时）：

```powershell
$py = 'C:\Users\AIR\Desktop\01_Folders\graphiti-main\graphiti-main\.venv\Scripts\python.exe'
& $py -m h_recongraph.examples.evaluate_read_context --natural-bindings --repeats 3 --output-json '逐文件审核记录/12-机制样本自然绑定召回成本消融.json'
& $py -m h_recongraph.examples.evaluate_read_context --write-bindings h_recongraph/examples/read_context_write_bindings.json --repeats 3 --output-json '逐文件审核记录/12-机制样本显式绑定召回成本消融.json'
& $py -m h_recongraph.examples.evaluate_longmemeval_read --repeats 3 --output-json '逐文件审核记录/12-LongMemEval证据召回成本Pareto消融.json'
```

## 验证结果

**16 个机制样本：自然自动绑定**

| 配置 | 平均字符 | 平均 Token | 汇总召回 | 完整样本 | 探针正确 | Token 缩减 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_source | 308.69 | 76.69 | 35/35，100% | 16/16 | 16/16 | 0% |
| w=0 | 161.94 | 40.13 | 26/35，74.29% | 7/16 | 8/16 | 47.68% |
| w=1 | 238.38 | 59.25 | 35/35，100% | 16/16 | 16/16 | 22.74% |
| w=2 | 308.69 | 76.69 | 35/35，100% | 16/16 | 16/16 | 0% |

自然绑定的事实阶段保留 23/35，依赖新增 3 条，来源正文新增 33 Token；`w=1` 再新增 9 条，正文新增 305 Token。`w=2` 与 `w=1` 保留相同证据，增加 283 Token 的窗口正文。

**16 个机制样本：显式完整绑定**

| 配置 | 平均 Token | 汇总召回 | 完整样本/探针正确 | Token 缩减 |
| --- | ---: | ---: | ---: | ---: |
| full_source | 87.56 | 35/35 | 16/16；16/16 | 0% |
| w=0 | 56.50 | 35/35 | 16/16；16/16 | 35.47% |
| w=1 | 87.56 | 35/35 | 16/16；16/16 | 0% |
| w=2 | 87.56 | 35/35 | 16/16；16/16 | 0% |

显式绑定的依赖新增 12 条支持片段、120 个正文 Token，达到 35/35；窗口增加无关上下文。两种绑定条件分别以其自身 `full_source` 为成本分母。全部 16 题均有有效必要证据标注，两类召回分母一致。

**LongMemEval oracle 500 题**

有效整体召回题数为 **479/500**，共 **896 条**标注支持轮次；有效裁剪条件召回题数为 **475/500**，命中来源包含 **880 条**支持轮次。21 题无支持轮次标注；其余 4 题的支持来源全部未命中。成本、答案字符串可见率和渲染延迟覆盖全部 500 题。

| 配置 | 平均字符 | 平均 Token | Token 缩减 | 整体召回 micro / macro | 条件召回 micro / macro | 完整保留率 | 答案可见率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_source | 28,987.04 | 7,246.39 | 0% | 98.21% / 98.72% | 100% / 100% | 469/479，97.91% | 46.60% |
| w=0 | 26,910.65 | 6,727.29 | 7.16% | 75.45% / 77.20% | 76.82% / 78.14% | 301/479，62.84% | 44.60% |
| w=1 | 28,598.62 | 7,149.28 | 1.34% | 97.43% / 97.71% | 99.20% / 98.98% | 462/479，96.45% | 46.40% |
| w=2 | 28,946.64 | 7,236.29 | 0.14% | 97.88% / 98.30% | 99.66% / 99.58% | 466/479，97.29% | 46.40% |

渲染中位延迟：全文约 **0.02 ms**，三组证据束约 **1.3 ms**，逐题实测值保存在 JSON。答案字符串可见率是保留诊断，后续问答实验另行评价模型生成答案。

| 装配阶段 | 保留支持轮次 | 对上一阶段新增 | 来源正文总 Token | 平均每题新增正文 Token |
| --- | ---: | ---: | ---: | ---: |
| 仅事实锚点 | 410/896 | — | 283,548 | — |
| 加已绑定依赖 | 676/896 | +266，整体召回 +29.69 个百分点 | 3,033,556 | +5,500.02 |
| 再加 w=1 | 873/896 | +197，整体召回 +21.99 个百分点 | 3,244,544 | +421.98 |
| 再加 w=2 | 877/896 | 相对依赖 +201；相对 w=1 +4 | 3,288,048 | 相对依赖 +508.98；相对 w=1 +87.01 |

其中 `w=1` 相比 `w=0` 的实际完整上下文平均增加 **421.99 Token**；`w=2` 相比 `w=1` 再增加 **87.01 Token**。来源正文与完整上下文的微小舍入差来自逐题整数 Token 估算。

**题型分组：整体汇总召回**

| 题型 | 有效题/全部题 | 支持轮次 | full_source | w=0 | w=1 | w=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| knowledge-update | 72/78 | 144 | 100% | 78.47% | 100% | 100% |
| multi-session | 125/133 | 327 | 96.33% | 73.39% | 95.72% | 96.02% |
| single-session-assistant | 56/56 | 56 | 100% | 91.07% | 94.64% | 98.21% |
| single-session-preference | 30/30 | 44 | 100% | 52.27% | 97.73% | 100% |
| single-session-user | 64/70 | 66 | 100% | 81.82% | 100% | 100% |
| temporal-reasoning | 132/133 | 259 | 98.46% | 75.29% | 98.07% | 98.07% |

**题型分组：相对全文 Token 缩减**

| 题型 | w=0 | w=1 | w=2 |
| --- | ---: | ---: | ---: |
| knowledge-update | 8.01% | 1.67% | 0.01% |
| multi-session | 6.92% | 1.36% | 0.09% |
| single-session-assistant | 12.71% | 4.52% | 1.31% |
| single-session-preference | 8.52% | 2.47% | 0% |
| single-session-user | 6.97% | 0.47% | 0% |
| temporal-reasoning | 6.43% | 0.94% | 0.22% |

JSON 同时给出每题型的 macro、micro、条件召回、完整率、字符/Token、答案可见率、延迟与有效分母。

**增量收益与主要损失**

- `w=0` 有 204 条裁剪损失：距依赖束 1 轮的 197 条、2 轮的 4 条、3 轮的 2 条、4 轮的 1 条。`w=1` 和 `w=2` 分别补回前 197 条和前 201 条。
- `w=0` 裁剪损失较多的题型为 multi-session（75 条）、temporal-reasoning（60 条），single-session-preference 的比例最大（21/44 条）。
- `w=2` 剩余 3 条裁剪损失：multi-session 1 条（距依赖束 3 轮）、temporal-reasoning 1 条（3 轮）、single-session-assistant 1 条（4 轮）。
- 全部配置另有同样 16 条来源未命中损失：multi-session 12 条，temporal-reasoning 4 条。它们构成固定检索条件下的共同损失。
- 成本主要集中在依赖展开：命中来源共 10,768 轮，事实锚点 931 轮，依赖展开后 7,774 轮。现有规则把含限定/路径/原因线索的轮次整体纳入，窗口再扩展后接近全文。
- 数据中的 10,960 条原始消息有 5,513 条含内部空行。第十一批用空行拆段，本批按原始消息保持整轮，并保留轮次标签，因此本批 `w=0` 的 7.16% 与先前段落粒度的 38.46% 对应不同裁剪单元。本批四组内部统一使用原始轮次和同一成本公式。

**Pareto 与参考线**

| 数据/绑定条件 | 非支配配置 | 达到 90% 召回的最低成本 | 达到 95% 召回的最低成本 |
| --- | --- | --- | --- |
| 16 题自然绑定 | w=0、w=1 | w=1，59.25 Token/题 | w=1，59.25 Token/题 |
| 16 题完整显式绑定 | w=0 | w=0，56.50 Token/题 | w=0，56.50 Token/题 |
| LongMemEval 500 | full_source、w=0、w=1、w=2 | w=1，7,149.28 Token/题 | w=1，7,149.28 Token/题 |

表中结论在整体/条件召回与 macro/micro 四个口径下相同。四组 LongMemEval 配置呈成本—召回取舍；`w=1` 是达到预设参考线的最低成本探索配置，节省量为 1.34%。当前最清晰的研究结论是：邻轮补充显著改善证据召回，来源整轮依赖展开同时承担了主要成本。后续重点可放在依赖绑定的查询相关性和更细粒度证据锚定；确认实验在冻结配置后另行评测。

**代码验证**

指定三文件加现有 LongMemEval 测试：**142 passed**，保留一项原有 Pydantic 弃用提示。五个改动代码/测试文件的 Ruff check、format --check 通过；本批代码差异的 `git diff --check` 与补丁应用检查通过。结果 JSON 已核对 2,000 行、阶段集合包含关系、增量差集与召回分母。
