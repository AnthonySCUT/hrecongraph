# 实验结果与逐文件附件

## 核心改动

集中保存各阶段的原始结果、完整代码说明和批次差异。阶段结果对应当时的实验配置；当前脚本的运行方式见[复现说明](../reproduction.md)。

| 结果文件 | 记录 |
|---|---|
| [07-读取对照结果.json](07-读取对照结果.json) | 32 |
| [08-完整写入绑定读取对照结果.json](08-完整写入绑定读取对照结果.json) | 分组与逐题记录见文件 |
| [09-连续更新配对实验结果.json](09-连续更新配对实验结果.json) | 704 |
| [11-LongMemEval读取成本配对结果.json](11-LongMemEval读取成本配对结果.json) | 1000 |
| [11-四维自然自动绑定结果.json](11-四维自然自动绑定结果.json) | 分组与逐题记录见文件 |
| [12-LongMemEval证据召回成本Pareto消融.json](12-LongMemEval证据召回成本Pareto消融.json) | 2000 |
| [12-机制样本显式绑定召回成本消融.json](12-机制样本显式绑定召回成本消融.json) | 分组与逐题记录见文件 |
| [12-机制样本自然绑定召回成本消融.json](12-机制样本自然绑定召回成本消融.json) | 分组与逐题记录见文件 |
| [12补充-LongMemEval单锚点对照.json](12补充-LongMemEval单锚点对照.json) | 2000 |
| [12补充-LongMemEval条件双锚点对照.json](12补充-LongMemEval条件双锚点对照.json) | 2000 |
| [12补充-条件双锚点配对汇总.json](12补充-条件双锚点配对汇总.json) | 分组与逐题记录见文件 |
| [13-LongMemEval条件双锚点去重对照.json](13-LongMemEval条件双锚点去重对照.json) | 2000 |
| [13-LongMemEval重复渲染与依赖作用域消融.json](13-LongMemEval重复渲染与依赖作用域消融.json) | 5500 |
| [13-依赖作用域与去重配对汇总.json](13-依赖作用域与去重配对汇总.json) | 分组与逐题记录见文件 |
| [13-机制样本显式绑定依赖消融.json](13-机制样本显式绑定依赖消融.json) | 分组与逐题记录见文件 |
| [13-机制样本自然绑定依赖消融.json](13-机制样本自然绑定依赖消融.json) | 分组与逐题记录见文件 |
| [14-LongMemEval写入绑定校准.json](14-LongMemEval写入绑定校准.json) | 分组与逐题记录见文件 |
| [14-写入绑定校准汇总.json](14-写入绑定校准汇总.json) | 分组与逐题记录见文件 |
| [14-机制样本写入绑定校准.json](14-机制样本写入绑定校准.json) | 分组与逐题记录见文件 |
| [15-LongMemEval-fact_scope-paragraph.json](15-LongMemEval-fact_scope-paragraph.json) | 2000 |
| [15-LongMemEval-fact_scope-sentence.json](15-LongMemEval-fact_scope-sentence.json) | 2000 |
| [15-LongMemEval-substring-paragraph.json](15-LongMemEval-substring-paragraph.json) | 2000 |
| [15-LongMemEval-substring-sentence.json](15-LongMemEval-substring-sentence.json) | 2000 |
| [15-LongMemEval-word_form_scope-paragraph.json](15-LongMemEval-word_form_scope-paragraph.json) | 2000 |
| [15-LongMemEval-word_form_scope-sentence.json](15-LongMemEval-word_form_scope-sentence.json) | 2000 |
| [15-LongMemEval句子锚点消融.json](15-LongMemEval句子锚点消融.json) | 分组与逐题记录见文件 |
| [15-句子锚点与词形消融汇总.json](15-句子锚点与词形消融汇总.json) | 分组与逐题记录见文件 |
| [15-机制样本句子锚点消融.json](15-机制样本句子锚点消融.json) | 分组与逐题记录见文件 |

## 代码变更

完整逐文件报告：

- [06-证据片段组装逐文件交付.md](../details/06-证据片段组装逐文件交付.md)
- [07-忠实性与读取成本对照逐文件交付.md](../details/07-忠实性与读取成本对照逐文件交付.md)
- [08-记忆束局部更新与改写裁剪逐文件交付.md](../details/08-记忆束局部更新与改写裁剪逐文件交付.md)
- [09-连续更新与增量记忆束逐文件交付.md](../details/09-连续更新与增量记忆束逐文件交付.md)
- [10-自然对话证据绑定与读取去重逐文件交付.md](../details/10-自然对话证据绑定与读取去重逐文件交付.md)
- [11-LongMemEval读取成本配对逐文件交付.md](../details/11-LongMemEval读取成本配对逐文件交付.md)
- [12-证据裁剪召回成本消融与Pareto探索逐文件交付.md](../details/12-证据裁剪召回成本消融与Pareto探索逐文件交付.md)
- [12补充-条件双锚点对照说明.md](../details/12补充-条件双锚点对照说明.md)
- [13-重复渲染修复与依赖作用域消融逐文件交付.md](../details/13-重复渲染修复与依赖作用域消融逐文件交付.md)
- [14-写入绑定校准逐文件交付.md](../details/14-写入绑定校准逐文件交付.md)
- [15-句子锚点与词形消融逐文件交付.md](../details/15-句子锚点与词形消融逐文件交付.md)

已有批次补丁：

- [01_graphiti_store.patch](../patches/01_graphiti_store.patch)
- [02-来源原文回溯.patch](../patches/02-来源原文回溯.patch)
- [02_manager.patch](../patches/02_manager.patch)
- [03-召回回退与状态恢复.patch](../patches/03-召回回退与状态恢复.patch)
- [03_test_h_recongraph_memory.patch](../patches/03_test_h_recongraph_memory.patch)
- [04-读取证据文本组装.patch](../patches/04-读取证据文本组装.patch)
- [05-M0相关性与稳定排序.patch](../patches/05-M0相关性与稳定排序.patch)
- [06-证据片段组装.patch](../patches/06-证据片段组装.patch)
- [07-忠实性与读取成本对照.patch](../patches/07-忠实性与读取成本对照.patch)
- [08-记忆束局部更新与改写裁剪.patch](../patches/08-记忆束局部更新与改写裁剪.patch)
- [09-连续更新与增量记忆束.patch](../patches/09-连续更新与增量记忆束.patch)
- [12-证据裁剪召回成本消融与Pareto探索.patch](../patches/12-证据裁剪召回成本消融与Pareto探索.patch)
- [12补充-条件双锚点对照.patch](../patches/12补充-条件双锚点对照.patch)
- [13-重复渲染修复与依赖作用域消融.patch](../patches/13-重复渲染修复与依赖作用域消融.patch)
- [14-写入绑定校准.patch](../patches/14-写入绑定校准.patch)
- [15-句子锚点与词形消融.patch](../patches/15-句子锚点与词形消融.patch)

## 验证结果

28 份 JSON 解析通过；11 份完整报告及 16 份补丁均有对应文件。指标定义与实验数据 SHA256 见[复现说明](../reproduction.md)。
