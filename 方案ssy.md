基于 DeBERTaV3 的偏见感知毒性评论检测（完备对齐版）
本项目已完全对齐用户上传的所有技术文档图片，旨在通过 DeBERTaV3 和针对性的偏见缓解机制实现高鲁棒性的检测系统。

1. 核心模型架构与路线图
我们将按以下五个步骤逐步构建并训练模型：

基础骨干: DeBERTa-v3-base + toxicity head (二分类)。
池化优化: 引入 Attention Pooling，利用 $h = \text{concat}(h_{cls}, h_{att})$。
辅助任务 A: 增加 subtypes head (子类型分类，$\alpha=0.5$)。
辅助任务 B: 增加 identity head (身份属性预测，$\beta=0.2$)。
偏见缓解: 加入 Identity-Aware Reweight ($w=2.5$) 进行 Stage 2 精调。
池化实现细节 (Attention Pooling)
打分函数: $s = \text{Linear}(H) \in [B, L, 1]$。
Mask处理: $s_i = s_i + (1 - \text{mask}) \cdot (-1e9)$ (即 masked_fill 为极小值)。
权重聚合: $a = \text{softmax}(s)$，$h_{att} = \sum a_i h_i$。
2. 训练策略与损失加权
双阶段训练 (Stage 1 & 2)
Stage 1 (预热): 训练 2~3 epochs，权重 $w=1.0$ (或仅给 identity-non-toxic 设 1.5)，初步掌握语义。
Stage 2 (去偏): 开启全重加权机制，针对 has_identity=1 且 y_tox=0 的样本设 $w=2.5$，大幅度降低身份词误报。
损失计算
实现方式：采用 reduction='none' 计算逐样本损失，手动乘以权重向量 $w$ 后取 batch_mean，防止被默认 reduction 机制覆盖。
3. 结果评估指标体系
A. 主任务指标
F1 (主指标): 用于模型选择与阈值确定。
Accuracy & PR-AUC: 针对类别不平衡提供更敏感的性能反馈。
阈值策略: 在验证集上执行 0.05 ~ 0.95 的 F1 扫描，选取最佳阈值应用于测试集。
B. 偏见/群体鲁棒性指标 (Nuanced Metrics)
Subgroup AUC: 考察特定群体子集的区分度。
BPSN AUC: 背景正样本 + 子群负样本，衡量“子群无毒被误判”的风险。
BNSP AUC: 背景负样本 + 子群正样本，衡量“子群有毒被漏掉”的风险。
综合得分:
$MeanBiasAUC = \text{mean}(AUC^{sub} + AUC^{bpsn} + AUC^{bnsp})$。
$WorstBiasAUC = \min(AUC^{sub}, AUC^{bpsn}, AUC^{bnsp})$。
4. 文件命名与目录规范 (一眼看懂)
为了确保您能“一眼看清楚谁是谁”，我们将严格执行以下命名准则：

A. 目录层级
data/: 存放原始数据、清洗后的 .parquet 或 .csv 以及 Tokenizer 缓存。
src_model/: 存放模型类定义，不含训练逻辑。
src_script/: 存放具体的执行脚本（数据处理、训练、评估）。
src_result/: 存放所有产出物（模型权重、日志、可视化图表）。
B. 文件命名规则
类型	命名格式 (前缀 + 功能)	示例
数据处理	data_xxx.py	data_preprocess.py (清洗), data_loader.py (加载)
模型类	model_xxx.py	model_deberta_mtl.py (主模型), model_baselines.py (基线)
训练脚本	train_xxx.py	train_stage1.py (预热), train_stage2_reweight.py (去偏)
评估/推理	eval_xxx.py	eval_fairness.py (偏见评估), eval_threshold.py (阈值扫描)
可视化	viz_xxx.py	viz_loss_curves.py, viz_auc_tables.py
结果文件	res_xxx.log/pth	res_stage2_final.pth, res_ablation_pool.log
5. 验证计划
代码核查: 重点检查 Linear(d->1) 的打分器和 -1e9 的 Mask 填充；验证逐样本 loss 加权逻辑。
阈值搜索: 在验证集上执行 0.05 ~ 0.95 的 F1 扫描。
指标产出: 自动化计算并输出包含 MeanBiasAUC 和 WorstBiasAUC 的完整报告。