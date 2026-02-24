================================================================================
                    !! 重要 !! 数据备份说明 !! 重要 !!
================================================================================

备份时间: 2025-02-24
备份人: 实验自动化脚本

================================================================================
1. 备份了什么？
================================================================================

备份目录: data/backup_seed42_300k/

  train_seed42_300k.parquet   -- 训练集 (约346,401条，含增强)
  val_seed42_300k.parquet     -- 验证集 (约30,000条)
  test_seed42_300k.parquet    -- 测试集 (约30,000条)

这三个文件是 data/ 下对应 *_processed.parquet 的完整副本。

================================================================================
2. 数据是怎么来的？
================================================================================

原始数据:   data/train.csv (Jigsaw Unintended Bias in Toxicity Classification)
采样参数:   sample_size=300,000  seed=42
处理脚本:   src_script/data/exp_data_preprocess.py
处理流程:
  1) 从 train.csv 读取全量数据 (~1.8M 条)
  2) 50:50 平衡采样 (有毒:正常)，共 300,000 条
  3) 对少数类进行 EDA 数据增强 (同义词替换/随机插入/随机删除/回译)
  4) 80/10/10 划分为 train/val/test
  5) 保存为 parquet 格式

================================================================================
3. 为什么要备份？
================================================================================

所有已完成的实验 (main 分支上的 6 个模型) 都基于这份数据训练和评估。
补充实验 (exp/supplement-ablation 分支) 必须使用完全相同的数据，
以保证多 seed 实验只改变模型初始化随机性，不改变数据子集。

!! 绝对不要重新运行 exp_data_preprocess.py !!
!! 绝对不要删除或覆盖 data/*_processed.parquet !!

如果不小心删除了，可以从 data/backup_seed42_300k/ 恢复:
  cp data/backup_seed42_300k/train_seed42_300k.parquet data/train_processed.parquet
  cp data/backup_seed42_300k/val_seed42_300k.parquet   data/val_processed.parquet
  cp data/backup_seed42_300k/test_seed42_300k.parquet   data/test_processed.parquet

================================================================================
4. 补充实验如何保证数据一致？
================================================================================

训练脚本新增了 --data_seed 参数 (默认值=42):
  - --data_seed 控制 sample_aligned_data() 的采样种子 (始终=42)
  - --seed 控制模型初始化、Dropout、DDP Sampler 等训练随机性

这样 seed=123 或 seed=2024 的实验用的是和 seed=42 完全相同的数据子集，
只有模型训练过程不同。

================================================================================
5. 已有模型检查点 (main分支, seed=42)
================================================================================

以下模型是基于此数据训练的，结果已记录在 EXPERIMENT_REPORT.md:

  src_result/models/VanillaBERT_Sample300000_0224_1246.pth
  src_result/models/VanillaRoBERTa_Sample300000_0224_1246.pth
  src_result/models/BertCNNBiLSTM_Sample300000_0224_1246.pth
  src_result/models/VanillaDeBERTa_Sample300000_0224_1246.pth
  src_result/models/DebertaV3MTL_S1_Sample300000_0224_1632.pth
  src_result/models/DebertaV3MTL_S2_Sample300000_0224_2117.pth
  src_result/models/DebertaV3MTL_S1_AblationBCE_Sample300000_0224_1652.pth
  src_result/models/DebertaV3MTL_S2_AblationBCE_Sample300000_0224_2117.pth
  src_result/models/lr_model.joblib
  src_result/models/tfidf_vectorizer.joblib

================================================================================
