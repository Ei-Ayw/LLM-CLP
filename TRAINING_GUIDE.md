# 训练运行指南 (exp/supplement-ablation 分支)

## 一、本次修改总览

### 修改目的
针对 S1/S2 过拟合问题进行全面参数优化，同时引入新的数据采样策略和身份分组差异化权重。

### 修改文件与具体变更

| # | 文件 | 修改项 | 旧值 | 新值 | 修改原因 |
|---|------|--------|------|------|----------|
| 1 | `src_model/model_deberta_v3_mtl.py:43` | dropout | 0.1 | **0.2** | 增强正则化，抑制过拟合 |
| 2 | `src_script/train/train_deberta_v3_mtl_s1.py:47,97` | Focal Loss alpha | 12.5 | **2.0** | 数据已1:1平衡，无需大幅补偿 |
| 3 | `src_script/train/train_deberta_v3_mtl_s1.py:122` | S1 学习率 | 2e-5 | **1e-5** | 降低学习率防止过拟合 |
| 4 | `src_script/train/train_deberta_v3_mtl_s1.py:123` | S1 调度器 | plateau | **linear** | warmup+线性衰减避免初始lr尖峰 |
| 5 | `src_script/train/train_deberta_v3_mtl_s1.py:129` | S1 梯度累积 | 1 | **2** | 等效batch翻倍，梯度更平滑 |
| 6 | `src_script/train/train_deberta_v3_mtl_s1.py:214` | S1 weight_decay | 0 | **0.01** | L2正则化 |
| 7 | `src_script/train/train_deberta_v3_mtl_s2.py:113` | S2 学习率 | 5e-6 | **3e-6** | 微调阶段更小学习率 |
| 8 | `src_script/train/train_deberta_v3_mtl_s2.py:114` | S2 调度器 | plateau | **linear** | warmup(10%)+线性衰减 |
| 9 | `src_script/train/train_deberta_v3_mtl_s2.py:123` | S2 梯度累积 | 2 | **4** | 等效batch更大，训练更稳定 |
| 10 | `src_script/train/train_deberta_v3_mtl_s2.py:211` | S2 weight_decay | 0 | **0.01** | L2正则化 |
| 11 | `src_script/train/train_deberta_v3_mtl_s2.py:42-60` | 身份权重 | 统一 w=2.5 | **分组差异化** | 稀有群体获更高权重 |
| 12 | `src_script/data/exp_data_preprocess.py:166-215` | 数据采样策略 | 随机50:50 30万 | **保留全部有标签+填充1:1** | 最大化身份标签覆盖率 |
| 13 | `run_supplement_experiments.py` | 实验运行器 | plateau/grad2 | **linear/grad4+预处理阶段** | 同步新参数 |

### 身份分组差异化权重详情

基于对数平滑逆频率公式: `w = 1 + log(max_count / count)`

| 身份组 | 样本数(原始1.8M) | 权重 |
|--------|-----------------|------|
| female | ~103K | 1.0 |
| male | ~78K | 1.2 |
| christian | ~62K | 1.3 |
| white | ~33K | 1.8 |
| muslim | ~29K | 1.9 |
| black | ~17K | 2.3 |
| homosexual_gay_or_lesbian | ~12K | 2.6 |
| jewish | ~9K | 2.9 |
| psychiatric_or_mental_illness | ~5K | 3.4 |

**加权逻辑**: 每个样本取所属身份组中最大的权重值（一个样本可能同时属于多个身份组）。

### 新数据采样策略

```
旧策略: 从1.8M中随机采样30万 (50%有毒, 50%无毒)
新策略: 1. 保留所有有身份/子类别标签的样本 (~26.7万)
        2. 从无标签样本中补充至有毒:无毒 = 1:1
        3. 预计总量 ~28.9万
```

优势: 身份标签覆盖率从 ~15% 提升至 ~92%+

---

## 二、运行方式

### 方式1: 一键运行全部实验 (推荐)

```bash
# 8卡服务器，预计 3-4 小时
python run_supplement_experiments.py --mode all --no_bar 2>&1 | tee supplement_experiment.log
```

### 方式2: 分阶段运行

```bash
# Step 1: 数据预处理 (新采样策略)
python run_supplement_experiments.py --mode preprocess --no_bar

# Step 2: 多seed实验
python run_supplement_experiments.py --mode multi_seed --no_bar

# Step 3: 消融实验
python run_supplement_experiments.py --mode ablation --no_bar

# Step 4: 仅评估
python run_supplement_experiments.py --mode eval_only --no_bar

# Step 5: 汇总结果
python run_supplement_experiments.py --mode aggregate
```

### 方式3: 仅运行数据预处理

```bash
python src_script/data/exp_data_preprocess.py --seed 42 --no_aug --keep_all_labeled
```

### 方式4: 手动运行单次训练 (用于调试)

```bash
# S1 训练 (4卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    src_script/train/train_deberta_v3_mtl_s1.py \
    --seed 42 --data_seed 42 --sample_size 300000 \
    --epochs 6 --no_bar

# S2 训练 (4卡, 需要S1的checkpoint路径)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    src_script/train/train_deberta_v3_mtl_s2.py \
    --s1_checkpoint src_result/models/DebertaV3MTL_S1_Seed42_Sample300000_XXXX_XXXX.pth \
    --seed 42 --data_seed 42 --sample_size 300000 \
    --epochs 4 --no_bar
```

---

## 三、实验矩阵

| 实验 | 说明 | GPU分配 | 预计时间 |
|------|------|---------|----------|
| 预处理 | 新采样策略生成数据 | CPU | ~15 min |
| MTL S1 x2 seeds | seed=123, 2024 并行 | 4+4卡 | ~2h |
| MTL S2 x2 seeds | seed=123, 2024 并行 | 4+4卡 | ~2h |
| Vanilla x2 seeds | seed=123, 2024 并行 | 4+4卡 | ~1.5h |
| 消融 S1 (NoPooling, NoFocal) | 2个并行 | 4+4卡 | ~2h |
| 消融 S2 (NoPooling, NoFocal, NoReweight) | 分批并行 | 4+4卡 | ~2h |
| 批量评估 | 所有新模型 | 8卡并行 | ~20 min |
| **总计** | | | **~10h** |

---

## 四、数据备份说明

- 旧数据备份位置: `data/backup_seed42_300k/`
- 备份说明文件: `DATA_BACKUP_README.txt`
- 运行新预处理后，`data/train_processed.parquet` 等将被覆盖为新采样数据
- 如需恢复旧数据: `cp data/backup_seed42_300k/*.parquet data/`

---

## 五、预期改善

| 指标 | 改善方向 | 机制 |
|------|----------|------|
| 过拟合 | 显著缓解 | dropout↑ + weight_decay + lr↓ + warmup |
| Worst Group AUC | 提升 | 分组差异化权重 + 全标签覆盖 |
| 统计可信度 | 增强 | 多seed (42, 123, 2024) 三次实验 |
| 消融完整性 | 补全 | NoPooling/NoFocal/NoReweight 细粒度消融 |

---

## 六、检查清单

- [ ] 确认 `data/train.csv` 存在 (原始 1.8M 数据)
- [ ] 确认 `data/backup_seed42_300k/` 备份完整
- [ ] 确认 8 张 GPU 可用: `nvidia-smi`
- [ ] 确认预训练模型已下载: `ls pretrained_models/hub/`
- [ ] 运行: `python run_supplement_experiments.py --mode all --no_bar`
- [ ] 完成后检查: `ls src_result/eval/*_metrics.json`
- [ ] 汇总结果: `python aggregate_results.py`
