# 训练运行指南 (exp/supplement-ablation 分支)

## 一、修改清单 (全部修改项)

### 第一轮修改 (反过拟合参数优化)

| # | 文件 | 修改项 | 旧值 | 新值 | 原因 |
|---|------|--------|------|------|------|
| 1 | model_deberta_v3_mtl.py | dropout | 0.1 | **0.2** | 增强正则化 |
| 2 | train_deberta_v3_mtl_s1.py | Focal alpha | 12.5 | **2.0** | 数据已平衡 |
| 3 | train_deberta_v3_mtl_s1.py | S1 lr | 2e-5 | **1e-5** | 防止过拟合 |
| 4 | train_deberta_v3_mtl_s1.py | S1 scheduler | plateau | **linear** | warmup避免lr尖峰 |
| 5 | train_deberta_v3_mtl_s1.py | S1 accum_steps | 1 | **2** | 梯度更平滑 |
| 6 | train_deberta_v3_mtl_s1.py | S1 weight_decay | 0 | **0.01** | L2正则化 |
| 7 | train_deberta_v3_mtl_s2.py | S2 lr | 5e-6 | **3e-6** | 微调阶段更小lr |
| 8 | train_deberta_v3_mtl_s2.py | S2 scheduler | plateau | **linear** | warmup(10%) |
| 9 | train_deberta_v3_mtl_s2.py | S2 grad_accum | 2 | **4** | 更大等效batch |
| 10 | train_deberta_v3_mtl_s2.py | S2 weight_decay | 0 | **0.01** | L2正则化 |
| 11 | train_deberta_v3_mtl_s2.py | 身份权重 | 统一2.5 | **分组差异化** | 稀有群体更高权重 |
| 12 | exp_data_preprocess.py | 采样策略 | 随机50:50 30万 | **保留全部有标签+填充1:1** | 最大化标签覆盖 |
| 13 | run_supplement_experiments.py | 运行器参数 | plateau/grad2 | **linear/grad4+预处理** | 同步新参数 |

### 第二轮修改 (架构+训练流程全面优化)

| # | 文件 | 修改项 | 旧行为 | 新行为 | 优先级 | 原因 |
|---|------|--------|--------|--------|--------|------|
| 14 | train_deberta_v3_mtl_s2.py | **[BUG] scheduler步数** | num_steps未除grad_accum | **÷ grad_accum** | P0 | warmup和衰减完全错误 |
| 15 | train_deberta_v3_mtl_s1.py | **S1 开启 AMP** | FP32 (AMP被注释掉) | **AMP FP16** | P0 | S1比S2慢40%+, 精度不一致 |
| 16 | train_deberta_v3_mtl_s1.py | **Focal alpha** | 2.0 | **1.0** | P0 | 1:1数据两类应等权 |
| 17 | S1 + S2 | **删 SyncBatchNorm** | convert_sync_batchnorm | **删除** | P0 | DeBERTa无BN层，纯多余 |
| 18 | S2 | **Layer-wise lr decay** | 所有层同lr | **底层0.64x/中层0.8x/顶层1x** | P1 | 防止灾难遗忘 |
| 19 | S1 + S2 | **AUC-based checkpoint** | val_loss选模型 | **val_AUC选模型** | P1 | loss和AUC不总是单调相关 |
| 20 | model | **Uncertainty Weighting** | 固定alpha=0.1/beta=0.2 | **自动学习任务权重** | P1 | 不确定性高的任务自动降权 |
| 21 | S1 | **评估包含多任务loss** | 评估只看tox loss | **tox+sub+id完整loss** | P1 | 与训练目标一致 |
| 22 | exp_data_preprocess.py | **Soft label** | y_tox二值化(0/1) | **保留原始连续值[0,1]** | P1 | 保留标注不确定性信息 |
| 23 | model | **分离特征头** | 三任务共享projection | **主任务/辅助任务独立proj** | P2 | 避免梯度冲突 |
| 24 | model | **AttentionPooling** | 单层Linear | **Tanh bottleneck非线性** | P2 | 提升表达能力 |
| 25 | S1 + S2 | **num_workers** | 0 | **4 + persistent_workers** | P2 | 数据加载不再是瓶颈 |
| 26 | exp_data_loader.py | **删除在线增强** | safe_aug随机交换词对 | **完全移除** | P2 | 任何文本扰动都可能引入标签噪声 |
| 27 | S2 | **group_weights预创建** | 每步新建tensor | **全局缓存tensor** | P3 | 减少CUDA内存碎片 |
| 28 | S2 | **S1→S2 权重兼容** | strict=True | **strict=False + 提示新参数** | P2 | 新增proj_tox/proj_aux自动处理 |

### 第三轮修改 (彻底移除数据增强)

| # | 文件 | 修改项 | 旧行为 | 新行为 | 原因 |
|---|------|--------|--------|--------|------|
| 29 | exp_data_preprocess.py | **删除 DataAugmenter 类** | 150行离线增强(SR/RI/RD/BT) | **完全移除** | 同义词替换/删词破坏毒性语义引入标签噪声; 回译慢且质量差; 只增强toxic会打破1:1平衡 |
| 30 | exp_data_preprocess.py | **删除 --no_aug 参数** | do_augment 控制增强开关 | **移除参数** | 不再需要开关 |
| 31 | exp_data_preprocess.py | **删除冗余依赖** | import nltk/tqdm/torch/transformers | **仅保留 pandas/numpy/sklearn/os/argparse** | 脚本从360行精简至145行 |
| 32 | exp_data_loader.py | **删除 safe_aug 方法** | 在线随机交换词对 | **完全移除** | DeBERTa dropout=0.2 已提供足够正则化 |
| 33 | S1 + S2 训练脚本 | **删除 --no_aug flag** | augment=not args.no_aug | **不传 augment 参数** | 增强已从 DataLoader 移除 |
| 34 | run_supplement_experiments.py | **删除 --no_aug** | 预处理命令含 --no_aug | **移除该参数** | 预处理不再有增强功能 |

---

## 二、核心改进原理详解

### 2.1 Uncertainty Weighting (Kendall et al., 2018)

```
旧: loss = l_tox + 0.1 * l_sub + 0.2 * l_id   (固定权重)
新: loss = l_tox/(2*exp(σ_tox)) + σ_tox/2 + l_sub/(2*exp(σ_sub)) + σ_sub/2 + ...
```

σ (log_var) 是可学习参数。不确定性高(loss大)的任务，σ自动增大，权重自动降低。
训练时会打印实时的任务权重变化。

### 2.2 Layer-wise Learning Rate Decay (S2)

```
Embeddings + 前6层:  lr × 0.64  (冻结底层语义知识)
后6层:               lr × 0.80  (微调上层特征)
Heads + Projection:  lr × 1.00  (完全训练任务层)
```

目的: S2是在S1基础上微调，全参数同lr会导致灾难遗忘。底层保持稳定，只调顶层。

### 2.3 分离特征头 (Gradient Conflict Resolution)

```
旧: h → proj → z → tox_head / subtype_head / identity_head (共享z)
新: h → proj_tox → z_tox → tox_head     (主任务独立)
    h → proj_aux → z_aux → subtype_head / identity_head  (辅助任务共享)
```

identity_head 需要识别身份词，tox_head 需要忽略身份词。共享z会让两个头互相拉扯。

### 2.4 Soft Label

```
旧: target=0.49 → y_tox=0 (和target=0.01一样)
新: target=0.49 → y_tox=0.49 (保留不确定性)
```

BCE loss 天然支持连续标签。评估时仍然用 ≥0.5 二值化计算 F1/ACC。

---

## 三、运行方式

### 方式1: 一键运行全部实验 (推荐)

```bash
python run_supplement_experiments.py --mode all --no_bar 2>&1 | tee supplement_experiment.log
```

### 方式2: 分阶段运行

```bash
# Step 1: 数据预处理
python run_supplement_experiments.py --mode preprocess --no_bar

# Step 2: 多seed实验
python run_supplement_experiments.py --mode multi_seed --no_bar

# Step 3: 消融实验
python run_supplement_experiments.py --mode ablation --no_bar

# Step 4: 评估 + 汇总
python run_supplement_experiments.py --mode eval_only --no_bar
python run_supplement_experiments.py --mode aggregate
```

### 方式3: 手动单次训练

```bash
# S1 (8卡 DDP + AMP)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 \
    src_script/train/train_deberta_v3_mtl_s1.py \
    --seed 42 --data_seed 42 --sample_size 300000 --epochs 6 --no_bar

# S2 (8卡 DDP + AMP + layer-wise lr)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 \
    src_script/train/train_deberta_v3_mtl_s2.py \
    --s1_checkpoint src_result/models/DebertaV3MTL_S1_Seed42_*.pth \
    --seed 42 --data_seed 42 --sample_size 300000 --epochs 4 --no_bar
```

---

## 四、身份分组差异化权重

| 身份组 | 样本数(1.8M) | 权重 |
|--------|-------------|------|
| female | ~103K | 1.0 |
| male | ~78K | 1.2 |
| christian | ~62K | 1.3 |
| white | ~33K | 1.8 |
| muslim | ~29K | 1.9 |
| black | ~17K | 2.3 |
| homosexual | ~12K | 2.6 |
| jewish | ~9K | 2.9 |
| psychiatric | ~5K | 3.4 |

---

## 五、数据备份

- 旧数据备份: `data/backup_seed42_300k/`
- 备份说明: `DATA_BACKUP_README.txt`
- 恢复: `cp data/backup_seed42_300k/*.parquet data/`

---

## 六、预期改善

| 指标 | 机制 |
|------|------|
| 过拟合 | dropout↑ + weight_decay + lr↓ + warmup + layer-wise decay |
| AUC | AUC-based checkpoint + soft label + 分离特征头 |
| Worst Group AUC | 分组差异化权重 + 全标签覆盖 + uncertainty weighting |
| 训练速度 | S1 AMP提速40%+ + num_workers=4 + 删SyncBatchNorm |
| 统计可信度 | 多seed (42, 123, 2024) × 3次实验 |
| 消融完整性 | NoPooling / NoFocal / NoReweight 细粒度消融 |

---

## 七、检查清单

- [ ] `data/train.csv` 存在 (原始 1.8M)
- [ ] `data/backup_seed42_300k/` 备份完整
- [ ] 8 张 GPU 可用: `nvidia-smi`
- [ ] 预训练模型已下载: `ls pretrained_models/hub/`
- [ ] 运行: `python run_supplement_experiments.py --mode all --no_bar`
- [ ] 检查: `ls src_result/eval/*_metrics.json`
- [ ] 汇总: `python aggregate_results.py`
