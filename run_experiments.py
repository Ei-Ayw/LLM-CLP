import os
import subprocess
import argparse
import sys
import time
from datetime import datetime

# =============================================================================
# ### 实验管理器：run_experiments.py (专业分层架构版) ###
# =============================================================================
# 设计说明：
# 本脚本作为整个项目的"控制塔"，通过四级分层子目录调度全量实验流程。
# 文件夹结构：data/, train/, eval/, viz/
# 数据划分策略:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  原始数据 (train.csv) ──> exp_data_preprocess.py                        │
# │                                                                         │
# │  ┌───────────────────┐   ┌──────────────┐   ┌──────────────┐           │
# │  │   Train (80%)     │   │  Val (10%)   │   │  Test (10%)  │           │
# │  │ train_processed   │   │ val_processed │   │ test_processed│          │
# │  │   .parquet        │   │   .parquet    │   │   .parquet   │           │
# │  └───────────────────┘   └──────────────┘   └──────────────┘           │
# │        ↓                       ↓                    ↓                   │
# │   模型训练用              训练过程验证/         最终评估指标            │
# │  (sample_size控制)        Early Stopping        (从未见过)             │
# └─────────────────────────────────────────────────────────────────────────┘
#
# 实验矩阵 (精简到文档要求):
#   Group 1: 传统强基线    - TF-IDF + Logistic Regression
#   Group 2: 混合强对照    - BERT + CNN + BiLSTM
#   Group 3: Transformer   - BERT, RoBERTa, DeBERTa-v3 (本文基座)
#   Group 4: 本文方案      - DeBERTa-v3 MTL (两阶段) + 消融实验
#
# 输出目录结构 (src_result/):
# ├── models/      # 存储 .pth 模型权重 (Best Model)
# ├── logs/        # 存储训练过程的 Loss 数据与曲线
# ├── eval/        # 存储评估报告与阈值曲线
# └── viz/         # 存储最终的可视化图表 (t-SNE等)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "src_result")
MODEL_DIR = os.path.join(RES_DIR, "models")
PYTHON_EXE = sys.executable + " -u"

# 核心离线环境配置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# [Stable Mode] 强制禁用 NCCL P2P 和 IB 以防止 DataParallel 段错误
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def run_script(folder, script_name, args_list):
    """ 执行分层目录下的脚本 """
    script_path = os.path.join(BASE_DIR, "src_script", folder, script_name)
    cmd = PYTHON_EXE.split() + [script_path] + args_list
    print(f"\n[HIERARCHY RUN] {folder}/{script_name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def find_latest_checkpoint(identifier=None):
    if not os.path.exists(MODEL_DIR): return None
    pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if identifier:
        pths = [f for f in pths if identifier in f]
    
    if not pths: return None
    # Sort by modification time (newest first)
    pths.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
    return os.path.join(MODEL_DIR, pths[0])

def find_best_s1():
    # Deprecated fallback
    return find_latest_checkpoint(identifier="S1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "ablation", "viz"], default="all")
    parser.add_argument("--sample_size", type=int, default=0, help="实验数据采样量 (0表示全量)")
    # 针对 4x A10 (24GB) 优化：显存充足，使用大 Batch 提高吞吐
    # MaxLen=128 下，单卡 24G 可支持 Batch=128+，4卡可支持 512+
    parser.add_argument("--batch_size", type=int, default=1280, help="默认基础 Batch Size (长期稳定: 1280 for 4x24GB)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=128, help="短序列加速，默认 128")
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3, help="验证集 Loss 早停等待轮数")
    parser.add_argument("--epochs", type=int, default=0, help="训练轮数 (0=默认/早停)")
    parser.add_argument("--no_bar", action="store_true", help="禁用 tqdm 进度条 (nohup模式下推荐)")
    args = parser.parse_args()
    
    # 记录实验开始时间戳，用于后续筛选本次实验产出的模型
    experiment_start_time = datetime.now()
    print(f"[{experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Experiment Session Started.")

    # 处理全量跑逻辑
    effective_sample_size = args.sample_size if args.sample_size > 0 else 10_000_000
    is_full_mode = args.sample_size <= 0
    print(f"[{'FULL' if is_full_mode else 'SAMPLED'} MODE] Effective Sample Size: {effective_sample_size}")

    # 自动检测 nohup / 非交互环境
    if not sys.stdout.isatty():
        args.no_bar = True
        print(">>> [Auto-Detect] Non-interactive shell detected. Disabling progress bars.")

    # 预先创建结果子目录 (防御式编程)
    for d in ["models", "logs", "eval", "viz"]:
        os.makedirs(os.path.join(RES_DIR, d), exist_ok=True)

    common = ["--sample_size", str(effective_sample_size), "--batch_size", str(args.batch_size), "--seed", str(args.seed), "--max_len", str(args.max_len), "--scheduler", args.scheduler, "--patience", str(args.patience), "--early_patience", str(args.early_patience)]
    
    if args.epochs > 0:
        common += ["--epochs", str(args.epochs)]
        
    if args.no_bar:
        common += ["--no_bar"]

    # 针对 DeBERTaV3 (参数量较大/Attention机制不同) 进行显存特别优化
    # [FP32 Safe Mode] Batch Size 降级至 64 (16 per GPU) 以确保绝对稳定
    deberta_batch_size = min(args.batch_size, 64)
    deberta_common = common[:]
    if "--batch_size" in deberta_common:
        idx = deberta_common.index("--batch_size")
        deberta_common[idx+1] = str(deberta_batch_size)


    # --- Phase 1: Data & Models Training ---
    if args.mode in ["all", "train"]:
        # 数据预处理: 生成 train_processed / val_processed / test_processed (80/10/10)
        # 优化：直接在预处理阶段进行采样，避免生成庞大的全量中间文件
        # run_script("data", "exp_data_preprocess.py", ["--sample_size", str(effective_sample_size), "--seed", str(args.seed)])

        # Group 1: Classical Strong Baseline (TF-IDF + LR)
        run_script("train", "train_classical_tfidf_lr.py", ["--mode", "train"])

        # Group 2: Hybrid Contrastive Baseline (Strong Contrast)
        run_script("train", "train_bert_cnn_bilstm.py", common)

        # Group 3: Pretrained Transformer Baselines
        run_script("train", "train_vanilla_bert.py", common)
        run_script("train", "train_vanilla_roberta.py", common)
        run_script("train", "train_vanilla_deberta_v3.py", deberta_common)

        print("\n>>> 训练本文提出方案 (Stage 1 & 2)")
        run_script("train", "train_deberta_v3_mtl_s1.py", deberta_common)
        s1_path = find_best_s1()
        if s1_path: 
            run_script("train", "train_deberta_v3_mtl_s2.py", ["--s1_checkpoint", s1_path] + deberta_common)

    # --- Phase 2: Ablation (Switch Mode) ---
    if args.mode in ["all", "ablation"]:
        print("\n>>> 启动严格消融实验组 (Strict Ablation Matrix: Full S1+S2 Pipeline)")
        
        # 基础命令集 (使用 DeBERTa 专用配置，因为消融实验全都是基于 DeBERTa 的)
        base_cmd = deberta_common

        # Define Ablation Cases
        # Format: (CaseName, S1_Extra_Args, S2_Extra_Args)
        ablations = [
            ("No_Augmentation", ["--no_aug"], ["--no_aug"]),           # 1. 移除数据增强
            ("No_Pooling", ["--no_pooling"], ["--no_pooling"]),        # 2. 移除 Attention Pooling
            ("No_Reweight", [], ["--no_reweight"]),                    # 3. 移除 S2 身份重加权 (S1 正常, S2 无重加权)
            ("No_Focal", ["--no_focal"], ["--no_focal"]),             # 4. 移除 Focal Loss (如果脚本支持，需确保脚本也加了此参数解析)
        ]

        for case_name, s1_args, s2_args in ablations:
            print(f"\n[Ablation Run] Case: {case_name}")
            
            # 1. Run Stage 1 for this ablation
            # 为了区分，我们需要给模型存盘名加后缀，或者依赖脚本内部的时间戳机制
            # 这里我们通过传入 --suffix 参数让脚本保存时带上标记 (需要在训练脚本里支持)
            # 或者我们简单地顺序跑，然后根据时间戳最新的去找。
            
            # Run S1
            print(f"  > Running Stage 1 ({case_name})...")
            run_script("train", "train_deberta_v3_mtl_s1.py", base_cmd + s1_args + ["--ablation_tag", case_name])
            
            # Find the S1 checkpoint we just trained
            # 假设脚本保存的文件名包含 case_name
            s1_ckpt = find_latest_checkpoint(identifier=f"_{case_name}_")
            
            if s1_ckpt:
                print(f"  > Found S1 Checkpoint: {s1_ckpt}")
                print(f"  > Running Stage 2 ({case_name})...")
                # Run S2 using the specific S1 checkpoint
                run_script("train", "train_deberta_v3_mtl_s2.py", base_cmd + s2_args + ["--s1_checkpoint", s1_ckpt, "--ablation_tag", case_name])
            else:
                print(f"  [Error] S1 training for {case_name} failed or checkpoint not found. Skipping S2.")

    # --- Phase 3: Evaluation ---
    if args.mode in ["all", "eval", "ablation"]:
        print("\n>>> 全自动化评估引擎启动...")
        if os.path.exists(MODEL_DIR):
            pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
            
            # --- 筛选逻辑优化：只评估本次实验新生成的模型 ---
            # 通过文件修改时间 (mtime) 与 experiment_start_time 对比
            new_pths = []
            for p in pths:
                full_p = os.path.join(MODEL_DIR, p)
                mtime = datetime.fromtimestamp(os.path.getmtime(full_p))
                if mtime >= experiment_start_time:
                    new_pths.append(p)
            
            pths = new_pths
            print(f"  > Detected {len(pths)} new checkpoints from this session.")

            # 智能筛选：每个模型配置只评估最新的权重
            checkpoint_groups = {}
            for pth in sorted(pths):
                # pattern: ModelName_SampleXXXX_MMDD_HHMM.pth
                # Split by '_' and remove the last two parts (date, time) to get the group key
                parts = pth.split('_')
                if len(parts) >= 3:
                    group_key = "_".join(parts[:-2])
                    checkpoint_groups[group_key] = pth
            
            print(f">>> 发现 {len(pths)} 个权重文件，筛选出 {len(checkpoint_groups)} 个最新模型进行评估。")

            for group, pth in checkpoint_groups.items():
                # 过滤策略: 跳过 S1 中间权重 (除非是单任务 OnlyTox，它没有 S2)
                if "_S1_" in pth and "OnlyTox" not in pth:
                    print(f"  [Skip] 跳过中间阶段权重 S1 (Intermediate Checkpoint): {pth}")
                    continue

                # 分类映射逻辑
                m_type = "deberta_mtl" if "DebertaV3MTL" in pth else \
                         "bert_cnn" if "BertCNN" in pth else \
                         "text_cnn" if "TextCNN" in pth else \
                         "bilstm" if "BiLSTM" in pth else \
                         "vanilla_bert" if "VanillaBERT" in pth else \
                         "vanilla_roberta" if "VanillaRoBERTa" in pth else \
                         "vanilla_deberta" if "VanillaDeBERTa" in pth else "vanilla"
                
                run_script("eval", "eval_universal_runner.py", ["--checkpoint", os.path.join(MODEL_DIR, pth), "--model_type", m_type, "--output_prefix", pth.replace(".pth", "")])

    # --- Phase 4: Viz ---
    if args.mode in ["all", "viz"]:
        run_script("viz", "viz_performance_summary.py", [])
        # 寻找 S2 正式权重进行可视化
        if os.path.exists(MODEL_DIR):
            files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("DebertaV3MTL_S2_Sample") and "No" not in f and f.endswith(".pth")])
            if files:
                run_script("viz", "viz_feature_t_sne.py", ["--checkpoint", os.path.join(MODEL_DIR, files[-1]), "--output_name", "viz_final_paper.png"])

    print("\n[FINISH] Professional Hierarchical Lifecycle Management Completed.")

if __name__ == "__main__":
    main()
