import os
import subprocess
import argparse
import sys
import time

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
PYTHON_EXE = sys.executable

# 核心离线环境配置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def run_script(folder, script_name, args_list):
    """ 执行分层目录下的脚本 """
    script_path = os.path.join(BASE_DIR, "src_script", folder, script_name)
    cmd = [PYTHON_EXE, script_path] + args_list
    print(f"\n[HIERARCHY RUN] {folder}/{script_name}: {' '.join(cmd)}")
    subprocess.run(cmd)

def find_best_s1():
    if not os.path.exists(MODEL_DIR): return None
    files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("DebertaV3MTL_S1_Sample") and f.endswith(".pth")])
    return os.path.join(MODEL_DIR, files[-1]) if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "ablation", "viz"], default="all")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3, help="验证集 Loss 早停等待轮数")
    args = parser.parse_args()

    # 预先创建结果子目录 (防御式编程)
    for d in ["models", "logs", "eval", "viz"]:
        os.makedirs(os.path.join(RES_DIR, d), exist_ok=True)

    common = ["--sample_size", str(args.sample_size), "--seed", str(args.seed), "--scheduler", args.scheduler, "--patience", str(args.patience), "--early_patience", str(args.early_patience)]

    # --- Phase 1: Data & Models Training ---
    if args.mode in ["all", "train"]:
        # 数据预处理: 生成 train_processed / val_processed / test_processed (80/10/10)
        # 优化：直接在预处理阶段进行采样，避免生成庞大的全量中间文件
        run_script("data", "exp_data_preprocess.py", ["--sample_size", str(args.sample_size), "--seed", str(args.seed)])

        # Group 1: Classical Strong Baseline (TF-IDF + LR)
        run_script("train", "train_classical_tfidf_lr.py", ["--mode", "train"])

        # Group 2: Hybrid Contrastive Baseline (Strong Contrast)
        run_script("train", "train_bert_cnn_bilstm.py", common)

        # Group 3: Pretrained Transformer Baselines
        run_script("train", "train_vanilla_bert.py", common)
        run_script("train", "train_vanilla_roberta.py", common)
        run_script("train", "train_vanilla_deberta_v3.py", common)

        # Group 4: Final Scheme (Proposed MTL Model)
        print("\n>>> 训练本文提出方案 (Stage 1 & 2)")
        run_script("train", "train_deberta_v3_mtl_s1.py", common)
        s1_path = find_best_s1()
        if s1_path: 
            run_script("train", "train_deberta_v3_mtl_s2.py", ["--s1_checkpoint", s1_path] + common)

    # --- Phase 2: Ablation (Switch Mode) ---
    if args.mode in ["all", "ablation"]:
        print("\n>>> 启动消融实验组 (Ablation Matrix)")
        # Ablation-1: Pooling
        run_script("train", "train_deberta_v3_mtl_s1.py", ["--no_pooling"] + common)
        # Ablation-2: MTL
        run_script("train", "train_deberta_v3_mtl_s1.py", ["--only_toxicity"] + common)
        # Ablation-3: Reweight
        s1_path = find_best_s1()
        if s1_path: 
            run_script("train", "train_deberta_v3_mtl_s2.py", ["--s1_checkpoint", s1_path, "--no_reweight"] + common)

    # --- Phase 3: Evaluation ---
    if args.mode in ["all", "eval", "ablation"]:
        print("\n>>> 全自动化评估引擎启动...")
        if os.path.exists(MODEL_DIR):
            pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
            for pth in sorted(pths):
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
