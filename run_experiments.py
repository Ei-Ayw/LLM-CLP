import os
import subprocess
import argparse
import sys
import time

# =============================================================================
# ### 实验管理器：run_experiments.py (混合驱动版) ###
# 设计说明：
# 本脚本遵循用户的最新指示：
# 1. 基准模型 (Baselines) 采用“一模型一物理文件”结构。
# 2. 消融实验 (Ablation Studies) 采用“开关式参数”设计。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src_script")
RES_DIR = os.path.join(BASE_DIR, "src_result")
PYTHON_EXE = sys.executable

# 核心离线环境配置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def run_script(script_name, args_list):
    cmd = [PYTHON_EXE, os.path.join(SRC_DIR, script_name)] + args_list
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd)

def find_latest(prefix):
    # 查找特定实验阶段的最优权重，逻辑上优先排除消融后缀
    files = sorted([f for f in os.listdir(RES_DIR) if f.startswith(prefix) and f.endswith(".pth") and "No" not in f and "Only" not in f])
    if not files: 
        # 如果没找到全量版，再尝试找最新产生的
        files = sorted([f for f in os.listdir(RES_DIR) if f.startswith(prefix) and f.endswith(".pth")])
    return os.path.join(RES_DIR, files[-1]) if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "ablation"], default="all")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    common = ["--sample_size", str(args.sample_size), "--seed", str(args.seed)]

    # --- Group 1-3 & Full Scheme Training ---
    if args.mode in ["all", "train"]:
        run_script("data_preprocess.py", [])
        run_script("run_classical_tfidf_lr.py", ["--mode", "train"])
        run_script("train_text_cnn.py", common)
        run_script("train_bilstm.py", common)
        run_script("train_vanilla_bert.py", common)
        run_script("train_vanilla_roberta.py", common)
        run_script("train_vanilla_deberta_v3.py", common)
        run_script("train_bert_cnn_bilstm.py", common)

        print("\n>>> 训练本文全量方案 (Full DebertaV3MTL)")
        run_script("train_deberta_v3_mtl_stage1.py", common)
        s1_best = find_latest("DebertaV3MTL_S1_Sample")
        if s1_best: run_script("train_deberta_v3_mtl_stage2.py", ["--s1_checkpoint", s1_best] + common)

    # --- Ablation Phase: 开关驱动模式 ---
    if args.mode in ["all", "ablation"]:
        print("\n>>> 启动消融实验组 (Switch-based Ablations)")
        # Ablation 1: No Attention Pooling
        run_script("train_deberta_v3_mtl_stage1.py", ["--no_pooling"] + common)
        # Ablation 2: No MTL
        run_script("train_deberta_v3_mtl_stage1.py", ["--only_toxicity"] + common)
        # Ablation 3: No Reweighting in S2
        s1_best = find_latest("DebertaV3MTL_S1_Sample")
        if s1_best: run_script("train_deberta_v3_mtl_stage2.py", ["--s1_checkpoint", s1_best, "--no_reweight"] + common)

    # --- Evaluation ---
    if args.mode in ["all", "eval", "ablation"]:
        print("\n>>> 全量指标自动化评估...")
        pths = [f for f in os.listdir(RES_DIR) if f.endswith(".pth")]
        for pth in pths:
            m_type = "deberta_mtl" if "DebertaV3MTL" in pth else \
                     "bert_cnn" if "BertCNN" in pth else \
                     "text_cnn" if "TextCNN" in pth else \
                     "bilstm" if "BiLSTM" in pth else \
                     "vanilla_bert" if "VanillaBERT" in pth else \
                     "vanilla_roberta" if "VanillaRoBERTa" in pth else \
                     "vanilla_deberta" if "VanillaDeBERTa" in pth else "vanilla"
            run_script("full_evaluation.py", ["--checkpoint", os.path.join(RES_DIR, pth), "--model_type", m_type, "--output_prefix", pth.replace(".pth", "")])

    print("\n[FINISH] Standardized Switch-based Experiment Lifecycle Completed.")

if __name__ == "__main__":
    main()
