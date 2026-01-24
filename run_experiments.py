import os
import subprocess
import argparse
import sys
import time

# =============================================================================
# ### 实验管理器：run_experiments.py (专业分层架构版) ###
# 设计说明：
# 本脚本作为整个项目的“控制塔”，通过四级分层子目录调度全量实验流程。
# 文件夹结构：data/, train/, eval/, viz/
# 涵盖五大实验矩阵：
# 1. 经典统计 2. 传统深度学习 3. 强对照组 4. 原生 Transformer 5. 本文方案及其消融。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "src_result")
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
    files = sorted([f for f in os.listdir(RES_DIR) if f.startswith("DebertaV3MTL_S1_Sample") and f.endswith(".pth")])
    return os.path.join(RES_DIR, files[-1]) if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "ablation", "viz"], default="all")
    parser.add_argument("--sample_size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    common = ["--sample_size", str(args.sample_size), "--seed", str(args.seed)]

    # --- Phase 1: Data & Models Training ---
    if args.mode in ["all", "train"]:
        # Data Preprocess
        run_script("data", "exp_data_preprocess.py", [])

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
        for pth in [f for f in os.listdir(RES_DIR) if f.endswith(".pth")]:
            # 分类映射逻辑
            m_type = "deberta_mtl" if "DebertaV3MTL" in pth else \
                     "bert_cnn" if "BertCNN" in pth else \
                     "text_cnn" if "TextCNN" in pth else \
                     "bilstm" if "BiLSTM" in pth else \
                     "vanilla_bert" if "VanillaBERT" in pth else \
                     "vanilla_roberta" if "VanillaRoBERTa" in pth else \
                     "vanilla_deberta" if "VanillaDeBERTa" in pth else "vanilla"
            
            run_script("eval", "eval_universal_runner.py", ["--checkpoint", os.path.join(RES_DIR, pth), "--model_type", m_type, "--output_prefix", pth.replace(".pth", "")])

    # --- Phase 4: Viz ---
    if args.mode in ["all", "viz"]:
        run_script("viz", "viz_performance_summary.py", [])
        # 寻找 S2 正式权重进行可视化
        files = sorted([f for f in os.listdir(RES_DIR) if f.startswith("DebertaV3MTL_S2_Sample") and "No" not in f and f.endswith(".pth")])
        if files:
            run_script("viz", "viz_feature_t_sne.py", ["--checkpoint", os.path.join(RES_DIR, files[-1]), "--output_name", "viz_final_paper.png"])

    print("\n[FINISH] Professional Hierarchical Lifecycle Management Completed.")

if __name__ == "__main__":
    main()
