import os
import subprocess
import argparse
import sys
import time

# =============================================================================
# ### 实验管理器：run_experiments.py (物理拆分增强版) ###
# 设计说明：
# 本脚本严格遵循“一模型一物理文件”原则，调度所有基准实验与消融实验。
# 涵盖 5 大实验物理阵列。
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
    print(f"\n[PHYSICAL RUN] {' '.join(cmd)}")
    subprocess.run(cmd)

def find_latest(prefix):
    files = sorted([f for f in os.listdir(RES_DIR) if f.startswith(prefix) and f.endswith(".pth")])
    return os.path.join(RES_DIR, files[-1]) if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "ablation"], default="all")
    parser.add_argument("--sample_size", type=int, default=200000)
    args = parser.parse_args()

    common = ["--sample_size", str(args.sample_size)]

    if args.mode in ["all", "train"]:
        # 1. 预处理
        run_script("data_preprocess.py", [])

        # 2. Group 1: Classical
        run_script("run_classical_tfidf_lr.py", ["--mode", "train"])

        # 3. Group 2: DL Classic
        run_script("train_text_cnn.py", common); run_script("train_bilstm.py", common)

        # 4. Group 3: Transformer Baselines
        run_script("train_vanilla_bert.py", common)
        run_script("train_vanilla_roberta.py", common)
        run_script("train_vanilla_deberta_v3.py", common)
        run_script("train_bert_cnn_bilstm.py", common)

        # 5. Group 5: Final Scheme (Proposed)
        run_script("train_deberta_v3_mtl_stage1.py", common)
        s1_full = find_latest("DebertaV3MTL_S1_Sample") # 这里的 prefix 避开 NoPooling 等
        if s1_full: run_script("train_deberta_v3_mtl_stage2.py", ["--s1_checkpoint", s1_full] + common)

    if args.mode in ["all", "ablation"]:
        # 6. Group 4: Ablation Study Matrix
        print("\n>>> Running Ablation Matrix (Physical Mode)")
        run_script("train_deberta_v3_cls_only.py", common)      # Ablation 1
        run_script("train_deberta_v3_single_task.py", common)   # Ablation 2
        
        s1_for_ab3 = find_latest("DebertaV3MTL_S1_Sample")
        if s1_for_ab3: 
            run_script("train_deberta_v3_mtl_stage2_no_reweight.py", ["--s1_checkpoint", s1_for_ab3] + common) # Ablation 3

    if args.mode in ["all", "eval", "ablation"]:
        print("\n>>> Launching Universal Evaluator...")
        for pth in [f for f in os.listdir(RES_DIR) if f.endswith(".pth")]:
            m_type = "deberta_mtl" if "DebertaV3MTL_S" in pth else \
                     "ablation_cls" if "CLSOnly" in pth else \
                     "ablation_single" if "SingleTask" in pth else \
                     "bert_cnn" if "BertCNN" in pth else \
                     "text_cnn" if "TextCNN" in pth else \
                     "bilstm" if "BiLSTM" in pth else \
                     "vanilla_bert" if "VanillaBERT" in pth else \
                     "vanilla_roberta" if "VanillaRoBERTa" in pth else \
                     "vanilla_deberta" if "VanillaDeBERTa" in pth else "vanilla"
            run_script("full_evaluation.py", ["--checkpoint", os.path.join(RES_DIR, pth), "--model_type", m_type, "--output_prefix", pth.replace(".pth", "")])

    print("\n[COMPLETE] Physical One-Model-One-Script Pipeline Finish.")

if __name__ == "__main__":
    main()
