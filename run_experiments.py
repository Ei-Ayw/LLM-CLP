import os
import subprocess
import argparse
import sys
import time

# =============================================================================
# ### 实验管理器：run_experiments.py ###
# 设计说明：
# 本脚本作为整个项目的“控制塔”，通过参数管理器统一调度各基准模型的训练与评估。
# 严格遵循“一模型一脚本”结构。
# 1. 经典统计组
# 2. 传统深度学习组 (TextCNN, BiLSTM)
# 3. 混合架构对照组 (BERT+CNN+BiLSTM)
# 4. 原生 Transformer 基准组 (Vanilla BERT/RoBERTa)
# 5. 本文改进方案组 (DeBERTa V3 MTL S1 & S2)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src_script")
RES_DIR = os.path.join(BASE_DIR, "src_result")
PYTHON_EXE = sys.executable

# 核心离线环境与镜像配置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def run_script(script_name, args_list):
    """ 执行脚本并捕获运行状态 """
    cmd = [PYTHON_EXE, os.path.join(SRC_DIR, script_name)] + args_list
    print(f"\n[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script_name} 运行失败。")

def main():
    parser = argparse.ArgumentParser(description="NLP Toxicity Classification Manager")
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "viz"], default="all")
    parser.add_argument("--sample_size", type=int, default=200000, help="全量对齐样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    common = ["--sample_size", str(args.sample_size), "--seed", str(args.seed)]

    if args.mode in ["all", "train"]:
        # Step 0: 数据准备
        run_script("data_preprocess.py", [])

        # Group 1: Classical
        run_script("train_tfidf_lr.py", ["--mode", "train"])

        # Group 2: Deep Learning (Classic)
        run_script("train_text_cnn.py", common + ["--epochs", "5"])
        run_script("train_bilstm.py", common + ["--epochs", "5"])

        # Group 3: Baselines (Transformer & Hybrid)
        run_script("train_bert_cnn_bilstm.py", common + ["--epochs", "3"])
        run_script("train_vanilla_bert.py", common + ["--epochs", "3"])
        run_script("train_vanilla_roberta.py", common + ["--epochs", "3"])

        # Group 4: Proposed Method
        run_script("train_deberta_mtl_stage1.py", common + ["--epochs", "3"])
        s1_files = sorted([f for f in os.listdir(RES_DIR) if f.startswith("DebertaV3MTL_S1") and f.endswith(".pth")])
        if s1_files:
            run_script("train_deberta_mtl_stage2.py", common + ["--s1_checkpoint", os.path.join(RES_DIR, s1_files[-1]), "--epochs", "2"])

    if args.mode in ["all", "eval"]:
        print("\n>>> 启动全量指标评估自动化流...")
        pths = [f for f in os.listdir(RES_DIR) if f.endswith(".pth")]
        for pth in pths:
            # 自动映射模型类型
            m_type = "deberta_mtl" if "Deberta" in pth else \
                     "bert_cnn" if "BertCNN" in pth else \
                     "text_cnn" if "TextCNN" in pth else \
                     "bilstm" if "BiLSTM" in pth else \
                     "vanilla_bert" if "VanillaBERT" in pth else \
                     "vanilla_roberta" if "VanillaRoBERTa" in pth else "vanilla"
            run_script("full_evaluation.py", ["--checkpoint", os.path.join(RES_DIR, pth), "--model_type", m_type, "--output_prefix", pth.replace(".pth", "")])

    if args.mode in ["all", "viz"]:
        run_script("viz_results.py", [])
        s2_files = sorted([f for f in os.listdir(RES_DIR) if f.startswith("DebertaV3MTL_S2") and f.endswith(".pth")])
        if s2_files:
            run_script("viz_feature_space.py", ["--checkpoint", os.path.join(RES_DIR, s2_files[-1]), "--output_name", "viz_final_mtl.png"])

    print("\n[FINISH] 标准化实验全生命周期管理已完成。")

if __name__ == "__main__":
    main()
