import os
import subprocess
import argparse
import sys
import time

# =============================================================================
# ### 实验管理器：run_experiments.py ###
# 设计说明：
# 本脚本作为整个项目的“控制塔”，通过参数管理器统一调度各基准模型的训练与评估。
# 涵盖四组对比实验：
# 1. 经典统计方法 (TF-IDF + LR)
# 2. 传统深度学习 (TextCNN, BiLSTM)
# 3. 原生 Transformer (BERT, RoBERTa, DeBERTa)
# 4. 本文提出方法 (DeBERTa V3 MTL Stage 1 & 2)
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
        print(f"[ERROR] {script_name} 失败。")

def main():
    parser = argparse.ArgumentParser(description="NLP Toxicity Classification Manager")
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "viz"], default="all")
    parser.add_argument("--sample_size", type=int, default=200000, help="全量对齐样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    common = ["--sample_size", str(args.sample_size), "--seed", str(args.seed)]

    if args.mode in ["all", "train"]:
        # Step 0: 预处理
        run_script("data_preprocess.py", [])

        # Group 1: 经典统计 (TF-IDF)
        run_script("train_tfidf_lr.py", ["--mode", "train"])

        # Group 2: 传统深度学习
        run_script("train_text_cnn.py", common + ["--epochs", "5"])
        run_script("train_bilstm.py", common + ["--epochs", "5"])

        # Group 3: 混合架构与原生 Transformer
        run_script("train_bert_cnn_bilstm.py", common + ["--epochs", "3"])
        run_script("train_vanilla_bert.py", common + ["--epochs", "3"])
        run_script("train_vanilla_transformers.py", common + ["--model_path", "roberta-base", "--model_tag", "RoBERTa", "--epochs", "3"])

        # Group 4: 本文提出方法 (MTL)
        run_script("train_deberta_mtl_stage1.py", common + ["--epochs", "3"])
        
        # 自动获取最新的 S1 权重来运行 S2
        s1_files = sorted([f for f in os.listdir(RES_DIR) if f.startswith("DebertaV3MTL_S1") and f.endswith(".pth")])
        if s1_files:
            run_script("train_deberta_mtl_stage2.py", common + ["--s1_checkpoint", os.path.join(RES_DIR, s1_files[-1]), "--epochs", "2"])

    if args.mode in ["all", "eval"]:
        print("\n>>> 启动全量指标评估自动化流...")
        pths = [f for f in os.listdir(RES_DIR) if f.endswith(".pth")]
        for pth in pths:
            # 自动推断模型类型进行评估
            m_type = "deberta_mtl" if "Deberta" in pth else \
                     "bert_cnn" if "BertCNN" in pth else \
                     "text_cnn" if "TextCNN" in pth else \
                     "bilstm" if "BiLSTM" in pth else \
                     "vanilla"
            run_script("full_evaluation.py", ["--checkpoint", os.path.join(RES_DIR, pth), "--model_type", m_type, "--output_prefix", pth.replace(".pth", "")])

    if args.mode in ["all", "viz"]:
        run_script("viz_results.py", [])
        # 最先进模型的特征空间展示
        s2_files = sorted([f for f in os.listdir(RES_DIR) if f.startswith("DebertaV3MTL_S2") and f.endswith(".pth")])
        if s2_files:
            run_script("viz_feature_space.py", ["--checkpoint", os.path.join(RES_DIR, s2_files[-1]), "--output_name", "viz_final_mtl.png"])

    print("\n[FINISH] 所有实验生命周期管理已完成。")

if __name__ == "__main__":
    main()
