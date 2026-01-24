import os
import subprocess
import argparse
import sys
import time

# =============================================================================
# ### 实验管理器：run_experiments.py ###
# 设计说明：
# 本脚本作为项目的“总指挥台”，支持全量对比实验与消融实验的一键闭环。
# 分为 5 组实验：
# 1. 经典统计组 (TF-IDF + LR)
# 2. 传统深度学习组 (TextCNN, BiLSTM)
# 3. 强对照组 (BERT+CNN+BiLSTM)
# 4. 原生 Transformer 基准组 (Vanilla BERT/RoBERTa)
# 5. 本文改进方案组 (DeBERTa V3 MTL S1 & S2) -> 包含消融实验开关
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

def find_latest_checkpoint(prefix):
    """ 查找目录下最新的、指定前缀的权重文件 """
    files = [f for f in os.listdir(RES_DIR) if f.startswith(prefix) and f.endswith(".pth")]
    if not files: return None
    return os.path.join(RES_DIR, sorted(files)[-1])

def main():
    parser = argparse.ArgumentParser(description="NLP Toxicity Classification Manager")
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "viz", "ablation"], default="all")
    parser.add_argument("--sample_size", type=int, default=200000, help="全量对齐样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    common = ["--sample_size", str(args.sample_size), "--seed", str(args.seed)]

    # --- [1] 训练阶段 ---
    if args.mode in ["all", "train"]:
        print("\n>>> [Phase 1] 启动标准化训练流程")
        # Step 0: 预处理
        run_script("data_preprocess.py", [])

        # Group 1: Classical
        run_script("train_tfidf_lr.py", ["--mode", "train"])

        # Group 2: Classic DL
        run_script("train_text_cnn.py", common + ["--epochs", "5"])
        run_script("train_bilstm.py", common + ["--epochs", "5"])

        # Group 3: Hybrid & Transformer Baselines
        run_script("train_bert_cnn_bilstm.py", common + ["--epochs", "3"])
        run_script("train_vanilla_bert.py", common + ["--epochs", "3"])
        run_script("train_vanilla_roberta.py", common + ["--epochs", "3"])

        # Group 4: 本文全量方案 (DeBERTa V3 MTL)
        print("\n>>> [Group 4] 训练本文提出方案 (Full Scheme)")
        run_script("train_deberta_v3_mtl_stage1.py", common + ["--epochs", "3"])
        s1_cp = find_latest_checkpoint("DebertaV3MTL_S1")
        if s1_cp:
            run_script("train_deberta_v3_mtl_stage2.py", common + ["--s1_checkpoint", s1_cp, "--epochs", "2"])

    # --- [2] 消融实验阶段 ---
    if args.mode in ["all", "ablation"]:
        print("\n>>> [Phase 2] 启动消融实验矩阵 (Ablation Studies)")
        
        # Ablation 1: Pooling 消融 (无 Attention Pooling)
        print("\n- Ablation 1: Pooling Impact (CLS only)")
        run_script("train_deberta_v3_mtl_stage1.py", common + ["--no_pooling", "--epochs", "3"])
        
        # Ablation 2: MTL 消融 (仅 Toxicity 任务)
        print("\n- Ablation 2: Multi-Task Impact (Only Toxicity)")
        run_script("train_deberta_v3_mtl_stage1.py", common + ["--only_toxicity", "--epochs", "3"])
        
        # Ablation 3: Reweight 消融 (S2 不加权，等同于对比 S1 和 S2)
        print("\n- Ablation 3: Identity-Aware Reweight Impact")
        s1_full_cp = find_latest_checkpoint("DebertaV3MTL_S1") # 这里的 S1 已经不加权了，通过对比 S1 结果和 S2 结果即可。
        # 显式跑一个带“不加权”后缀的 S2 以便对比：
        if s1_full_cp:
            run_script("train_deberta_v3_mtl_stage2.py", common + ["--s1_checkpoint", s1_full_cp, "--no_reweight", "--epochs", "2"])

    # --- [3] 评估阶段 ---
    if args.mode in ["all", "eval", "ablation"]:
        print("\n>>> [Phase 3] 启动全量指标评估流")
        pths = [f for f in os.listdir(RES_DIR) if f.endswith(".pth")]
        for pth in pths:
            # 自动映射模型类型与评估入口
            m_type = "deberta_mtl" if "DebertaV3MTL" in pth else \
                     "bert_cnn" if "BertCNN" in pth else \
                     "text_cnn" if "TextCNN" in pth else \
                     "bilstm" if "BiLSTM" in pth else \
                     "vanilla_bert" if "VanillaBERT" in pth else \
                     "vanilla_roberta" if "VanillaRoBERTa" in pth else "vanilla"
            
            run_script("full_evaluation.py", [
                "--checkpoint", os.path.join(RES_DIR, pth), 
                "--model_type", m_type, 
                "--output_prefix", pth.replace(".pth", "")
            ])

    # --- [4] 可视化阶段 ---
    if args.mode in ["all", "viz"]:
        print("\n>>> [Phase 4] 生成可视化图表")
        run_script("viz_results.py", []) # 汇总图
        latest_s2 = find_latest_checkpoint("DebertaV3MTL_S2_Sample") # 排除 NoReweight 找全量版
        if latest_s2:
            run_script("viz_feature_space.py", ["--checkpoint", latest_s2, "--output_name", "viz_final_scheme.png"])

    print("\n[FINISH] 实验全生命周期管理已圆满完成。")

if __name__ == "__main__":
    main()
