import os
import subprocess
import argparse
import sys
import time

# =============================================================================
# ### 实验管理器：run_experiments_manager.py ###
# 设计说明：
# 本脚本作为整个项目的“控制塔”，通过参数管理器统一调度各模型的训练、评估与可视化。
# 支持全量运行或指定模型运行，并确保所有实验均遵循相同的 200k 数据对齐策略。
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src_script")
PYTHON_EXE = sys.executable

# 设置 Hugging Face 离线模式与镜像
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def run_script(script_name, args_list):
    """ 执行指定的脚本并传递参数 """
    cmd = [PYTHON_EXE, os.path.join(SRC_DIR, script_name)] + args_list
    print(f"\n[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] 脚本 {script_name} 运行失败，返回码: {result.returncode}")
        # 这里可以选择是否直接退出
        # sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="NLP Toxicity Classification Experiment Manager")
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "viz"], default="all", help="运行模式")
    parser.add_argument("--sample_size", type=int, default=200000, help="全量实验对齐的样本量")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--skip_classical", action="store_true", help="是否跳过经典的 TF-IDF + LR 实验")
    args = parser.parse_args()

    print("="*60)
    print(f" NLP 毒性分类全量实验管理中心")
    print(f" 启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 训练数据规模: {args.sample_size} | 随机种子: {args.seed}")
    print("="*60)

    # 1. 数据预处理
    if args.mode in ["all", "train"]:
        print("\n>>> [Step 1] 数据预处理")
        run_script("data_preprocess.py", [])

    # 2. 模型训练阶段
    if args.mode in ["all", "train"]:
        common_args = ["--sample_size", str(args.sample_size), "--seed", str(args.seed)]
        
        # (A) 经典基准：TF-IDF + LR
        if not args.skip_classical:
            print("\n>>> [Step 2.1] 训练经典基准 (TF-IDF + LR)")
            run_script("train_tfidf_lr.py", ["--mode", "train"])

        # (B) 混合架构：BERT + CNN + BiLSTM
        print("\n>>> [Step 2.2] 训练对比模型 (BERT + CNN + BiLSTM)")
        run_script("train_bert_cnn_bilstm.py", common_args + ["--epochs", "3"])

        # (C) 核心模型：DeberTa-V3 MTL Stage 1
        print("\n>>> [Step 2.3] 训练核心模型 (DeberTa-V3 MTL) - Stage 1")
        run_script("train_deberta_mtl_stage1.py", common_args + ["--epochs", "3"])
        
        # (D) 核心模型：DeberTa-V3 MTL Stage 2 (需要自动锁定最新的 S1 权重，或者手动指定)
        print("\n>>> [Step 2.4] 训练核心模型 (DeberTa-V3 MTL) - Stage 2 (Reweighting)")
        # 注意：这里需要根据 S1 的输出查找最新的 .pth
        res_files = [f for f in os.listdir(os.path.join(BASE_DIR, "src_result")) if f.startswith("DebertaV3MTL_S1")]
        if res_files:
            latest_s1 = os.path.join(BASE_DIR, "src_result", sorted(res_files)[-1])
            run_script("train_deberta_mtl_stage2.py", common_args + ["--s1_checkpoint", latest_s1, "--epochs", "2"])
        else:
            print("[Warning] 未找到 Stage 1 权重，跳过 Stage 2 训练。")

    # 3. 评估阶段
    if args.mode in ["all", "eval"]:
        print("\n>>> [Step 3] 全量指标评估")
        # 调用 full_evaluation.py 对所有生成的 .pth 进行扫描
        res_dir = os.path.join(BASE_DIR, "src_result")
        pths = [f for f in os.listdir(res_dir) if f.endswith(".pth")]
        
        for pth in pths:
            m_type = "deberta_mtl" if "Deberta" in pth else "bert_cnn"
            run_script("full_evaluation.py", ["--checkpoint", os.path.join(res_dir, pth), "--model_type", m_type, "--output_prefix", pth.replace(".pth", "")])

    # 4. 可视化阶段
    if args.mode in ["all", "viz"]:
        print("\n>>> [Step 4] 可视化分析")
        res_dir = os.path.join(BASE_DIR, "src_result")
        # 选择最新的 S2 模型进行特征空间可视化
        s2_files = [f for f in os.listdir(res_dir) if f.startswith("DebertaV3MTL_S2")]
        if s2_files:
            latest_s2 = os.path.join(res_dir, sorted(s2_files)[-1])
            run_script("viz_feature_space.py", ["--checkpoint", latest_s2, "--subgroup", "black", "--output_name", "viz_s2_black.png"])
        
        # 运行综合对比图
        run_script("viz_results.py", [])

    print("\n" + "="*60)
    print(" 所有实验流程执行完毕。")
    print("="*60)

if __name__ == "__main__":
    main()
