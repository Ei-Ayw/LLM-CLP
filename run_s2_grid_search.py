#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
### S2 超参数网格搜索脚本：run_s2_grid_search.py ###
=============================================================================
自动化搜索 DebertaV3MTL Stage 2 的关键超参数组合：
  - lr: 学习率
  - alpha: 辅助任务权重
  - w_identity: 身份样本权重

用法:
    nohup python run_s2_grid_search.py > grid_search.log 2>&1 &
=============================================================================
"""
import os
import sys
import subprocess
import itertools
import json
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "src_result", "models")
EVAL_DIR = os.path.join(BASE_DIR, "src_result", "eval")

# 核心离线环境配置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# =============================================================================
# 超参数搜索空间
# =============================================================================
SEARCH_SPACE = {
    "lr": [5e-6, 1e-5, 2e-5],
    "alpha": [0.3, 0.5],
    "w_identity": [2.0, 3.0],
}

# 固定参数
FIXED_PARAMS = {
    "sample_size": 300000,
    "batch_size": 32,  # 单卡 batch size (DDP 下)
    "max_len": 256,
    "epochs": 10,
    "early_patience": 3,
    "scheduler": "plateau",
    "patience": 1,
}


def find_latest_s1():
    """查找最新的 S1 权重文件"""
    if not os.path.exists(MODEL_DIR):
        return None
    pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth") and "S1" in f]
    if not pths:
        return None
    pths.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
    return os.path.join(MODEL_DIR, pths[0])


def run_single_experiment(s1_path, lr, alpha, w_identity, run_id):
    """运行单个超参数配置的实验"""
    tag = f"GS_lr{lr}_a{alpha}_w{w_identity}"
    print(f"\n{'='*60}")
    print(f">>> [{run_id}] 启动实验: {tag}")
    print(f"{'='*60}")
    
    # 构建训练命令
    train_script = os.path.join(BASE_DIR, "src_script", "train", "train_deberta_v3_mtl_s2.py")
    train_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=3",  # 根据服务器 GPU 数量调整
        train_script,
        "--s1_checkpoint", s1_path,
        "--lr", str(lr),
        "--alpha", str(alpha),
        "--w_identity", str(w_identity),
        "--ablation_tag", tag,
        "--sample_size", str(FIXED_PARAMS["sample_size"]),
        "--batch_size", str(FIXED_PARAMS["batch_size"]),
        "--max_len", str(FIXED_PARAMS["max_len"]),
        "--epochs", str(FIXED_PARAMS["epochs"]),
        "--early_patience", str(FIXED_PARAMS["early_patience"]),
        "--scheduler", FIXED_PARAMS["scheduler"],
        "--patience", str(FIXED_PARAMS["patience"]),
        "--no_bar",
    ]
    
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f">>> [ERROR] 训练失败: {e}")
        return None
    
    # 查找刚刚生成的权重文件
    pths = [f for f in os.listdir(MODEL_DIR) if tag in f and f.endswith(".pth")]
    if not pths:
        print(f">>> [ERROR] 未找到权重文件: {tag}")
        return None
    pths.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
    ckpt_path = os.path.join(MODEL_DIR, pths[0])
    
    # 运行评估
    eval_script = os.path.join(BASE_DIR, "src_script", "eval", "eval_universal_runner.py")
    output_prefix = pths[0].replace(".pth", "")
    eval_cmd = [
        sys.executable, eval_script,
        "--checkpoint", ckpt_path,
        "--model_type", "deberta_mtl",
        "--output_prefix", output_prefix,
    ]
    
    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f">>> [ERROR] 评估失败: {e}")
        return None
    
    # 读取评估结果
    eval_json = os.path.join(EVAL_DIR, f"{output_prefix}_metrics.json")
    if os.path.exists(eval_json):
        with open(eval_json, 'r') as f:
            metrics = json.load(f)
        return {
            "tag": tag,
            "lr": lr,
            "alpha": alpha,
            "w_identity": w_identity,
            "f1_optimal": metrics["primary_metrics_optimal"]["f1"],
            "f1_fixed": metrics["primary_metrics_fixed_0.5"]["f1"],
            "roc_auc": metrics["primary_metrics_optimal"]["roc_auc"],
            "mean_bias_auc": metrics["bias_metrics"]["mean_bias_auc"],
            "worst_bias_auc": metrics["bias_metrics"]["worst_group_bias_auc"],
            "optimal_threshold": metrics["optimal_threshold"],
        }
    return None


def main():
    print(f"\n{'#'*60}")
    print(f"# S2 超参数网格搜索")
    print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # 查找 S1 权重
    s1_path = find_latest_s1()
    if not s1_path:
        print(">>> [FATAL] 未找到 S1 权重文件！请先运行 S1 训练。")
        sys.exit(1)
    print(f">>> [INFO] 使用 S1 权重: {s1_path}")
    
    # 生成所有超参数组合
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    configs = list(itertools.product(*values))
    
    print(f">>> [INFO] 总计 {len(configs)} 个超参数组合")
    for i, cfg in enumerate(configs):
        print(f"    [{i+1}] lr={cfg[0]}, alpha={cfg[1]}, w_identity={cfg[2]}")
    
    # 运行所有实验
    results = []
    for i, (lr, alpha, w_identity) in enumerate(configs):
        result = run_single_experiment(s1_path, lr, alpha, w_identity, f"{i+1}/{len(configs)}")
        if result:
            results.append(result)
    
    # 生成汇总报告
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("f1_optimal", ascending=False)
        
        summary_path = os.path.join(BASE_DIR, "src_result", "s2_grid_search_summary.csv")
        df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print(">>> 网格搜索完成！结果汇总：")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\n>>> 详细报告已保存至: {summary_path}")
        
        # 输出最佳配置
        best = df.iloc[0]
        print(f"\n>>> 🏆 最佳配置：")
        print(f"    lr={best['lr']}, alpha={best['alpha']}, w_identity={best['w_identity']}")
        print(f"    F1 (Optimal): {best['f1_optimal']:.4f}")
        print(f"    F1 (Fixed 0.5): {best['f1_fixed']:.4f}")
        print(f"    ROC-AUC: {best['roc_auc']:.4f}")
    else:
        print(">>> [WARNING] 没有成功完成的实验！")
    
    print(f"\n>>> 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
