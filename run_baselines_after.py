#!/usr/bin/env python3
"""
接力脚本：等待 supplement 实验完成后，自动跑所有 GPU 基线 + 评估 + 汇总。
用法: nohup python run_baselines_after.py > baselines_run.log 2>&1 &
"""
import os, sys, subprocess, time, json, glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "src_result")
MODEL_DIR = os.path.join(RES_DIR, "models")
EVAL_DIR = os.path.join(RES_DIR, "eval")
PYTHON = sys.executable

# 离线环境
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
TOTAL_GPUS = torch.cuda.device_count()
ALL_GPUS = ",".join(str(i) for i in range(TOTAL_GPUS))

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

def wait_for_supplement():
    """等待 run_supplement_experiments.py 进程结束"""
    log("等待 supplement 实验完成...")
    while True:
        result = subprocess.run(["pgrep", "-f", "run_supplement_experiments.py"], capture_output=True)
        if result.returncode != 0:
            log("supplement 实验已完成（或未运行）")
            return
        time.sleep(60)

def run_training(name, script, args, gpus=None, ddp=True):
    """运行一个训练任务"""
    if gpus is None:
        gpus = ALL_GPUS
    nproc = len(gpus.split(","))
    script_path = os.path.join(BASE_DIR, "src_script", "train", script)
    log_path = os.path.join(RES_DIR, "logs", f"baseline_{name}.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    if ddp:
        cmd = [PYTHON, "-m", "torch.distributed.run",
               f"--nproc_per_node={nproc}", "--master_port=29500",
               script_path] + args
    else:
        cmd = [PYTHON, script_path] + args

    log(f"启动: {name} (GPU={gpus}, nproc={nproc})")
    log(f"  cmd: {' '.join(cmd)}")

    t0 = time.time()
    with open(log_path, 'w') as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"FAILED(rc={proc.returncode})"
    log(f"完成: {name} -> {status} ({elapsed:.1f} min)")
    return proc.returncode == 0

def run_eval(checkpoint, model_type):
    """运行评估"""
    basename = os.path.basename(checkpoint).replace(".pth", "")
    eval_json = os.path.join(EVAL_DIR, f"{basename}_metrics.json")
    if os.path.exists(eval_json):
        log(f"  [Skip] {basename} 已有评估结果")
        return

    script_path = os.path.join(BASE_DIR, "src_script", "eval", "eval_universal_runner.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = os.path.join(RES_DIR, "logs", f"eval_{basename}.log")

    cmd = [PYTHON, script_path,
           "--checkpoint", checkpoint,
           "--model_type", model_type,
           "--output_prefix", basename]

    log(f"  评估: {basename}")
    with open(log_path, 'w') as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        log(f"  [WARN] 评估失败: {basename}, 查看 {log_path}")

def find_checkpoint(keyword):
    if not os.path.exists(MODEL_DIR):
        return None
    pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth") and keyword in f]
    if not pths:
        return None
    pths.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
    return os.path.join(MODEL_DIR, pths[0])

def print_summary():
    """汇总所有评估结果"""
    log("\n" + "=" * 80)
    log(">>> 全部实验结果汇总")
    log("=" * 80)

    results = []
    for f in sorted(os.listdir(EVAL_DIR)):
        if not f.endswith("_metrics.json"):
            continue
        with open(os.path.join(EVAL_DIR, f)) as fh:
            r = json.load(fh)
        p = r.get("primary_metrics_optimal", {})
        b = r.get("bias_metrics", {})
        name = f.replace("_metrics.json", "")
        results.append({
            "name": name,
            "f1": p.get("f1", 0),
            "acc": p.get("accuracy", 0),
            "roc_auc": p.get("roc_auc", 0),
            "pr_auc": p.get("pr_auc", 0),
            "threshold": p.get("threshold", 0.5),
            "mean_bias": b.get("mean_bias_auc", 0),
            "worst_bias": b.get("worst_group_bias_auc", 0),
        })

    # 分类展示
    categories = {
        "传统基线": ["TFIDF"],
        "深度学习基线": ["BertCNN"],
        "Transformer基线": ["VanillaBERT_", "VanillaRoBERTa_", "VanillaDeBERTa_"],
        "主模型 (S2)": ["DebertaV3MTL_S2_Seed"],
        "消融实验 (S2)": ["DebertaV3MTL_S2_"] ,  # 排除 Seed
    }

    header = f"{'模型':<58s} {'F1':>6s} {'AUC':>6s} {'PR-AUC':>6s} {'MBias':>6s} {'WBias':>6s} {'Thresh':>6s}"
    log(header)
    log("-" * 96)

    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        line = f"{r['name']:<58s} {r['f1']:>6.4f} {r['roc_auc']:>6.4f} {r['pr_auc']:>6.4f} {r['mean_bias']:>6.4f} {r['worst_bias']:>6.4f} {r['threshold']:>6.2f}"
        log(line)

    # 主模型多seed统计
    log("\n>>> 主模型多seed统计 (S2)")
    s2_seeds = [r for r in results if "S2_Seed" in r['name'] and "Ablation" not in r['name'] and "BCE" not in r['name']]
    if len(s2_seeds) >= 2:
        import numpy as np
        for metric in ['f1', 'roc_auc', 'mean_bias', 'worst_bias']:
            vals = [r[metric] for r in s2_seeds]
            log(f"  {metric:>12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")

    # 保存汇总JSON
    summary_path = os.path.join(EVAL_DIR, "_ALL_RESULTS_SUMMARY.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\n>>> 汇总已保存: {summary_path}")


def main():
    start = datetime.now()
    log(f"接力脚本启动 | GPU数: {TOTAL_GPUS}")
    os.makedirs(os.path.join(RES_DIR, "logs"), exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ===== Step 1: 等待 supplement 实验 =====
    wait_for_supplement()
    time.sleep(10)  # 等 GPU 释放

    # ===== Step 2: 公共训练参数 =====
    common = ["--sample_size", "300000", "--data_seed", "42",
              "--batch_size", "16", "--max_len", "256",
              "--scheduler", "linear", "--patience", "1", "--early_patience", "3"]

    # ===== Step 3: Vanilla DeBERTa seed=42 =====
    log("\n>>> 基线1: Vanilla DeBERTa (seed=42)")
    run_training("VanillaDeBERTa_Seed42",
                 "train_vanilla_deberta_v3.py",
                 common + ["--seed", "42", "--epochs", "10"])

    # ===== Step 4: Vanilla BERT seed=42 =====
    log("\n>>> 基线2: Vanilla BERT (seed=42)")
    run_training("VanillaBERT_Seed42",
                 "train_vanilla_bert.py",
                 common + ["--seed", "42", "--epochs", "10"])

    # ===== Step 5: Vanilla RoBERTa seed=42 =====
    log("\n>>> 基线3: Vanilla RoBERTa (seed=42)")
    run_training("VanillaRoBERTa_Seed42",
                 "train_vanilla_roberta.py",
                 common + ["--seed", "42", "--epochs", "10"])

    # ===== Step 6: BERT + CNN-BiLSTM seed=42 =====
    log("\n>>> 基线4: BERT + CNN-BiLSTM (seed=42)")
    run_training("BertCNNBiLSTM_Seed42",
                 "train_bert_cnn_bilstm.py",
                 common + ["--seed", "42", "--epochs", "10"])

    # ===== Step 7: 评估所有缺评估的模型 =====
    log("\n" + "=" * 60)
    log(">>> 批量评估所有模型")
    log("=" * 60)

    MODEL_TYPE_MAP = {
        "DebertaV3MTL_S2": "deberta_mtl",
        "DebertaV3MTL_S1": "deberta_mtl",
        "VanillaDeBERTa": "vanilla_deberta",
        "VanillaBERT": "vanilla_bert",
        "VanillaRoBERTa": "vanilla_roberta",
        "BertCNNBiLSTM": "bert_cnn",
    }

    for f in sorted(os.listdir(MODEL_DIR)):
        if not f.endswith(".pth"):
            continue
        # 只评估 S2 和基线，跳过 S1（中间产物）
        if "S1_" in f:
            continue
        basename = f.replace(".pth", "")
        ckpt_path = os.path.join(MODEL_DIR, f)
        m_type = None
        for prefix, mtype in MODEL_TYPE_MAP.items():
            if f.startswith(prefix):
                m_type = mtype
                break
        if m_type:
            run_eval(ckpt_path, m_type)

    # ===== Step 8: 汇总 =====
    print_summary()

    elapsed = (datetime.now() - start).total_seconds() / 60
    log(f"\n[FINISH] 接力脚本完成。总耗时: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
