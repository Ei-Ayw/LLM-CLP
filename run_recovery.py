#!/usr/bin/env python3
"""
恢复脚本：跳过已完成的 baseline，直接跑剩余任务
- S1 checkpoints 已存在 (best model from Epoch 1)
- 只需跑: S2 + Ablation S2 (并行) → TF-IDF + Eval (并行) → Viz
- 目标: 1小时内完成
"""
import os
import sys
import subprocess
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "src_result")
MODEL_DIR = os.path.join(RES_DIR, "models")
LOG_DIR = os.path.join(RES_DIR, "logs")

# 离线环境 + 镜像
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for d in ["models", "logs", "eval", "viz"]:
    os.makedirs(os.path.join(RES_DIR, d), exist_ok=True)


def run_parallel(tasks):
    """并行执行多个任务"""
    procs = {}
    t0s = {}
    lfs = {}

    for t in tasks:
        script = os.path.join(BASE_DIR, "src_script", t['folder'], t['script'])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = t.get('gpus', '')

        if t.get('ddp'):
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={t['nproc']}",
                f"--master_port={t.get('port', 29500)}",
                script
            ] + t['args']
        else:
            cmd = [sys.executable, "-u", script] + t['args']

        log = os.path.join(LOG_DIR, f"recovery_{t['name']}.log")
        lf = open(log, 'w')
        print(f"  [{t['name']}] GPU [{t.get('gpus','CPU')}] -> {log}")

        procs[t['name']] = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        t0s[t['name']] = time.time()
        lfs[t['name']] = lf

    # 等待全部完成
    for name, p in procs.items():
        p.wait()
        dt = (time.time() - t0s[name]) / 60
        rc = p.returncode
        lfs[name].close()
        s = "OK" if rc == 0 else f"FAILED(rc={rc})"
        print(f"  [{name}] {s} ({dt:.1f} min)")

    failed = [n for n, p in procs.items() if p.returncode != 0]
    if failed:
        print(f"  [WARNING] Failed: {failed}")
    return failed


def find_checkpoint(identifier):
    if not os.path.exists(MODEL_DIR): return None
    pths = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth") and identifier in f],
        key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True
    )
    return os.path.join(MODEL_DIR, pths[0]) if pths else None


def main():
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S')}] Recovery started. Target: finish in 1 hour.")
    print()

    # 通用参数
    common_args = [
        "--sample_size", "300000", "--batch_size", "16", "--seed", "42",
        "--max_len", "256", "--scheduler", "plateau", "--patience", "1",
        "--early_patience", "3", "--epochs", "20", "--no_bar",
    ]

    # ================================================================
    # 已完成的 checkpoints
    # ================================================================
    s1_path = find_checkpoint("DebertaV3MTL_S1_Sample")
    s1_abl_path = find_checkpoint("DebertaV3MTL_S1_AblationBCE")

    print("=== Existing Checkpoints ===")
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.endswith('.pth'):
            print(f"  [OK] {f}")
    print(f"  S1 path: {s1_path}")
    print(f"  S1 Ablation path: {s1_abl_path}")
    print()

    # ================================================================
    # Phase A: S2 + Ablation S2 并行 (4+4 GPU) + TF-IDF (CPU)
    # 预计 ~30-40 min
    # ================================================================
    print("=" * 60)
    print("Phase A: MTL S2 (4 GPU) + Ablation S2 (4 GPU) + TF-IDF (CPU)")
    print("=" * 60)

    phase_a = []

    if s1_path:
        phase_a.append({
            'name': 'MTL_S2', 'folder': 'train',
            'script': 'train_deberta_v3_mtl_s2.py',
            'args': ['--s1_checkpoint', s1_path] + common_args,
            'gpus': '0,1,2,3', 'ddp': True, 'nproc': 4, 'port': 29500,
        })
    else:
        print("[ERROR] S1 checkpoint missing! Cannot run S2.")

    if s1_abl_path:
        phase_a.append({
            'name': 'AblationBCE_S2', 'folder': 'train',
            'script': 'train_deberta_v3_mtl_s2_ablation_bce.py',
            'args': ['--s1_checkpoint', s1_abl_path] + common_args,
            'gpus': '4,5,6,7', 'ddp': True, 'nproc': 4, 'port': 29501,
        })
    else:
        print("[ERROR] S1 Ablation checkpoint missing! Cannot run Ablation S2.")

    # TF-IDF 同时跑 (CPU)
    phase_a.append({
        'name': 'TF-IDF_LR', 'folder': 'train',
        'script': 'train_classical_tfidf_lr.py',
        'args': ['--mode', 'train'],
        'gpus': '', 'ddp': False,
    })

    if phase_a:
        run_parallel(phase_a)

    dt_a = (datetime.now() - start).total_seconds() / 60
    print(f"\nPhase A done. Elapsed: {dt_a:.1f} min\n")

    # ================================================================
    # Phase B: 全部模型评估 (6 models, 并行 each 1 GPU)
    # 预计 ~10-15 min
    # ================================================================
    print("=" * 60)
    print("Phase B: Evaluate ALL models (parallel, 1 GPU each)")
    print("=" * 60)

    MODEL_MAP = {
        "DebertaV3MTL_S2_AblationBCE": "deberta_mtl",
        "DebertaV3MTL_S2": "deberta_mtl",
        "BertCNNBiLSTM": "bert_cnn",
        "VanillaBERT": "vanilla_bert",
        "VanillaRoBERTa": "vanilla_roberta",
        "VanillaDeBERTa": "vanilla_deberta",
    }

    eval_tasks = []
    gpu_idx = 0
    pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

    for prefix, mtype in MODEL_MAP.items():
        matched = [p for p in pths if p.startswith(prefix)]
        if matched:
            matched.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
            pth = matched[0]
            eval_tasks.append({
                'name': f'Eval_{prefix}', 'folder': 'eval',
                'script': 'eval_universal_runner.py',
                'args': [
                    "--checkpoint", os.path.join(MODEL_DIR, pth),
                    "--model_type", mtype,
                    "--output_prefix", pth.replace(".pth", ""),
                ],
                'gpus': str(gpu_idx), 'ddp': False,
            })
            gpu_idx = (gpu_idx + 1) % 8

    if eval_tasks:
        print(f"  Evaluating {len(eval_tasks)} models...")
        run_parallel(eval_tasks)
    else:
        print("  No models to evaluate.")

    dt_b = (datetime.now() - start).total_seconds() / 60
    print(f"\nPhase B done. Elapsed: {dt_b:.1f} min\n")

    # ================================================================
    # Phase C: Visualization
    # 预计 ~5 min
    # ================================================================
    print("=" * 60)
    print("Phase C: Visualization")
    print("=" * 60)

    viz_tasks = [{
        'name': 'PerfSummary', 'folder': 'viz',
        'script': 'viz_performance_summary.py', 'args': [],
        'gpus': '0', 'ddp': False,
    }]

    s2_path = find_checkpoint("DebertaV3MTL_S2_Sample")
    if s2_path and "Ablation" not in s2_path:
        viz_tasks.append({
            'name': 'tSNE', 'folder': 'viz',
            'script': 'viz_feature_t_sne.py',
            'args': ["--checkpoint", s2_path, "--output_name", "viz_final_paper.png"],
            'gpus': '1', 'ddp': False,
        })

    run_parallel(viz_tasks)

    total = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"[DONE] Total time: {total:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
