import os
import subprocess
import argparse
import sys
import torch
import time
from datetime import datetime

# =============================================================================
# ### 实验管理器：run_experiments.py (8-GPU 并行调度版) ###
# =============================================================================
# 设计说明：
# 本脚本作为整个项目的"控制塔"，通过四级分层子目录调度全量实验流程。
# 文件夹结构：data/, train/, eval/, viz/
# 数据划分策略:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  原始数据 (train.csv) ──> exp_data_preprocess.py                        │
# │                                                                         │
# │  ┌───────────────────┐   ┌──────────────┐   ┌──────────────┐           │
# │  │   Train (80%)     │   │  Val (10%)   │   │  Test (10%)  │           │
# │  │ train_processed   │   │ val_processed │   │ test_processed│          │
# │  │   .parquet        │   │   .parquet    │   │   .parquet   │           │
# │  └───────────────────┘   └──────────────┘   └──────────────┘           │
# │        ↓                       ↓                    ↓                   │
# │   模型训练用              训练过程验证/         最终评估指标            │
# │  (sample_size控制)        Early Stopping        (从未见过)             │
# └─────────────────────────────────────────────────────────────────────────┘
#
# 实验矩阵 (全量):
#   Group 1: 传统强基线    - TF-IDF + Logistic Regression
#   Group 2: 混合强对照    - BERT + CNN + BiLSTM
#   Group 3: Transformer   - BERT, RoBERTa, DeBERTa-v3 (本文基座)
#   Group 4: 本文方案      - DeBERTa-v3 MTL (两阶段) + 消融实验
#
# 并行调度策略 (8x GPU):
#   Phase 1: 4组 baseline 并行 (各2卡) + TF-IDF (CPU)
#   Phase 2: MTL S1 全8卡 DDP
#   Phase 3: MTL S2 (4卡) + BCE Ablation S1 (4卡) 并行
#   Phase 4: BCE Ablation S2 (4卡) + 已完成模型评估 (4卡) 并行
#   Phase 5: 剩余评估 + 可视化
#
# 输出目录结构 (src_result/):
# ├── models/      # 存储 .pth 模型权重 (Best Model)
# ├── logs/        # 存储训练过程的 Loss 数据与曲线
# ├── eval/        # 存储评估报告与阈值曲线
# └── viz/         # 存储最终的可视化图表 (t-SNE等)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "src_result")
MODEL_DIR = os.path.join(RES_DIR, "models")
LOG_DIR = os.path.join(RES_DIR, "logs")
PYTHON_EXE = sys.executable + " -u"

# 核心离线环境配置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# [Stable Mode] 强制禁用 NCCL P2P 和 IB 以防止跨卡通信问题
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 检测全部可用 GPU (不做人为限制)
TOTAL_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"[System] Detected {TOTAL_GPUS} GPUs. Full parallel mode enabled.")


# =============================================================================
# 核心调度函数
# =============================================================================

def run_script(folder, script_name, args_list, gpu_ids=None, ddp_nproc=None, master_port=29500):
    """执行分层目录下的脚本 (支持指定 GPU 子集和 DDP)"""
    script_path = os.path.join(BASE_DIR, "src_script", folder, script_name)

    env = os.environ.copy()
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    is_ddp = "deberta" in script_name and "train" in script_name and ddp_nproc and ddp_nproc > 0

    if is_ddp:
        print(f"\n[RUN-DDP] {script_name} on GPU [{gpu_ids}] x{ddp_nproc}")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={ddp_nproc}",
            f"--master_port={master_port}",
            script_path
        ] + args_list
    else:
        cmd = PYTHON_EXE.split() + [script_path] + args_list
        print(f"\n[RUN] {folder}/{script_name} on GPU [{gpu_ids or 'all'}]")

    subprocess.run(cmd, check=True, env=env)


def run_parallel_tasks(tasks):
    """
    并行执行多个训练/评估任务，每个任务分配独立的 GPU 子集。

    Args:
        tasks: list of dicts, each with:
            - 'name': 任务名称
            - 'folder': 脚本子目录 (e.g., 'train', 'eval')
            - 'script': 脚本文件名
            - 'args': 命令行参数列表
            - 'gpus': str, CUDA_VISIBLE_DEVICES (e.g., "0,1")
            - 'ddp': bool, 是否用 torchrun 启动 DDP
            - 'nproc': int, DDP 进程数 (仅 ddp=True 时)
            - 'master_port': int, DDP rendezvous 端口 (仅 ddp=True 时)

    Returns:
        dict: {task_name: (return_code, elapsed_seconds)}
    """
    processes = {}
    start_times = {}
    log_files = {}

    for task in tasks:
        script_path = os.path.join(BASE_DIR, "src_script", task['folder'], task['script'])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = task.get('gpus', '')

        if task.get('ddp', False):
            nproc = task.get('nproc', len(task['gpus'].split(',')))
            master_port = task.get('master_port', 29500)
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                f"--master_port={master_port}",
                script_path
            ] + task['args']
        else:
            cmd = PYTHON_EXE.split() + [script_path] + task['args']

        print(f"[PARALLEL] Launching: {task['name']} on GPU [{task.get('gpus', 'CPU')}]")

        # 每个并行任务输出到独立日志文件，避免交叉输出
        log_path = os.path.join(LOG_DIR, f"parallel_{task['name']}.log")
        lf = open(log_path, 'w')
        log_files[task['name']] = lf

        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        processes[task['name']] = proc
        start_times[task['name']] = time.time()

    # 等待所有任务完成
    results = {}
    for name, proc in processes.items():
        proc.wait()
        elapsed = time.time() - start_times[name]
        results[name] = (proc.returncode, elapsed)
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"[PARALLEL] {name}: {status} ({elapsed/60:.1f} min)")
        log_files[name].close()

    # 报告失败任务 (不中断流程)
    failed = {n: r for n, r in results.items() if r[0] != 0}
    if failed:
        print(f"\n[WARNING] {len(failed)} task(s) failed: {list(failed.keys())}")
        for name in failed:
            log_path = os.path.join(LOG_DIR, f"parallel_{name}.log")
            print(f"  -> Check log: {log_path}")

    return results


def find_latest_checkpoint(identifier=None):
    if not os.path.exists(MODEL_DIR): return None
    pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if identifier:
        pths = [f for f in pths if identifier in f]

    if not pths: return None
    pths.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
    return os.path.join(MODEL_DIR, pths[0])


def build_eval_tasks(gpus_start=0, exclude_prefixes=None):
    """构建评估任务列表，用于并行执行。"""
    MODEL_PREFIX_MAP = {
        "DebertaV3MTL_S2_AblationBCE": "deberta_mtl",
        "DebertaV3MTL_S2": "deberta_mtl",
        "BertCNNBiLSTM": "bert_cnn",
        "VanillaBERT": "vanilla_bert",
        "VanillaRoBERTa": "vanilla_roberta",
        "VanillaDeBERTa": "vanilla_deberta",
    }

    if not os.path.exists(MODEL_DIR):
        return []

    pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

    best_checkpoints = {}
    for prefix, m_type in MODEL_PREFIX_MAP.items():
        matched = [p for p in pths if p.startswith(prefix)]
        if matched:
            matched.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
            best_checkpoints[prefix] = (matched[0], m_type)

    if exclude_prefixes:
        best_checkpoints = {k: v for k, v in best_checkpoints.items() if k not in exclude_prefixes}

    tasks = []
    gpu_idx = gpus_start
    for prefix, (pth, m_type) in best_checkpoints.items():
        tasks.append({
            'name': f'Eval_{prefix}',
            'folder': 'eval', 'script': 'eval_universal_runner.py',
            'args': [
                "--checkpoint", os.path.join(MODEL_DIR, pth),
                "--model_type", m_type,
                "--output_prefix", pth.replace(".pth", ""),
            ],
            'gpus': str(gpu_idx),
            'ddp': False,
        })
        gpu_idx += 1
        if gpu_idx > 7:
            gpu_idx = gpus_start
    return tasks


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval", "ablation", "viz", "grid_search"], default="all")
    parser.add_argument("--sample_size", type=int, default=0, help="实验数据采样量 (0表示全量)")
    default_bs = 48 if torch.cuda.is_available() else 16
    parser.add_argument("--batch_size", type=int, default=default_bs, help="Baseline 模型 Batch Size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=256, help="序列最大长度")
    parser.add_argument("--scheduler", type=str, choices=["linear", "plateau"], default="plateau")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=3, help="验证集 Loss 早停等待轮数")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数 (0=默认/早停)")
    parser.add_argument("--no_bar", action="store_true", help="禁用 tqdm 进度条 (nohup模式下推荐)")
    args = parser.parse_args()

    experiment_start_time = datetime.now()
    print(f"[{experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Experiment Session Started.")
    print(f"[System] GPU Count: {TOTAL_GPUS}")

    effective_sample_size = args.sample_size if args.sample_size > 0 else 300_000
    is_full_mode = args.sample_size <= 0
    print(f"[{'FULL' if is_full_mode else 'SAMPLED'} MODE] Effective Sample Size: {effective_sample_size}")

    if not sys.stdout.isatty():
        args.no_bar = True
        print(">>> [Auto-Detect] Non-interactive shell detected. Disabling progress bars.")

    # 预先创建结果子目录
    for d in ["models", "logs", "eval", "viz"]:
        os.makedirs(os.path.join(RES_DIR, d), exist_ok=True)

    # Baseline 通用参数 (BERT/RoBERTa/BertCNNBiLSTM 用)
    common = [
        "--sample_size", str(effective_sample_size),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--max_len", str(args.max_len),
        "--scheduler", args.scheduler,
        "--patience", str(args.patience),
        "--early_patience", str(args.early_patience),
    ]
    if args.epochs > 0:
        common += ["--epochs", str(args.epochs)]
    if args.no_bar:
        common += ["--no_bar"]

    # DeBERTa 专用参数 (DDP 下 batch_size 为单卡值, 16 per GPU)
    deberta_batch_size = 16
    deberta_common = common[:]
    if "--batch_size" in deberta_common:
        idx = deberta_common.index("--batch_size")
        deberta_common[idx+1] = str(deberta_batch_size)

    # GPU ID 字符串
    all_gpus = ",".join(str(i) for i in range(TOTAL_GPUS))  # "0,1,2,3,4,5,6,7"
    gpus_0_1 = "0,1"
    gpus_2_3 = "2,3"
    gpus_4_5 = "4,5"
    gpus_6_7 = "6,7"
    gpus_0_3 = "0,1,2,3"
    gpus_4_7 = "4,5,6,7"

    # =====================================================================
    # Phase 0: 数据预处理 (CPU, 仅在缺失时执行)
    # =====================================================================
    if args.mode in ["all", "train"]:
        data_files = ["train_processed.parquet", "val_processed.parquet", "test_processed.parquet"]
        if not all(os.path.exists(os.path.join(BASE_DIR, "data", f)) for f in data_files):
            print("\n" + "="*70)
            print(">>> Phase 0: Data Preprocessing")
            print("="*70)
            run_script("data", "exp_data_preprocess.py",
                       ["--sample_size", str(effective_sample_size), "--seed", str(args.seed)])
        else:
            print("\n>>> Phase 0: Skipped (preprocessed data already exists)")

    # =====================================================================
    # Phase 1: Baseline 并行训练 (4组 x 2卡 + CPU)
    # =====================================================================
    if args.mode in ["all", "train"]:
        print("\n" + "="*70)
        print(">>> Phase 1: Parallel Baseline Training (4 groups x 2 GPUs + CPU)")
        print("="*70)

        phase1_tasks = [
            {
                'name': 'VanillaBERT',
                'folder': 'train', 'script': 'train_vanilla_bert.py',
                'args': common,
                'gpus': gpus_0_1, 'ddp': False,
            },
            {
                'name': 'VanillaRoBERTa',
                'folder': 'train', 'script': 'train_vanilla_roberta.py',
                'args': common,
                'gpus': gpus_2_3, 'ddp': False,
            },
            {
                'name': 'BertCNNBiLSTM',
                'folder': 'train', 'script': 'train_bert_cnn_bilstm.py',
                'args': common,
                'gpus': gpus_4_5, 'ddp': False,
            },
            {
                'name': 'VanillaDeBERTa',
                'folder': 'train', 'script': 'train_vanilla_deberta_v3.py',
                'args': deberta_common,
                'gpus': gpus_6_7, 'ddp': True, 'nproc': 2, 'master_port': 29500,
            },
            {
                'name': 'TF-IDF_LR',
                'folder': 'train', 'script': 'train_classical_tfidf_lr.py',
                'args': ['--mode', 'train'],
                'gpus': '',  # CPU only
                'ddp': False,
            },
        ]
        run_parallel_tasks(phase1_tasks)

    # =====================================================================
    # Phase 2: 本文方案 MTL S1 (全8卡 DDP)
    # =====================================================================
    if args.mode in ["all", "train"]:
        print("\n" + "="*70)
        print(f">>> Phase 2: MTL Stage 1 Training ({TOTAL_GPUS}-GPU DDP)")
        print("="*70)

        run_script("train", "train_deberta_v3_mtl_s1.py", deberta_common,
                   gpu_ids=all_gpus, ddp_nproc=TOTAL_GPUS, master_port=29500)

        s1_path = find_latest_checkpoint("DebertaV3MTL_S1")
        if s1_path:
            print(f">>> [Found] S1 checkpoint: {s1_path}")

    # =====================================================================
    # Phase 3: MTL S2 (4卡) + BCE Ablation S1 (4卡) 并行
    # =====================================================================
    if args.mode in ["all", "train"]:
        print("\n" + "="*70)
        print(">>> Phase 3: MTL S2 + BCE Ablation S1 (parallel, 4+4 GPUs)")
        print("="*70)

        s1_path = find_latest_checkpoint("DebertaV3MTL_S1")
        phase3_tasks = []

        if s1_path:
            phase3_tasks.append({
                'name': 'MTL_S2',
                'folder': 'train', 'script': 'train_deberta_v3_mtl_s2.py',
                'args': ['--s1_checkpoint', s1_path] + deberta_common,
                'gpus': gpus_0_3, 'ddp': True, 'nproc': 4, 'master_port': 29500,
            })
        else:
            print("[WARNING] S1 checkpoint not found. Skipping MTL S2.")

        phase3_tasks.append({
            'name': 'AblationBCE_S1',
            'folder': 'train', 'script': 'train_deberta_v3_mtl_s1_ablation_bce.py',
            'args': deberta_common,
            'gpus': gpus_4_7, 'ddp': True, 'nproc': 4, 'master_port': 29501,
        })

        if phase3_tasks:
            run_parallel_tasks(phase3_tasks)

    # =====================================================================
    # Phase 4: BCE Ablation S2 (4卡) + 已完成模型评估 (4卡) 并行
    # =====================================================================
    if args.mode in ["all", "train"]:
        print("\n" + "="*70)
        print(">>> Phase 4: BCE Ablation S2 + Early Evaluations (parallel)")
        print("="*70)

        phase4_tasks = []

        s1_ablation_path = find_latest_checkpoint("DebertaV3MTL_S1_AblationBCE")
        if s1_ablation_path:
            print(f">>> [Found] BCE Ablation S1 checkpoint: {s1_ablation_path}")
            phase4_tasks.append({
                'name': 'AblationBCE_S2',
                'folder': 'train', 'script': 'train_deberta_v3_mtl_s2_ablation_bce.py',
                'args': ['--s1_checkpoint', s1_ablation_path] + deberta_common,
                'gpus': gpus_0_3, 'ddp': True, 'nproc': 4, 'master_port': 29500,
            })
        else:
            print("[WARNING] BCE Ablation S1 checkpoint not found. Skipping Ablation S2.")

        # 同时在 GPU 4-7 上并行评估已完成的 baseline 模型
        eval_tasks = build_eval_tasks(
            gpus_start=4,
            exclude_prefixes=["DebertaV3MTL_S2_AblationBCE"],  # S2 ablation 还在训练
        )
        phase4_tasks.extend(eval_tasks)

        if phase4_tasks:
            run_parallel_tasks(phase4_tasks)

    # =====================================================================
    # Phase 5: 剩余评估 + 可视化
    # =====================================================================
    if args.mode in ["all", "eval"]:
        print("\n" + "="*70)
        print(">>> Phase 5: Remaining Evaluations + Visualization")
        print("="*70)

        # 评估所有尚未评估的模型 (包括刚训练完的 Ablation S2)
        remaining_eval = build_eval_tasks(gpus_start=0)
        # 过滤掉已有评估结果的模型
        eval_dir = os.path.join(RES_DIR, "eval")
        if os.path.exists(eval_dir):
            existing_evals = set(f.replace("_metrics.json", "") for f in os.listdir(eval_dir) if f.endswith("_metrics.json"))
            remaining_eval = [t for t in remaining_eval
                              if t['args'][t['args'].index("--output_prefix") + 1] not in existing_evals]

        if remaining_eval:
            print(f">>> Evaluating {len(remaining_eval)} remaining model(s)...")
            run_parallel_tasks(remaining_eval)
        else:
            print(">>> All models already evaluated.")

    if args.mode in ["all", "viz"]:
        print("\n>>> Generating visualizations...")
        run_script("viz", "viz_performance_summary.py", [])

        # t-SNE
        if os.path.exists(MODEL_DIR):
            s2_files = [f for f in os.listdir(MODEL_DIR)
                        if f.startswith("DebertaV3MTL_S2") and "Ablation" not in f and f.endswith(".pth")]
            if s2_files:
                s2_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
                latest_s2 = s2_files[0]
                print(f">>> [Viz] t-SNE using: {latest_s2}")
                run_script("viz", "viz_feature_t_sne.py",
                           ["--checkpoint", os.path.join(MODEL_DIR, latest_s2),
                            "--output_name", "viz_final_paper.png"])

    # =====================================================================
    # Grid Search (独立模式)
    # =====================================================================
    if args.mode in ["grid_search"]:
        print("\n>>> S2 Grid Search...")
        grid_script = os.path.join(BASE_DIR, "run_s2_grid_search.py")
        subprocess.run([sys.executable, grid_script], check=True)

    # =====================================================================
    # Ablation 扩展模式 (独立模式, 需要显式 --mode ablation)
    # =====================================================================
    if args.mode in ["ablation"]:
        print("\n>>> Strict Ablation Matrix (Full S1+S2 Pipeline)")

        base_cmd = deberta_common
        ablations = [
            ("No_Augmentation", ["--no_aug"], ["--no_aug"]),
            ("No_Pooling", ["--no_pooling"], ["--no_pooling"]),
            ("No_Reweight", [], ["--no_reweight"]),
            ("No_Focal", ["--no_focal"], ["--no_focal"]),
        ]

        for case_name, s1_args, s2_args in ablations:
            print(f"\n[Ablation Run] Case: {case_name}")

            print(f"  > Running Stage 1 ({case_name})...")
            run_script("train", "train_deberta_v3_mtl_s1.py",
                       base_cmd + s1_args + ["--ablation_tag", case_name],
                       gpu_ids=all_gpus, ddp_nproc=TOTAL_GPUS, master_port=29500)

            s1_ckpt = find_latest_checkpoint(identifier=f"_{case_name}_")
            if s1_ckpt:
                print(f"  > Found S1 Checkpoint: {s1_ckpt}")
                print(f"  > Running Stage 2 ({case_name})...")
                run_script("train", "train_deberta_v3_mtl_s2.py",
                           base_cmd + s2_args + ["--s1_checkpoint", s1_ckpt, "--ablation_tag", case_name],
                           gpu_ids=all_gpus, ddp_nproc=TOTAL_GPUS, master_port=29500)
            else:
                print(f"  [Error] S1 for {case_name} failed or checkpoint not found. Skipping S2.")

        # 评估消融实验产出的模型
        print("\n>>> Evaluating ablation models...")
        eval_tasks = build_eval_tasks(gpus_start=0)
        if eval_tasks:
            run_parallel_tasks(eval_tasks)

    # 完成
    elapsed_total = (datetime.now() - experiment_start_time).total_seconds()
    print(f"\n[FINISH] All experiments completed. Total time: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
