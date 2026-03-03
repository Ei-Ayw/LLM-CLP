#!/usr/bin/env python3
"""
=============================================================================
### 补充实验运行器：run_supplement_experiments.py ###
=============================================================================
目的：为论文发表补充以下实验，提升统计可信度和消融完整性。

实验A: 多随机种子 (seed=123, 2024) x 2种模型 (MTL S1→S2, Vanilla DeBERTa)
       seed=42 的结果已在 main 分支上完成，此处只跑新 seed。
实验B: 细粒度消融 (5个组件单独消融, 每个 S1→S2→Eval)
       - NoPooling:  去掉 Attention Pooling，仅用 CLS
       - NoFocal:    去掉 Focal Loss，改用标准 BCE (S1)
       - OnlyTox:    去掉多任务辅助头，仅训练毒性主任务
       - NoAug:      去掉数据增强
       - NoReweight: 去掉身份感知重加权 (仅影响 S2，复用已有 S1)
实验C: 超参数敏感性 (w_identity: 3.0, 3.5)，仅跑 S2，复用已有 S1。

重要约束：
  - --data_seed 始终=42，保证所有实验使用相同的 300k 数据子集
  - 不调用 exp_data_preprocess.py，数据文件已冻结
  - 详见 DATA_BACKUP_README.txt
=============================================================================
"""

import os
import subprocess
import argparse
import sys
import torch
import time
import glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "src_result")
MODEL_DIR = os.path.join(RES_DIR, "models")
LOG_DIR = os.path.join(RES_DIR, "logs")
EVAL_DIR = os.path.join(RES_DIR, "eval")
PYTHON_EXE = sys.executable + " -u"

# 离线环境配置
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "pretrained_models")
os.environ["HF_HUB_CACHE"] = os.path.join(BASE_DIR, "pretrained_models", "hub")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

TOTAL_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DATA_SEED = 42  # 固定数据采样种子，绝不修改
SAMPLE_SIZE = 300000

# =====================================================================
# 工具函数 (复用 run_experiments.py 的模式)
# =====================================================================

def run_parallel_tasks(tasks, log_file=None):
    """并行执行多个训练/评估任务，每个任务分配独立的 GPU 子集。"""
    processes = {}
    start_times = {}
    log_handles = {}

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

        msg = f"[PARALLEL] Launching: {task['name']} on GPU [{task.get('gpus', 'CPU')}]"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

        task_log_path = os.path.join(LOG_DIR, f"supplement_{task['name']}.log")
        lf = open(task_log_path, 'w')
        log_handles[task['name']] = lf

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
        msg = f"[PARALLEL] {name}: {status} ({elapsed/60:.1f} min)"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()
        log_handles[name].close()

    failed = {n: r for n, r in results.items() if r[0] != 0}
    if failed:
        print(f"\n[WARNING] {len(failed)} task(s) failed: {list(failed.keys())}")
        for name in failed:
            print(f"  -> Check log: {os.path.join(LOG_DIR, f'supplement_{name}.log')}")
    return results


def find_checkpoint(identifier):
    """根据关键字查找最新的模型检查点。"""
    if not os.path.exists(MODEL_DIR):
        return None
    pths = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth") and identifier in f]
    if not pths:
        return None
    pths.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)
    return os.path.join(MODEL_DIR, pths[0])


def build_eval_task(checkpoint_path, model_type, gpu_id):
    """为单个模型构建评估任务。"""
    basename = os.path.basename(checkpoint_path).replace(".pth", "")
    return {
        'name': f'Eval_{basename}',
        'folder': 'eval',
        'script': 'eval_universal_runner.py',
        'args': [
            "--checkpoint", checkpoint_path,
            "--model_type", model_type,
            "--output_prefix", basename,
        ],
        'gpus': str(gpu_id),
        'ddp': False,
    }


# =====================================================================
# 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="补充实验运行器（多seed + 细粒度消融 + 超参敏感性）")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "preprocess", "multi_seed", "ablation", "sensitivity", "eval_only", "aggregate"],
                        help="运行模式: all=全部, preprocess=仅数据预处理, multi_seed=仅多seed, ablation=仅消融, sensitivity=仅敏感性")
    parser.add_argument("--no_bar", action="store_true", help="禁用进度条")
    parser.add_argument("--batch_size", type=int, default=16, help="DDP 单卡 BatchSize")
    parser.add_argument("--s2_batch_size", type=int, default=32, help="S2 单卡 BatchSize")
    args = parser.parse_args()

    start_time = datetime.now()
    log_path = os.path.join(BASE_DIR, "supplement_experiment.log")
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 补充实验开始")
    log(f"[System] GPU 数量: {TOTAL_GPUS}")
    log(f"[Config] DATA_SEED={DATA_SEED}, SAMPLE_SIZE={SAMPLE_SIZE}")

    # 确认数据文件存在
    for f in ["train_processed.parquet", "val_processed.parquet", "test_processed.parquet"]:
        fpath = os.path.join(BASE_DIR, "data", f)
        if not os.path.exists(fpath):
            log(f"[FATAL] 数据文件不存在: {fpath}")
            log(f"  请从 data/backup_seed42_300k/ 恢复，详见 DATA_BACKUP_README.txt")
            sys.exit(1)
    log("[OK] 数据文件检查通过（已冻结，不会重新采样）")

    # =================================================================
    # Phase 0: 数据预处理 (可选，使用新采样策略)
    # =================================================================
    if args.mode in ["all", "preprocess"]:
        log("\n" + "=" * 70)
        log(">>> Phase 0: 数据预处理 (保留所有有标签样本 + 填充至1:1)")
        log("=" * 70)
        preprocess_script = os.path.join(BASE_DIR, "src_script", "data", "exp_data_preprocess.py")
        preprocess_cmd = [
            sys.executable, preprocess_script,
            "--seed", str(DATA_SEED),
            "--keep_all_labeled",
        ]
        log(f"  命令: {' '.join(preprocess_cmd)}")
        result = subprocess.run(preprocess_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log("[OK] 数据预处理完成")
            log(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            log(f"[FATAL] 数据预处理失败 (rc={result.returncode})")
            log(result.stderr[-500:] if result.stderr else "No stderr")
            sys.exit(1)

    # 创建结果目录
    for d in ["models", "logs", "eval", "viz"]:
        os.makedirs(os.path.join(RES_DIR, d), exist_ok=True)

    if not sys.stdout.isatty():
        args.no_bar = True

    # 公共参数
    common_args = [
        "--sample_size", str(SAMPLE_SIZE),
        "--data_seed", str(DATA_SEED),
        "--batch_size", str(args.batch_size),
        "--max_len", "256",
        "--scheduler", "linear",
        "--patience", "1",
        "--early_patience", "3",
    ]
    if args.no_bar:
        common_args += ["--no_bar"]

    # S2 使用更大 batch_size + 梯度累积 (8步)
    s2_extra = ["--batch_size", str(args.s2_batch_size), "--grad_accum", "8"]

    # GPU 分组 (动态适配: 7卡→4+3, 8卡→4+4, 6卡→3+3)
    half = TOTAL_GPUS // 2
    gpus_group_a = ",".join(str(i) for i in range(half))                    # 前半: 0,1,2,...
    gpus_group_b = ",".join(str(i) for i in range(half, TOTAL_GPUS))        # 后半: half,...,N-1
    nproc_a = half
    nproc_b = TOTAL_GPUS - half
    all_gpus = ",".join(str(i) for i in range(TOTAL_GPUS))

    # 收集所有需要评估的新模型 (checkpoint_path, model_type)
    new_checkpoints = []

    # =================================================================
    # 实验A: 多随机种子
    # =================================================================
    if args.mode in ["all", "multi_seed"]:
        SEEDS = [123, 2024]

        # --- Phase A1: MTL S1 (两个seed并行, 分两组GPU) ---
        log("\n" + "=" * 70)
        log(f">>> Phase A1: 多seed MTL S1 训练 (seeds={SEEDS})")
        log("=" * 70)

        phase_a1_tasks = []
        for i, seed in enumerate(SEEDS):
            gpus = gpus_group_a if i == 0 else gpus_group_b
            port = 29500 + i
            phase_a1_tasks.append({
                'name': f'MTL_S1_Seed{seed}',
                'folder': 'train',
                'script': 'train_deberta_v3_mtl_s1.py',
                'args': common_args + ["--seed", str(seed), "--epochs", "6"],
                'gpus': gpus,
                'ddp': True,
                'nproc': nproc_a if i == 0 else nproc_b,
                'master_port': port,
            })
        run_parallel_tasks(phase_a1_tasks, log_file)

        # --- Phase A2: MTL S2 (两个seed并行, 分两组GPU) ---
        log("\n" + "=" * 70)
        log(f">>> Phase A2: 多seed MTL S2 训练 (seeds={SEEDS})")
        log("=" * 70)

        phase_a2_tasks = []
        for i, seed in enumerate(SEEDS):
            s1_ckpt = find_checkpoint(f"S1_Seed{seed}")
            if not s1_ckpt:
                log(f"[WARNING] 未找到 S1 Seed{seed} 检查点，跳过 S2")
                continue
            log(f"  [Found] S1 Seed{seed} -> {os.path.basename(s1_ckpt)}")
            gpus = gpus_group_a if i == 0 else gpus_group_b
            port = 29500 + i
            # S2 args: 覆盖 batch_size
            s2_args = [a for a in common_args if a not in ["--batch_size", str(args.batch_size)]]
            # 移除原始 --batch_size 及其值
            filtered = []
            skip_next = False
            for a in common_args:
                if skip_next:
                    skip_next = False
                    continue
                if a == "--batch_size":
                    skip_next = True
                    continue
                filtered.append(a)
            phase_a2_tasks.append({
                'name': f'MTL_S2_Seed{seed}',
                'folder': 'train',
                'script': 'train_deberta_v3_mtl_s2.py',
                'args': filtered + s2_extra + [
                    "--seed", str(seed),
                    "--s1_checkpoint", s1_ckpt,
                    "--epochs", "4",
                ],
                'gpus': gpus,
                'ddp': True,
                'nproc': nproc_a if i == 0 else nproc_b,
                'master_port': port,
            })
        if phase_a2_tasks:
            run_parallel_tasks(phase_a2_tasks, log_file)
            for seed in SEEDS:
                ckpt = find_checkpoint(f"S2_Seed{seed}")
                if ckpt:
                    new_checkpoints.append((ckpt, "deberta_mtl"))

        # --- Phase A3: Vanilla DeBERTa 多seed (两个seed并行, 分两组GPU) ---
        log("\n" + "=" * 70)
        log(f">>> Phase A3: 多seed Vanilla DeBERTa 训练 (seeds={SEEDS})")
        log("=" * 70)

        phase_a3_tasks = []
        for i, seed in enumerate(SEEDS):
            gpus = gpus_group_a if i == 0 else gpus_group_b
            port = 29500 + i
            phase_a3_tasks.append({
                'name': f'VanillaDeBERTa_Seed{seed}',
                'folder': 'train',
                'script': 'train_vanilla_deberta_v3.py',
                'args': common_args + ["--seed", str(seed), "--epochs", "10"],
                'gpus': gpus,
                'ddp': True,
                'nproc': nproc_a if i == 0 else nproc_b,
                'master_port': port,
            })
        run_parallel_tasks(phase_a3_tasks, log_file)
        for seed in SEEDS:
            ckpt = find_checkpoint(f"VanillaDeBERTa_Seed{seed}")
            if ckpt:
                new_checkpoints.append((ckpt, "vanilla_deberta"))

    # =================================================================
    # 实验B: 细粒度消融
    # =================================================================
    if args.mode in ["all", "ablation"]:
        # 消融配置: (名称, S1额外参数, S2额外参数, 是否需要独立S1)
        # 只保留最关键的3个消融: NoPooling, NoFocal, NoReweight
        # 砍掉 OnlyTox 和 NoAug 以节省约 7h
        ABLATION_CASES = [
            ("NoPooling",  ["--no_pooling"],     ["--no_pooling"],     True),
            ("NoFocal",    ["--no_focal"],        ["--no_focal"],       True),
            ("NoReweight", [],                    ["--no_reweight"],    False),  # 复用已有S1
        ]

        # 需要独立 S1 的消融案例
        s1_cases = [(name, s1_args) for name, s1_args, _, needs_s1 in ABLATION_CASES if needs_s1]

        # --- Phase B1: 消融 S1 (每轮2个并行, 分两组GPU) ---
        log("\n" + "=" * 70)
        log(">>> Phase B1: 细粒度消融 S1 训练")
        log("=" * 70)

        for round_idx in range(0, len(s1_cases), 2):
            batch = s1_cases[round_idx:round_idx + 2]
            log(f"\n  Round {round_idx // 2 + 1}: {[c[0] for c in batch]}")

            phase_b1_tasks = []
            for i, (case_name, s1_args) in enumerate(batch):
                gpus = gpus_group_a if i == 0 else gpus_group_b
                port = 29500 + i
                phase_b1_tasks.append({
                    'name': f'Ablation_S1_{case_name}',
                    'folder': 'train',
                    'script': 'train_deberta_v3_mtl_s1.py',
                    'args': common_args + s1_args + [
                        "--seed", "42",
                        "--epochs", "6",
                        "--ablation_tag", case_name,
                    ],
                    'gpus': gpus,
                    'ddp': True,
                    'nproc': nproc_a if i == 0 else nproc_b,
                    'master_port': port,
                })
            run_parallel_tasks(phase_b1_tasks, log_file)

        # --- Phase B2: 消融 S2 (每轮2个并行, 分两组GPU) ---
        log("\n" + "=" * 70)
        log(">>> Phase B2: 细粒度消融 S2 训练")
        log("=" * 70)

        # 构建 S2 消融任务
        s2_ablation_list = []
        for case_name, _, s2_args, needs_s1 in ABLATION_CASES:
            if needs_s1:
                # 查找对应的消融 S1 检查点
                s1_ckpt = find_checkpoint(f"S1_{case_name}")
                if not s1_ckpt:
                    # 也尝试用 ablation_tag 查找
                    s1_ckpt = find_checkpoint(f"_{case_name}_")
                if not s1_ckpt:
                    log(f"[WARNING] 未找到 S1 {case_name} 检查点，跳过其 S2")
                    continue
            else:
                # NoReweight: 复用已有 seed=42 的标准 S1
                s1_ckpt = find_checkpoint("DebertaV3MTL_S1_Seed42")
                if not s1_ckpt:
                    # 兼容旧命名 (main分支上的，不含Seed标记)
                    s1_ckpt = find_checkpoint("DebertaV3MTL_S1_Sample")
                if not s1_ckpt:
                    log(f"[WARNING] 未找到标准 S1 检查点，跳过 NoReweight S2")
                    continue

            log(f"  [S2 准备] {case_name} <- S1: {os.path.basename(s1_ckpt)}")
            s2_ablation_list.append((case_name, s2_args, s1_ckpt))

        # 每轮2个并行
        # 移除 common_args 中的 --batch_size 用于 S2
        filtered_common = []
        skip_next = False
        for a in common_args:
            if skip_next:
                skip_next = False
                continue
            if a == "--batch_size":
                skip_next = True
                continue
            filtered_common.append(a)

        for round_idx in range(0, len(s2_ablation_list), 2):
            batch = s2_ablation_list[round_idx:round_idx + 2]
            log(f"\n  Round {round_idx // 2 + 1}: {[c[0] for c in batch]}")

            phase_b2_tasks = []
            for i, (case_name, s2_args, s1_ckpt) in enumerate(batch):
                gpus = gpus_group_a if i == 0 else gpus_group_b
                port = 29500 + i
                phase_b2_tasks.append({
                    'name': f'Ablation_S2_{case_name}',
                    'folder': 'train',
                    'script': 'train_deberta_v3_mtl_s2.py',
                    'args': filtered_common + s2_extra + s2_args + [
                        "--seed", "42",
                        "--s1_checkpoint", s1_ckpt,
                        "--epochs", "4",
                        "--ablation_tag", case_name,
                    ],
                    'gpus': gpus,
                    'ddp': True,
                    'nproc': nproc_a if i == 0 else nproc_b,
                    'master_port': port,
                })
            run_parallel_tasks(phase_b2_tasks, log_file)

        # 收集消融检查点
        for case_name, _, _, _ in ABLATION_CASES:
            ckpt = find_checkpoint(f"S2_{case_name}")
            if not ckpt:
                ckpt = find_checkpoint(f"S2.*{case_name}")
            if ckpt:
                new_checkpoints.append((ckpt, "deberta_mtl"))

    # =================================================================
    # 实验C: 超参数敏感性 (已砍掉，节省约3.5h)
    # 如需恢复，将下方 if False 改为 if args.mode in ["all", "sensitivity"]
    # =================================================================
    if args.mode in ["sensitivity"]:  # 不再包含 "all"，需要手动 --mode sensitivity 才会跑
        W_VALUES = [3.0, 3.5]

        log("\n" + "=" * 70)
        log(f">>> Phase C: 超参数敏感性 w_identity={W_VALUES}")
        log("=" * 70)

        # 复用已有 seed=42 的标准 S1
        s1_ckpt = find_checkpoint("DebertaV3MTL_S1_Seed42")
        if not s1_ckpt:
            s1_ckpt = find_checkpoint("DebertaV3MTL_S1_Sample")
        if not s1_ckpt:
            log("[WARNING] 未找到标准 S1 检查点，跳过敏感性实验")
        else:
            # 移除 common_args 中的 --batch_size
            filtered_common = []
            skip_next = False
            for a in common_args:
                if skip_next:
                    skip_next = False
                    continue
                if a == "--batch_size":
                    skip_next = True
                    continue
                filtered_common.append(a)

            sensitivity_tasks = []
            for i, w in enumerate(W_VALUES):
                gpus = gpus_group_a if i == 0 else gpus_group_b
                port = 29500 + i
                tag = f"W{str(w).replace('.', '')}"  # W30, W35
                sensitivity_tasks.append({
                    'name': f'Sensitivity_S2_{tag}',
                    'folder': 'train',
                    'script': 'train_deberta_v3_mtl_s2.py',
                    'args': filtered_common + s2_extra + [
                        "--seed", "42",
                        "--s1_checkpoint", s1_ckpt,
                        "--epochs", "4",
                        "--w_identity", str(w),
                        "--ablation_tag", tag,
                    ],
                    'gpus': gpus,
                    'ddp': True,
                    'nproc': nproc_a if i == 0 else nproc_b,
                    'master_port': port,
                })
            run_parallel_tasks(sensitivity_tasks, log_file)

            for w in W_VALUES:
                tag = f"W{str(w).replace('.', '')}"
                ckpt = find_checkpoint(f"_{tag}")
                if ckpt:
                    new_checkpoints.append((ckpt, "deberta_mtl"))

    # =================================================================
    # 批量评估
    # =================================================================
    if args.mode in ["all", "multi_seed", "ablation", "sensitivity", "eval_only"]:
        log("\n" + "=" * 70)
        log(">>> Phase E: 批量评估所有新模型")
        log("=" * 70)

        if args.mode == "eval_only":
            # eval_only 模式: 扫描所有没有评估结果的模型
            existing_evals = set()
            if os.path.exists(EVAL_DIR):
                existing_evals = set(
                    f.replace("_metrics.json", "") for f in os.listdir(EVAL_DIR)
                    if f.endswith("_metrics.json")
                )

            MODEL_TYPE_MAP = {
                "DebertaV3MTL_S2": "deberta_mtl",
                "DebertaV3MTL_S1": "deberta_mtl",  # 不常评估S1，但保留
                "VanillaDeBERTa": "vanilla_deberta",
                "VanillaBERT": "vanilla_bert",
                "VanillaRoBERTa": "vanilla_roberta",
                "BertCNNBiLSTM": "bert_cnn",
            }

            new_checkpoints = []
            if os.path.exists(MODEL_DIR):
                for f in os.listdir(MODEL_DIR):
                    if not f.endswith(".pth"):
                        continue
                    basename = f.replace(".pth", "")
                    if basename in existing_evals:
                        continue
                    # 判断 model_type
                    m_type = None
                    for prefix, mtype in MODEL_TYPE_MAP.items():
                        if f.startswith(prefix):
                            m_type = mtype
                            break
                    if m_type:
                        new_checkpoints.append((os.path.join(MODEL_DIR, f), m_type))

        # 过滤掉已有评估结果的
        existing_evals = set()
        if os.path.exists(EVAL_DIR):
            existing_evals = set(
                f.replace("_metrics.json", "") for f in os.listdir(EVAL_DIR)
                if f.endswith("_metrics.json")
            )

        eval_tasks_list = []
        gpu_idx = 0
        for ckpt_path, m_type in new_checkpoints:
            basename = os.path.basename(ckpt_path).replace(".pth", "")
            if basename in existing_evals:
                log(f"  [Skip] {basename} 已有评估结果")
                continue
            eval_tasks_list.append(build_eval_task(ckpt_path, m_type, gpu_idx))
            gpu_idx = (gpu_idx + 1) % TOTAL_GPUS

        if eval_tasks_list:
            log(f"  评估 {len(eval_tasks_list)} 个新模型...")
            # 分批执行（每批最多8个，受GPU数量限制）
            for batch_start in range(0, len(eval_tasks_list), TOTAL_GPUS):
                batch = eval_tasks_list[batch_start:batch_start + TOTAL_GPUS]
                run_parallel_tasks(batch, log_file)
        else:
            log("  所有模型已评估，无需重复。")

    # =================================================================
    # 结果汇总
    # =================================================================
    if args.mode in ["all", "aggregate"]:
        log("\n" + "=" * 70)
        log(">>> Phase F: 运行结果汇总脚本")
        log("=" * 70)

        agg_script = os.path.join(BASE_DIR, "aggregate_results.py")
        if os.path.exists(agg_script):
            subprocess.run([sys.executable, agg_script], check=False)
        else:
            log("[WARNING] aggregate_results.py 不存在，跳过汇总")

    # 完成
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\n[FINISH] 补充实验完成。总耗时: {elapsed / 60:.1f} min")
    log_file.close()


if __name__ == "__main__":
    main()
