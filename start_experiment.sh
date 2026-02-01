#!/bin/bash
# 实验一键启动脚本: start_experiment.sh

echo ">>> [1/3] 正在清理残留进程 (run_experiments.py, train_*.py)..."
pkill -9 -f "run_experiments.py|train_"
sleep 2

echo ">>> [2/3] 启动全量实验流程 (Group 1-4 + 消融实验)..."
echo ">>> 日志实时输出至: experiment_full_final.log"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# 强制使用当前项目下的预训练模型缓存
export HF_HOME="$(pwd)/pretrained_models"
export TRANSFORMERS_OFFLINE=1

nohup python run_experiments.py > experiment_full_final.log 2>&1 &

echo ">>> [3/3] 启动成功！"
echo ">>> 你可以使用 'tail -f experiment_full_final.log' 实时查看进度。"
echo ">>> 正在显示初始日志输出..."
sleep 1
tail -n 20 experiment_full_final.log
