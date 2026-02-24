"""
=============================================================================
### 统一路径配置：path_config.py ###
设计说明：
集中管理项目的输出目录结构，确保所有脚本使用一致的路径。
=============================================================================
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 输出目录结构 (src_result/)
# =============================================================================
# src_result/
# ├── models/      # 模型权重 (.pth, .joblib, .pkl)
# ├── logs/        # 训练日志 (loss.json, loss.png)
# ├── eval/        # 评估报告 (metrics.json, threshold_scan.png)
# └── viz/         # 可视化图表 (t-SNE, performance summary)
# =============================================================================

RESULT_DIR = os.path.join(BASE_DIR, "src_result")
MODEL_DIR = os.path.join(RESULT_DIR, "models")
LOG_DIR = os.path.join(RESULT_DIR, "logs")
EVAL_DIR = os.path.join(RESULT_DIR, "eval")
VIZ_DIR = os.path.join(RESULT_DIR, "viz")

def ensure_dirs():
    """确保所有输出目录存在"""
    for d in [MODEL_DIR, LOG_DIR, EVAL_DIR, VIZ_DIR]:
        os.makedirs(d, exist_ok=True)

# 便捷函数
def get_model_path(filename):
    ensure_dirs()
    return os.path.join(MODEL_DIR, filename)

def get_log_path(filename):
    ensure_dirs()
    return os.path.join(LOG_DIR, filename)

def get_eval_path(filename):
    ensure_dirs()
    return os.path.join(EVAL_DIR, filename)

def get_viz_path(filename):
    ensure_dirs()
    return os.path.join(VIZ_DIR, filename)
