from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
EVAL_DIR = OUTPUT_DIR / "eval"
PLOT_DIR = OUTPUT_DIR / "plots"


def ensure_output_dirs() -> None:
    """确保所有输出目录存在，不存在则创建。"""
    for directory in (OUTPUT_DIR, MODEL_DIR, LOG_DIR, EVAL_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = ["ROOT_DIR", "OUTPUT_DIR", "MODEL_DIR", "LOG_DIR", "EVAL_DIR", "PLOT_DIR", "ensure_output_dirs", "model_path", "log_path", "eval_path", "plot_path"]


def model_path(filename: str) -> str:
    """返回模型输出目录下的文件路径。"""
    ensure_output_dirs()
    return str(MODEL_DIR / filename)


def log_path(filename: str) -> str:
    """返回日志输出目录下的文件路径。"""
    ensure_output_dirs()
    return str(LOG_DIR / filename)


def eval_path(filename: str) -> str:
    """返回评估输出目录下的文件路径。"""
    ensure_output_dirs()
    return str(EVAL_DIR / filename)


def plot_path(filename: str) -> str:
    """返回图表输出目录下的文件路径。"""
    ensure_output_dirs()
    return str(PLOT_DIR / filename)
