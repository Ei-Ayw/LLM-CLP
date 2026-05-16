"""
统一日志配置模块。

支持同时向控制台和文件输出结构化日志。
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Union

from .io import ensure_dir


def setup_logger(
    name: str = "llm_clp",
    output_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """配置一个同时输出到控制台和可选日志文件的 logger。

    Args:
        name: Logger 名称。
        output_dir: 日志文件输出目录。若为 None，则仅输出到控制台。
        level: 日志级别（如 logging.INFO、logging.DEBUG）。
        format_str: 自定义格式字符串。默认为：
            "[%(asctime)s] [%(levelname)s] %(message)s"

    Returns:
        已配置的 logger 对象。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    if format_str is None:
        format_str = "[%(asctime)s] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")

    # 控制台处理器
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件处理器
    if output_dir is not None:
        ensure_dir(output_dir)
        log_path = Path(output_dir) / "run.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 防止日志传播到根 logger
    logger.propagate = False

    return logger