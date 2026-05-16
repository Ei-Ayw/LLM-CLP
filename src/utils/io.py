"""
统一文件 I/O 工具集。

提供 JSON、Parquet 及目录管理的一致性读写接口。
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，如不存在则自动创建。

    Args:
        path: 目录路径。

    Returns:
        Path 对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """读取 JSON 文件。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的 JSON 字典。
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Union[str, Path], **kwargs) -> None:
    """将数据写入 JSON 文件。

    Args:
        data: 待序列化的数据。
        path: 输出路径。
        **kwargs: 传递给 json.dump 的额外参数。
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, **kwargs)


def read_parquet(path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """读取 Parquet 文件。

    Args:
        path: Parquet 文件路径。
        columns: 可选，指定读取的列名列表。

    Returns:
        DataFrame。
    """
    return pd.read_parquet(path, columns=columns)


def write_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """将 DataFrame 写入 Parquet 文件。

    Args:
        df: 待写入的 DataFrame。
        path: 输出路径。
        **kwargs: 传递给 df.to_parquet 的额外参数。
    """
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False, **kwargs)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """读取 JSONL（JSON Lines）文件。

    Args:
        path: JSONL 文件路径。

    Returns:
        每行对应一个字典的列表。
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """将记录列表写入 JSONL 文件。

    Args:
        records: 待写入的字典列表。
        path: 输出路径。
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")