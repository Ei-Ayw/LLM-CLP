"""
Unified file I/O utilities.

Provides consistent read/write for JSON, Parquet, and directory management.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON dict.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Union[str, Path], **kwargs) -> None:
    """Write data to a JSON file.

    Args:
        data: Data to serialize.
        path: Output path.
        **kwargs: Extra arguments passed to json.dump.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, **kwargs)


def read_parquet(path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a Parquet file.

    Args:
        path: Path to Parquet file.
        columns: Optional list of columns to read.

    Returns:
        DataFrame.
    """
    return pd.read_parquet(path, columns=columns)


def write_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Write a DataFrame to Parquet.

    Args:
        df: DataFrame to write.
        path: Output path.
        **kwargs: Extra arguments passed to df.to_parquet.
    """
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False, **kwargs)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL (JSON Lines) file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of dicts, one per line.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Write records to a JSONL file.

    Args:
        records: List of dicts to write.
        path: Output path.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")