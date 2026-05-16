from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at top level")
    return data


def apply_config_defaults(parser: argparse.ArgumentParser, argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    if pre_args.config:
        cfg = _load_yaml(pre_args.config)
        aliases = {
            "data": "data_dir",
            "model": "model_name",
            "ckpt": "checkpoint",
            "input": "input_path",
        }
        normalized = {}
        for key, value in cfg.items():
            normalized[aliases.get(key, key)] = value
        parser.set_defaults(**normalized)

    args, _ = parser.parse_known_args(argv)
    return args


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
