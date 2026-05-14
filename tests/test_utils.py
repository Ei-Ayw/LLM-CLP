"""
Tests for utility functions.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.llm_clp.utils.io import (
    ensure_dir,
    read_json,
    write_json,
    read_parquet,
    write_parquet,
    read_jsonl,
    write_jsonl,
)


class TestIO:
    def test_ensure_dir(self, tmp_path):
        p = tmp_path / "a" / "b" / "c"
        result = ensure_dir(p)
        assert result.exists()
        assert result.is_dir()

    def test_json_roundtrip(self, tmp_path):
        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": True}}
        path = tmp_path / "test.json"
        write_json(data, path)
        loaded = read_json(path)
        assert loaded == data

    def test_parquet_roundtrip(self, tmp_path):
        df = pd.DataFrame({
            "text": ["hello", "world"],
            "label": [0, 1],
        })
        path = tmp_path / "test.parquet"
        write_parquet(df, path)
        loaded = read_parquet(path)
        pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df)

    def test_jsonl_roundtrip(self, tmp_path):
        records = [
            {"text": "hello", "cf": "world"},
            {"text": "foo", "cf": "bar"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert loaded == records


class TestSeed:
    def test_set_seed_deterministic(self):
        import random
        import torch

        from src.llm_clp.utils.seed import set_seed

        set_seed(42, deterministic=True)
        a = random.random()
        b = torch.randn(3)

        set_seed(42, deterministic=True)
        c = random.random()
        d = torch.randn(3)

        assert a == c
        np.testing.assert_array_equal(a for a in [a, c])
        # Same random sequence on second call
        assert a == c
