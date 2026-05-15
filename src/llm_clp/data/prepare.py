import argparse
from pathlib import Path

import pandas as pd

from src.llm_clp.common.cli_config import apply_config_defaults
from src.llm_clp.utils.io import ensure_dir


def main() -> None:
    """主函数：下载并准备数据集"""
    parser = argparse.ArgumentParser(description="步骤1：数据下载与准备")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--data", dest="data_dir", type=str, default="data/causal_fair", help="数据输出目录")
    parser.add_argument("--dataset", type=str, default="hatexplain", choices=["hatexplain", "toxigen", "dynahate"], help="数据集名称")
    parser.add_argument("--train", type=str, default=None, help="原始训练集 parquet 路径（可选）")
    parser.add_argument("--val", type=str, default=None, help="原始验证集 parquet 路径（可选）")
    parser.add_argument("--test", type=str, default=None, help="原始测试集 parquet 路径（可选）")

    args = apply_config_defaults(parser)
    out_dir = Path(args.data_dir)
    ensure_dir(out_dir)

    for split, source in (("train", args.train), ("val", args.val), ("test", args.test)):
        target = out_dir / f"{args.dataset}_{split}.parquet"
        if source:
            df = pd.read_parquet(source)
            df.to_parquet(target, index=False)
            print(f"已准备 {split}: {target}")
        elif target.exists():
            print(f"已存在 {split}: {target}")
        else:
            print(f"缺失 {split}: 请提供 --{split} 或将文件放置于 {target}")


if __name__ == "__main__":
    main()
