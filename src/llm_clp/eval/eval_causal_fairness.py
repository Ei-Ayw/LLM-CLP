import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.llm_clp.eval.metrics import evaluate_causal_fairness
from src.llm_clp.models.classifier import DebertaV3CausalFair


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="因果公平性评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hatexplain")
    parser.add_argument("--cf_method", type=str, default="llm", choices=["swap", "llm"])
    parser.add_argument("--model_name", type=str, default=str(BASE_DIR / "models" / "deberta-v3-base"))
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data_dir", type=str, default=str(BASE_DIR / "data" / "causal_fair"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaV3CausalFair(args.model_name, num_classes=2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_df = pd.read_parquet(Path(args.data_dir) / f"{args.dataset}_test.parquet")

    cf_suffix = "cf_swap" if args.cf_method == "swap" else "cf_llm"
    cf_path = Path(args.data_dir) / f"{args.dataset}_test_{cf_suffix}.parquet"
    cf_df = pd.read_parquet(cf_path) if cf_path.exists() else None

    results = evaluate_causal_fairness(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        cf_df=cf_df,
        device=device,
        max_len=args.max_len,
        threshold=args.threshold,
    )

    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "_fairness.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
