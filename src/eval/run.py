import argparse
import runpy
import sys

from src.common.cli_config import apply_config_defaults


def main() -> None:
    parser = argparse.ArgumentParser(description="Step3: unified evaluation entry")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", dest="data_dir", type=str, default="data/causal_fair")
    parser.add_argument("--model", dest="model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--ckpt", dest="checkpoint", type=str, default=None)
    parser.add_argument("--method", type=str, default="ours", choices=["ours", "vanilla", "ear", "getfair", "ccdf", "davani", "ramponi"])

    args = apply_config_defaults(parser)
    if not args.checkpoint:
        raise SystemExit("--ckpt is required (or provide checkpoint in --config)")
    _, passthrough = parser.parse_known_args()
    passthrough = [x for x in passthrough if x not in {"--config", args.config}]

    if args.method == "ours":
        sys.argv = [
            "eval_causal_fairness.py",
            "--checkpoint", args.checkpoint,
            "--data_dir", args.data_dir,
            "--model_name", args.model_name,
            *passthrough,
        ]
        runpy.run_module("src.eval.eval_causal_fairness", run_name="__main__")
        return

    sys.argv = [
        "evaluate_all.py",
        "--method", args.method,
        "--checkpoint", args.checkpoint,
        "--data_dir", args.data_dir,
        "--model_name", args.model_name,
        *passthrough,
    ]
    runpy.run_module("src.eval.baselines.evaluate_all", run_name="__main__")


if __name__ == "__main__":
    main()
