import argparse
import runpy
import sys

from src.llm_clp.common.cli_config import apply_config_defaults


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2: unified training entry")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", dest="data_dir", type=str, default="data/causal_fair")
    parser.add_argument("--model", dest="model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--method", type=str, default="ours", choices=["ours", "vanilla", "ear", "getfair", "ccdf", "davani", "ramponi"])

    args = apply_config_defaults(parser)
    _, passthrough = parser.parse_known_args()
    passthrough = [x for x in passthrough if x not in {"--config", args.config}]
    forwarded = ["--data_dir", args.data_dir, "--model_name", args.model_name] + passthrough

    if args.method == "ours":
        sys.argv = ["train_causal_fair.py", *forwarded]
        runpy.run_module("src.llm_clp.train.train_causal_fair", run_name="__main__")
        return

    module_map = {
        "vanilla": "src.llm_clp.train.baselines.vanilla",
        "ear": "src.llm_clp.train.baselines.ear",
        "getfair": "src.llm_clp.train.baselines.getfair",
        "ccdf": "src.llm_clp.train.baselines.ccdf",
        "davani": "src.llm_clp.train.baselines.davani",
        "ramponi": "src.llm_clp.train.baselines.ramponi",
    }
    sys.argv = [f"{args.method}.py", *forwarded]
    runpy.run_module(module_map[args.method], run_name="__main__")


if __name__ == "__main__":
    main()
