import argparse
import runpy
import sys

from src.llm_clp.common.cli_config import apply_config_defaults


def main() -> None:
    """主函数：统一可视化入口"""
    parser = argparse.ArgumentParser(description="步骤4：统一可视化入口")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--input", dest="input_path", type=str, default=None, help="输入文件路径")
    parser.add_argument("--type", type=str, default="performance", choices=["performance", "tsne"], help="可视化类型")

    args = apply_config_defaults(parser)
    _, passthrough = parser.parse_known_args()
    passthrough = [x for x in passthrough if x not in {"--config", args.config}]
    
    # 根据类型选择不同的可视化脚本
    if args.type == "tsne":
        sys.argv = ["feature_tsne.py", *passthrough]
        runpy.run_module("src.llm_clp.eval.plots.feature_tsne", run_name="__main__")
        return

    sys.argv = ["performance_summary.py", *passthrough]
    runpy.run_module("src.llm_clp.eval.plots.performance_summary", run_name="__main__")


if __name__ == "__main__":
    main()
