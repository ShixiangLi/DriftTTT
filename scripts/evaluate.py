from __future__ import annotations

import argparse

from utils.config import load_config
from utils.engine import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an RUL Transformer checkpoint"
    )
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    arguments = parser.parse_args()
    evaluate_experiment(load_config(arguments.config))


if __name__ == "__main__":
    main()
