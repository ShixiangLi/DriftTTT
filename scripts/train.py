from __future__ import annotations

import argparse

from utils.config import load_config
from utils.engine import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an RUL Transformer"
    )
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    arguments = parser.parse_args()
    train_experiment(load_config(arguments.config))


if __name__ == "__main__":
    main()
