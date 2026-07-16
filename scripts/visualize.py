"""Generate training and endpoint-RUL plots from an existing run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.visualization import create_run_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in create_run_visualizations(args.run_dir, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
