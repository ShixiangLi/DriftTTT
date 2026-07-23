from __future__ import annotations

import argparse

from utils.visualization import visualize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots for an experiment")
    parser.add_argument("--run-dir", required=True, help="Experiment output directory")
    arguments = parser.parse_args()
    visualize_run(arguments.run_dir)


if __name__ == "__main__":
    main()
