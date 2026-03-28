"""Run the full experiment pipeline for a dataset.

Steps: build_db -> build_null -> validate_null -> run_retrieval -> run_eval

Usage: python -m experiments.scripts.run_all --config experiments/configs/amazon_compound.yaml
       python -m experiments.scripts.run_all --config ... --from run_retrieval
       python -m experiments.scripts.run_all --config ... --force
"""

import argparse
import subprocess
import sys


STEPS = ["build_db", "build_null", "validate_null", "run_retrieval", "run_eval"]


def run_step(step: str, config_path: str, force: bool) -> int:
    """Run a single pipeline step as a subprocess. Returns exit code."""
    cmd = [
        sys.executable, "-m", f"experiments.scripts.{step}",
        "--config", config_path,
    ]
    if force:
        cmd.append("--force")

    print(f"\n{'=' * 70}")
    print(f"  Step: {step}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Force rerun all steps")
    parser.add_argument(
        "--from", dest="start_from", default=None, choices=STEPS,
        help="Start from a specific step (skip earlier steps)",
    )
    args = parser.parse_args()

    steps = STEPS
    if args.start_from:
        start_idx = STEPS.index(args.start_from)
        steps = STEPS[start_idx:]

    print(f"Running pipeline: {' -> '.join(steps)}")
    print(f"Config: {args.config}")

    for step in steps:
        exit_code = run_step(step, args.config, args.force)

        if exit_code != 0:
            if step == "validate_null":
                print(f"\n{'!' * 70}")
                print(f"  VALIDATION FAILED — stopping pipeline.")
                print(f"  Fix the null distribution before proceeding.")
                print(f"{'!' * 70}")
            else:
                print(f"\n  Step '{step}' failed with exit code {exit_code}.")
            sys.exit(exit_code)

    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
