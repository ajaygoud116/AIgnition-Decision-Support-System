#!/usr/bin/env python
"""Run a single experiment by name.

Usage: python -m research.run_one --exp 01
       python -m research.run_one --exp 02 --config research_config.yaml
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig


EXPERIMENTS = {
    "01": "research.experiments.exp01_forecast_benchmark",
    "02": "research.experiments.exp02_feature_ablation",
    "03": "research.experiments.exp03_uncertainty_calibration",
    "04": "research.experiments.exp04_business_evaluation",
    "05": "research.experiments.exp05_sensitivity_analysis",
    "06": "research.experiments.exp06_failure_analysis",
    "07": "research.experiments.exp07_optimization_validation",
    "08": "research.experiments.exp08_complexity_evaluation",
}

EXPERIMENT_NAMES = {
    "01": "Forecast Benchmark",
    "02": "Feature Ablation",
    "03": "Uncertainty Calibration",
    "04": "Business Evaluation",
    "05": "Sensitivity Analysis",
    "06": "Failure Analysis",
    "07": "Optimization Validation",
    "08": "Complexity Evaluation",
}


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--exp", required=True, help="Experiment number (01-08)")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    exp_num = args.exp.zfill(2)
    if exp_num not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_num}. Choose from: {', '.join(sorted(EXPERIMENTS.keys()))}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Running Experiment {exp_num}: {EXPERIMENT_NAMES[exp_num]}")
    print(f"{'=' * 60}\n")

    # Load configs
    src_config = SrcConfig(Path(args.config))
    exp_config = ExperimentConfig()

    # Dynamically import and run
    import importlib
    module = importlib.import_module(EXPERIMENTS[exp_num])
    result = module.run(exp_config, src_config)

    print(f"Experiment {exp_num} completed successfully.")
    return result


if __name__ == "__main__":
    main()
