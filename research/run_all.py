#!/usr/bin/env python
"""Run all experiments sequentially and produce a summary report.

Usage: python -m research.run_all [--config config.yaml]
"""

import argparse
import importlib
import time
from pathlib import Path

from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig


EXPERIMENTS = [
    ("01", "research.experiments.exp01_forecast_benchmark", "Forecast Benchmark"),
    ("02", "research.experiments.exp02_feature_ablation", "Feature Ablation"),
    ("03", "research.experiments.exp03_uncertainty_calibration", "Uncertainty Calibration"),
    ("04", "research.experiments.exp04_business_evaluation", "Business Evaluation"),
    ("05", "research.experiments.exp05_sensitivity_analysis", "Sensitivity Analysis"),
    ("06", "research.experiments.exp06_failure_analysis", "Failure Analysis"),
    ("07", "research.experiments.exp07_optimization_validation", "Optimization Validation"),
    ("08", "research.experiments.exp08_complexity_evaluation", "Complexity Evaluation"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--skip", nargs="*", default=[], help="Experiment numbers to skip")
    args = parser.parse_args()

    src_config = SrcConfig(Path(args.config))
    exp_config = ExperimentConfig()
    results = {}
    total_time = 0.0

    print(f"\n{'=' * 60}")
    print("AIgnition Forecasting: Full Experiment Suite")
    print(f"{'=' * 60}\n")

    for exp_num, module_path, exp_name in EXPERIMENTS:
        if exp_num in args.skip:
            print(f"Skipping Experiment {exp_num}: {exp_name}")
            continue

        print(f"\n{'#' * 60}")
        print(f"# Experiment {exp_num}: {exp_name}")
        print(f"{'#' * 60}\n")

        t0 = time.time()
        try:
            module = importlib.import_module(module_path)
            result = module.run(exp_config, src_config)
            results[exp_num] = {"status": "success", "name": exp_name}
        except Exception as e:
            print(f"Experiment {exp_num} FAILED: {e}")
            results[exp_num] = {"status": "failed", "name": exp_name, "error": str(e)}
            import traceback
            traceback.print_exc()

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"Elapsed: {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    successes = sum(1 for r in results.values() if r["status"] == "success")
    failures = sum(1 for r in results.values() if r["status"] == "failed")
    print(f"  Total: {len(EXPERIMENTS) - len(args.skip)}  "
          f"Success: {successes}  Failed: {failures}  "
          f"Total time: {total_time:.0f}s")
    for exp_num, r in sorted(results.items()):
        status_tag = "[OK]" if r["status"] == "success" else "[FAIL]"
        print(f"  {status_tag} Exp {exp_num}: {r['name']}")

    # Generate consolidated report
    _generate_consolidated_report(exp_config, results)


def _generate_consolidated_report(config: ExperimentConfig, results: dict):
    """Generate a consolidated markdown report."""
    from research.core.reporting import ExperimentReporter

    reporter = ExperimentReporter(config.output_dir)

    sections = []
    sections.append("# AIgnition Forecasting: Experimental Validation Report\n")
    sections.append("## Executive Summary\n")
    sections.append(f"**Experiments run**: {len(results)}\n")
    sections.append(f"**Status**: {sum(1 for r in results.values() if r['status'] == 'success')} succeeded, "
                    f"{sum(1 for r in results.values() if r['status'] == 'failed')} failed\n")

    # Read individual summaries
    summary_dir = config.output_dir / "reports"
    for exp_num in sorted(results.keys()):
        summary_path = summary_dir / f"exp0{exp_num}_*_summary.json"
        from pathlib import Path as P
        matching = list(P(config.output_dir).glob(f"reports/exp0{exp_num}_*_summary.json"))
        if matching:
            import json
            with open(matching[0]) as f:
                data = json.load(f)
            sections.append(f"\n## Experiment {exp_num}: {results[exp_num]['name']}\n")
            sections.append(f"```json\n{json.dumps(data, indent=2)}\n```\n")

    report_content = "\n".join(sections)
    reporter.write_report_section("consolidated_report", report_content)
    print(f"Consolidated report saved: research/reports/consolidated_report.md")


if __name__ == "__main__":
    main()
