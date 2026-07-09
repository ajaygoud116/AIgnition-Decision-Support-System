"""Experiment 7: Optimization Validation.

Tests whether the BudgetOptimizer's recommendations are actually optimal
by comparing against equal allocation, proportional allocation, and greedy allocation.

Uses a score-based utility function as the evaluation metric.
"""

import numpy as np
import pandas as pd

from src.decision.engine import DecisionEngine
from src.forecasting.forecaster import Forecaster
from src.simulation.baselines import extract_baselines
from src.uncertainty.engine import UncertaintyEngine
from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig
from research.core.data import ExperimentData
from research.core.reporting import ExperimentReporter


def run(config: ExperimentConfig, src_config: SrcConfig) -> dict:
    print("=" * 60)
    print("EXP07: Optimization Validation")
    print("=" * 60)

    data = ExperimentData(src_config)
    feature_df = data.get_feature_data()
    print(f"Loaded feature DataFrame: {feature_df.shape}")

    reporter = ExperimentReporter(config.output_dir)

    forecaster = Forecaster(src_config)
    forecaster.fit(feature_df)
    forecast_result = forecaster.predict(feature_df)

    uncertainty_engine = UncertaintyEngine(src_config)
    uncertainty_report = uncertainty_engine.compute(forecast_result)

    baselines = extract_baselines(feature_df)

    decision_engine = DecisionEngine(src_config)
    optimization_report = decision_engine.analyze(
        forecast_result, uncertainty_report, baselines, feature_df
    )

    total_budget = sum(bl.total_spend for bl in baselines.values())
    n = len(baselines)
    if n == 0:
        return {}

    # ---- Define utility function ----
    def allocation_utility(budgets_dict, baselines_dict):
        """Utility = sum of (ROAS * confidence * budget) with concentration penalty."""
        total = 0.0
        for eid, bl in baselines_dict.items():
            b = budgets_dict.get(eid, 0)
            unc = [e for e in uncertainty_report.entities if e.entity_id == eid]
            conf = unc[0].confidence_score if unc else 0.5
            roas = bl.historical_roas
            total += roas * conf * b

        # Concentration penalty: prefer balanced allocations
        vals = np.array(list(budgets_dict.values()))
        if vals.sum() > 0 and len(vals) > 1:
            gini = 1.0 - (vals / vals.sum()).var() * len(vals)
            total *= (1.0 - 0.05 * gini)  # up to 5% penalty for concentration
        return total

    # Strategy 1: Current allocation
    current_budgets = {eid: bl.total_spend for eid, bl in baselines.items()}
    current_utility = allocation_utility(current_budgets, baselines)

    # Strategy 2: Equal allocation
    equal_budgets = {eid: total_budget / n for eid in baselines}
    equal_utility = allocation_utility(equal_budgets, baselines)

    # Strategy 3: Proportional to current
    total_current = sum(current_budgets.values())
    prop_budgets = {}
    for eid, bl in baselines.items():
        prop_budgets[eid] = total_budget * (bl.total_spend / total_current) if total_current > 0 else 0
    prop_utility = allocation_utility(prop_budgets, baselines)

    # Strategy 4: Greedy (all budget to highest-ROAS campaign)
    best_campaign = max(baselines.items(), key=lambda x: x[1].historical_roas)[0]
    greedy_budgets = {eid: (total_budget if eid == best_campaign else 0) for eid in baselines}
    greedy_utility = allocation_utility(greedy_budgets, baselines)

    # Strategy 5: Optimizer
    opt_budgets = {r.entity_id: r.recommended_budget for r in optimization_report.recommendations}
    opt_utility = allocation_utility(opt_budgets, baselines)

    # Compute revenues per strategy
    def compute_revenue(budgets_dict):
        rev = 0.0
        for eid, bl in baselines.items():
            rev += bl.historical_roas * budgets_dict.get(eid, 0)
        return rev

    results = {
        "equal": {
            "utility": round(equal_utility, 2),
            "expected_revenue": round(compute_revenue(equal_budgets), 2),
            "expected_roas": round(compute_revenue(equal_budgets) / total_budget, 4) if total_budget > 0 else 0,
            "concentration": round(max(equal_budgets.values()) / total_budget, 4) if total_budget > 0 else 0,
            "n_active_campaigns": sum(1 for b in equal_budgets.values() if b > 0),
        },
        "proportional": {
            "utility": round(prop_utility, 2),
            "expected_revenue": round(compute_revenue(prop_budgets), 2),
            "expected_roas": round(compute_revenue(prop_budgets) / total_budget, 4) if total_budget > 0 else 0,
            "concentration": round(max(prop_budgets.values()) / total_budget, 4) if total_budget > 0 else 0,
            "n_active_campaigns": sum(1 for b in prop_budgets.values() if b > 0),
        },
        "current": {
            "utility": round(current_utility, 2),
            "expected_revenue": round(compute_revenue(current_budgets), 2),
            "expected_roas": round(compute_revenue(current_budgets) / total_budget, 4) if total_budget > 0 else 0,
            "concentration": round(max(current_budgets.values()) / total_budget, 4) if total_budget > 0 else 0,
            "n_active_campaigns": sum(1 for b in current_budgets.values() if b > 0),
        },
        "greedy": {
            "utility": round(greedy_utility, 2),
            "expected_revenue": round(compute_revenue(greedy_budgets), 2),
            "expected_roas": round(compute_revenue(greedy_budgets) / total_budget, 4) if total_budget > 0 else 0,
            "concentration": round(max(greedy_budgets.values()) / total_budget, 4) if total_budget > 0 else 0,
            "n_active_campaigns": sum(1 for b in greedy_budgets.values() if b > 0),
        },
        "optimizer": {
            "utility": round(opt_utility, 2),
            "expected_revenue": round(compute_revenue(opt_budgets), 2),
            "expected_roas": round(compute_revenue(opt_budgets) / total_budget, 4) if total_budget > 0 else 0,
            "concentration": round(max(opt_budgets.values()) / total_budget, 4) if total_budget > 0 else 0,
            "n_active_campaigns": sum(1 for b in opt_budgets.values() if b > 0),
        },
    }

    reporter.save_metrics_table(results, "exp07_optimization")
    print("Saved: research/tables/exp07_optimization.csv")

    summary = {
        "experiment": "exp07_optimization_validation",
        "best_utility_strategy": max(results, key=lambda s: results[s]["utility"]),
        "optimizer_utility": results["optimizer"]["utility"],
        "current_utility": results["current"]["utility"],
        "optimizer_improvement_pct": round(
            (results["optimizer"]["utility"] - results["current"]["utility"])
            / results["current"]["utility"] * 100, 2
        ) if results["current"]["utility"] > 0 else 0,
        "optimizer_beats_equal": results["optimizer"]["utility"] > results["equal"]["utility"],
        "optimizer_beats_greedy": results["optimizer"]["utility"] > results["greedy"]["utility"],
    }
    reporter.save_summary(summary, "exp07_optimization")

    print(f"\n  Optimization Strategy Comparison:")
    for s, r in results.items():
        print(f"    {s:15s} Utility={r['utility']:8.2f}  Revenue={r['expected_revenue']:10.2f}  "
              f"ROAS={r['expected_roas']:.2f}  Active={r['n_active_campaigns']}")

    print("\nEXP07 complete.\n")
    return results
