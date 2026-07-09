"""Experiment 4: Business Evaluation.

Compares the decision engine's recommendations against simple baseline strategies
(current allocation, uniform allocation, proportional allocation).

Measures: expected revenue, expected ROAS, budget efficiency, risk, confidence.
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
    print("EXP04: Business Evaluation")
    print("=" * 60)

    data = ExperimentData(src_config)
    feature_df = data.get_feature_data()
    print(f"Loaded feature DataFrame: {feature_df.shape}")

    reporter = ExperimentReporter(config.output_dir)

    # Train forecaster
    forecaster = Forecaster(src_config)
    forecaster.fit(feature_df)
    forecast_result = forecaster.predict(feature_df)

    # Uncertainty
    uncertainty_engine = UncertaintyEngine(src_config)
    uncertainty_report = uncertainty_engine.compute(forecast_result)

    # Baselines
    baselines = extract_baselines(feature_df)

    # Decision engine
    decision_engine = DecisionEngine(src_config)
    optimization_report = decision_engine.analyze(
        forecast_result, uncertainty_report, baselines, feature_df
    )

    # Evaluate strategies
    total_budget = sum(bl.total_spend for bl in baselines.values())
    n_campaigns = len(baselines)

    # Strategy 1: Current allocation (baseline)
    current_revenue = sum(bl.total_revenue for bl in baselines.values())
    current_spend = total_budget
    current_roas = current_revenue / current_spend if current_spend > 0 else 0

    # Strategy 2: Uniform allocation
    uniform_spend = total_budget / n_campaigns if n_campaigns > 0 else 0
    uniform_budgets = {eid: uniform_spend for eid in baselines}
    uniform_revenue = sum(
        bl.historical_roas * uniform_budgets[eid]
        for eid, bl in baselines.items()
    )
    uniform_roas = uniform_revenue / total_budget if total_budget > 0 else 0

    # Strategy 3: Proportional to historical spend
    prop_budgets = {}
    for eid, bl in baselines.items():
        prop_budgets[eid] = total_budget * (bl.total_spend / current_spend) if current_spend > 0 else 0
    prop_revenue = sum(
        bl.historical_roas * prop_budgets[eid]
        for eid, bl in baselines.items()
    )
    prop_roas = prop_revenue / total_budget if total_budget > 0 else 0

    # Strategy 4: Optimizer recommendations
    opt_budgets = {r.entity_id: r.recommended_budget for r in optimization_report.recommendations}
    opt_revenue = 0
    for r in optimization_report.recommendations:
        bl = baselines.get(r.entity_id)
        if bl:
            opt_revenue += bl.historical_roas * r.recommended_budget
    opt_roas = opt_revenue / total_budget if total_budget > 0 else 0

    # Risk scores (coefficient of variation of budget distribution)
    def _risk_score(budgets_dict):
        vals = np.array(list(budgets_dict.values()))
        if vals.sum() == 0:
            return 1.0
        return float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 1.0

    # Budget efficiency: revenue per dollar weighted by concentration penalty
    def _efficiency(rev, spend, budgets_dict):
        if spend == 0:
            return 0.0
        conc = max(budgets_dict.values()) / spend if spend > 0 else 1.0
        penalty = 1.0 / (1.0 + conc)
        return rev / spend * penalty

    results = {
        "current": {
            "expected_revenue": round(current_revenue, 2),
            "expected_roas": round(current_roas, 4),
            "risk_score": round(_risk_score({eid: bl.total_spend for eid, bl in baselines.items()}), 4),
            "budget_efficiency": round(_efficiency(current_revenue, current_spend,
                                                    {eid: bl.total_spend for eid, bl in baselines.items()}), 4),
        },
        "uniform": {
            "expected_revenue": round(uniform_revenue, 2),
            "expected_roas": round(uniform_roas, 4),
            "risk_score": round(_risk_score(uniform_budgets), 4),
            "budget_efficiency": round(_efficiency(uniform_revenue, total_budget, uniform_budgets), 4),
        },
        "proportional": {
            "expected_revenue": round(prop_revenue, 2),
            "expected_roas": round(prop_roas, 4),
            "risk_score": round(_risk_score(prop_budgets), 4),
            "budget_efficiency": round(_efficiency(prop_revenue, total_budget, prop_budgets), 4),
        },
        "optimizer": {
            "expected_revenue": round(opt_revenue, 2),
            "expected_roas": round(opt_roas, 4),
            "risk_score": round(_risk_score(opt_budgets), 4),
            "budget_efficiency": round(_efficiency(opt_revenue, total_budget, opt_budgets), 4),
        },
    }

    reporter.save_metrics_table(results, "exp04_business")
    print("Saved: research/tables/exp04_business.csv")

    summary = {
        "experiment": "exp04_business_evaluation",
        "best_revenue_strategy": max(results, key=lambda s: results[s]["expected_revenue"]),
        "best_roas_strategy": max(results, key=lambda s: results[s]["expected_roas"]),
        "best_efficiency_strategy": max(results, key=lambda s: results[s]["budget_efficiency"]),
        "optimizer_revenue_uplift_pct": round(
            (results["optimizer"]["expected_revenue"] - results["current"]["expected_revenue"])
            / results["current"]["expected_revenue"] * 100, 2
        ) if results["current"]["expected_revenue"] > 0 else 0,
    }
    reporter.save_summary(summary, "exp04_business")

    print(f"\n  Strategy Comparison:")
    for s, r in results.items():
        print(f"    {s:15s} Rev={r['expected_revenue']:10.2f}  ROAS={r['expected_roas']:.2f}  "
              f"Risk={r['risk_score']:.2f}  Eff={r['budget_efficiency']:.2f}")

    print("\nEXP04 complete.\n")
    return results
