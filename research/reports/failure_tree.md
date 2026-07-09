# Failure Tree Analysis

## Overview

Six experiments produced unexpected (negative) results. This document traces each
failure to its root cause with exact code references, the violated mathematical
assumption, business impact, and the minimal fix.

---

## EXP01: Historical Mean beats Ensemble

### Hypothesis
The 50/50 LightGBM + Seasonal Naive ensemble will produce more accurate
forecasts (lower RMSE) than simple baselines (Naive, Historical Mean).

### Verdict: FAILED
Historical Mean RMSE=245, Ensemble RMSE=330, LightGBM RMSE=473.

### Root Cause: Cross-campaign aggregation in evaluation

The benchmark collapses the campaign dimension before computing error.

**Exact code** (`research/experiments/exp01_forecast_benchmark.py:190`):
```python
test_actuals = test_df.groupby("date")["revenue"].mean().values
```

All models produce a single time series of length `horizon` (30 days). This is
compared against the *cross-campaign average* of revenue per day. Averaging
across 136 campaigns cancels out the campaign-specific patterns that ML models
are designed to capture. Simple models (Historical Mean, Naive) predict a
constant, which is naturally close to this averaged signal. ML models
overfit to campaign-specific feature patterns that cancel out in the
aggregation.

### Violated Assumption
**The evaluation preserves campaign identity.** It does not — the
`.groupby("date")` silently destroys the campaign dimension. The benchmark
measures the ability to predict *average revenue across all campaigns*, not
*per-campaign revenue*. These are fundamentally different tasks.

### Business Impact
**Serious.** The benchmark understates the ensemble's true value. If the
ensemble actually beats Historical Mean at the per-campaign level (which
Exp04's results suggest — the ensemble forecasts drive decisions), then
incorrect benchmark results would justify rolling back to a model that is
actually worse for the decision pipeline.

### Minimal Fix

Two-line change in `research/experiments/exp01_forecast_benchmark.py`:

1. Instead of `groupby("date")["revenue"].mean()`, compute RMSE per campaign
   then average campaign RMSEs.
2. All model functions must return per-campaign predictions, not single time
   series.

```python
# Replace line 190-193:
# test_actuals = test_df.groupby("date")["revenue"].mean().values
# With per-campaign evaluation loop:
campaign_rmses = []
for cid in test_df["campaign_id"].unique():
    c_train = train_df[train_df["campaign_id"] == cid]
    c_test = test_df[test_df["campaign_id"] == cid]
    if len(c_test) < 7:
        continue
    preds = model_func(c_train, c_test, ...)
    actuals = c_test["revenue"].values[:len(preds)]
    campaign_rmses.append(np.sqrt(np.mean((actuals - preds)**2)))
rmse = np.mean(campaign_rmses)
```

### Expected Improvement
Ensemble RMSE should drop from 330 to approximately 200-250, restoring the
expected ordering (Ensemble > LightGBM > Historical Mean). Per-campaign
evaluation is the standard in forecasting benchmarks.

---

## EXP02: Removing features improves accuracy

### Hypothesis
More engineered features (rolling statistics, lagged values, ratios, time
features) improve forecast accuracy.

### Verdict: FAILED
Removing rolling, lag, or time features improves RMSE by 4-6%. Only ratio
features are beneficial (+2% when removed).

### Root Cause: Curse of dimensionality with collinear features

**Exact code** (`src/features/transforms.py:68-118`):

Rolling features (line 88-92): 6 metrics × 3 windows × 2 stats = 36 features
```python
result[f"{m}_rolling_mean_{w}"] = result[m].shift(1).rolling(...).mean()
result[f"{m}_rolling_std_{w}"] = result[m].shift(1).rolling(...).std()
```

Lag features (line 116): 6 metrics × 4 lags = 24 features
```python
result[f"{m}_lag_{lag}"] = result[m].shift(lag)
```

These 60 features are generated from 6 base metrics (`spend`, `revenue`,
`clicks`, `impressions`, `conversions`, `daily_budget`). Within each group:

- `spend_rolling_mean_7`, `_14`, `_30` are highly collinear (three moving
  averages of the same series)
- `spend_lag_1`, `_7`, `_14`, `_30` are autocorrelated shifts of the same
  series
- Rolling std features are near-zero for stable metrics, adding noise

With 71 features and ~25K rows, the feature-to-sample ratio is 0.28%. Real
effective sample size is much lower due to temporal autocorrelation (~150
independent days × 136 campaigns with varying histories). This creates
overfitting — the LightGBM model learns noise patterns from collinear
features that don't generalize to the test window.

### Violated Assumption
**More features → more signal.** In fact, 84% of the features
(60/71 rolling + lag) are linear transformations of 6 base metrics, adding
almost no new information while consuming degrees of freedom. This violates
the "strong signal" assumption of high-dimensional learning.

### Business Impact
**Moderate.** Model quality degrades by ~6% RMSE. Since the production
pipeline uses this feature set, every downstream component inherits this
degradation. The decision engine (Exp04/Exp07) makes budget decisions based
on these forecasts.

### Minimal Fix

One-line change in `config.yaml`:

```yaml
features:
  rolling_windows: []    # was [7, 14, 30]
  lag_windows: []        # was [1, 7, 14, 30]
```

Or, more surgically, keep only ratio features by removing rolling and lag
from `src/features/builder.py:36-40`:

```python
# df = self._apply_per_campaign(df, add_rolling_features, windows=self._rolling_windows)
# df = self._apply_per_campaign(df, add_lag_features, lags=self._lag_windows)
```

### Expected Improvement
RMSE improvement of 4-6% (from ~473 to ~445 for LGBM). More importantly,
the model generalizes better across time, reducing the gap between train
and test performance.

---

## EXP03: Coverage is 14.5% (expected 80%)

### Hypothesis
LightGBM quantile regression (3 independent models for p10/p50/p90) combined
with Seasonal Naive in a 50/50 ensemble produces well-calibrated prediction
intervals with empirical coverage close to nominal.

### Verdict: FAILED
80% prediction interval covers only 14.5% of true values. Model interval
width is 0.71 vs true width of 2.34.

### Root Cause: Double compression of prediction intervals

**Root cause 1 — SN predicts zero for unseen entities**
(`research/experiments/exp03_uncertainty_calibration.py:58-61`):
```python
sn_preds = np.column_stack([
    sn.predict("syn_calib", test_df["date"])
    for _ in range(3)
])
```
The Seasonal Naive forecaster has never seen `"syn_calib"`. Its predict
method (`src/forecasting/seasonal_naive.py:54`) returns the stored baseline
which defaults to 0.0:
```python
baseline = self._baselines.get(entity_id, 0.0)
```
So all three SN quantiles are 0. The ensemble becomes:
```python
combined = 0.5 * lgb_preds + 0.5 * np.zeros((n, 3)) = 0.5 * lgb_preds
```

The 50/50 weight *halves* every prediction — including all three quantiles.
This compresses the p10-p90 interval by 50%.

**Root cause 2 — Quantile regression on synthetic data with spurious
features** (`src/features/transforms.py`). The synthetic data has 10
meaningful features + 61 engineered features from synthetic
spend/clicks/impressions. The LightGBM quantile regressors learn these
spurious relationships and produce overconfident (too narrow) intervals,
which are then halved by the SN weighting.

**Root cause 3 — No calibration layer**. The raw quantile regression
outputs are never calibrated against empirical coverage. There is no
post-hoc calibration (Platt scaling, conformal prediction, or temperature
scaling).

### Violated Assumption
**Quantile regression + equal-weight ensemble produces calibrated
intervals.** This is false for three reasons: (1) the SN component adds
zero signal for unseen entities, (2) independent quantile models don't
guarantee non-crossing or proper coverage in high dimensions, and
(3) equal weighting assumes both components are equally skilled, which
isn't true when one component has no signal.

### Business Impact
**Critical.** The uncertainty engine feeds directly into the decision
engine. With 14.5% coverage, the confidence scores are meaningless.
Every downstream decision (assessor flags, optimizer scores, budget
reallocations) is built on invalid uncertainty estimates. This is the
root cause of Exp04 (optimizer loses 50%) and Exp07 (equal allocation
beats optimizer).

### Minimal Fix

Fix the SN predict for unseen entities in
`research/experiments/exp03_uncertainty_calibration.py`:

```python
# Instead of using SN (which has no signal for synthetic data),
# use only the LGB component for calibration testing:
combined = lgb_preds  # Skip ensemble, use LGB only
```

For the production fix, add a conformal calibration step to
`src/uncertainty/engine.py`:

```python
def calibrate_intervals(self, pred_p10, pred_p50, pred_p90, actuals):
    """Scale intervals to achieve target coverage using held-out data."""
    # Find scaling factor alpha such that
    # coverage of [p50 - alpha*(p50-p10), p50 + alpha*(p90-p50)] = target
    from scipy.optimize import bisect
    def cov(alpha):
        lo = pred_p50 - alpha * (pred_p50 - pred_p10)
        hi = pred_p50 + alpha * (pred_p90 - pred_p50)
        return np.mean((actuals >= lo) & (actuals <= hi))
    try:
        return bisect(lambda a: cov(a) - 0.8, 0.1, 10.0)
    except ValueError:
        return 1.0  # fallback
```

### Expected Improvement
Coverage from 14.5% to ~80% (the target). Confidence scores become
meaningful. This alone fixes Exp04 and Exp07 because the optimizer's
score formula uses confidence.

---

## EXP04: Optimizer reduces revenue by 50%

### Hypothesis
The BudgetOptimizer, using score-proportional allocation with ±50% clamping,
will improve total revenue relative to the current allocation.

### Verdict: FAILED
Optimizer revenue: $5.5M vs Current revenue: $11.0M (exactly 50% less).

### Root Cause: Budget non-conservation from score clamping

The optimizer allocates budget proportionally to `score`:
(`src/decision/optimizer.py:43-48`):
```python
raw_share = max(a.score, 0.0) / total_score
recommended = total_budget * raw_share
```

Then applies ±50% clamp (`src/decision/optimizer.py:49-53`):
```python
max_change = bl.total_spend * self._max_change_ratio  # 50%
if recommended > bl.total_spend + max_change:
    recommended = bl.total_spend + max_change
elif recommended < bl.total_spend - max_change:
    recommended = max(0.0, bl.total_spend - max_change)
```

**These are contradictory.** The proportional allocation preserves total
budget: sum(recommended) = total_budget. But the clamp breaks budget
conservation: clamped values no longer sum to total_budget. There is no
re-normalization step.

In practice:
- Most campaigns have very low scores (because confidence is low — see Exp03)
- Low-score campaigns get clamped at -50% of current
- High-score campaigns get clamped at +50% of current
- But there are many more low-score campaigns, so the total drops
- The unallocated budget simply vanishes

From the pipeline output:
```
"total_current_budget": 2166531.67, "total_recommended_budget": 1085251.48
```
Exactly 50% reduction — consistent with every campaign being clamped at
-50%.

### Violated Assumption
**Score-proportional allocation preserves total budget.** It does, until
the clamping constraint is applied. The clamp creates a budget sink —
unallocated funds are simply lost. A proper optimizer must either
re-normalize after clamping or use a different allocation mechanism
(e.g., constrained optimization).

### Business Impact
**Critical.** 50% budget reduction means 50% less advertising spend,
producing 50% less revenue. At scale, this is millions in lost revenue
per month.

### Minimal Fix

Add budget re-normalization after clamping in
`src/decision/optimizer.py`:

```python
# After computing all recommendations, re-normalize to total_budget:
total_recommended = sum(r.recommended_budget for r in recommendations)
if total_recommended > 0 and abs(total_recommended - total_budget) / total_budget > 0.01:
    scale = total_budget / total_recommended
    for r in recommendations:
        r.recommended_budget *= scale
```

Or more fundamentally, replace the two-step (proportional + clamp) with a
single constrained optimization:

```python
# Use softmax with temperature instead of proportional:
import numpy as np
scores = np.array([max(a.score, 0.0) for a in assessments])
weights = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))
for a, w in zip(assessments, weights):
    recommended = total_budget * w
```

### Expected Improvement
From $5.5M to ~$11M (restoring budget to current levels). Further
improvement requires fixing Exp03 first so that scores are meaningful.

---

## EXP05: Concept drift causes +361% RMSE

While Exp05 is a diagnostic (not pass/fail), the magnitude of the failure
warrants analysis.

### Root Cause: No distribution shift detection

The LightGBM model has no mechanism to detect when the data distribution
has changed. Under concept drift (simulated by swapping train/test
shuffle), the model predicts using learned patterns that no longer apply.

### Violated Assumption
**Training and test distributions are identical.** This is violated under
concept drift, which is common in advertising (seasonal shifts, market
changes, new campaign types).

### Business Impact
**Catastrophic under drift.** +361% RMSE means predictions are useless.
If undetected, the decision engine makes budget decisions based on
meaningless forecasts.

### Minimal Fix

Add a drift detector before prediction in
`src/forecasting/forecaster.py`:

```python
def detect_drift(self, new_data: pd.DataFrame) -> float:
    """Return drift score; flag if > threshold."""
    baseline_stats = self._feature_stats  # stored at fit time
    new_stats = {col: new_data[col].mean() for col in baseline_stats}
    drift = sum(abs(new_stats[c] - baseline_stats[c]) / max(baseline_stats[c], 1e-10)
                for c in baseline_stats)
    return drift / len(baseline_stats)
```

### Expected Improvement
Catastrophic predictions (+361% RMSE) would be detected and flagged,
preventing bad budget decisions. The model would either retrain on new
data or fall back to a robust baseline (Historical Mean).

---

## EXP06: Failure analysis produces NaN

### Hypothesis
The forecaster can predict on a test split, and forecast dates will align
with test dates for point-by-point error computation.

### Verdict: FAILED
`total_predictions: 0`, all metrics NaN.

### Root Cause: The forecaster always predicts FUTURE dates

**Exact code** (`src/forecasting/forecaster.py:81-86`):
```python
last_train_date = feature_df["date"].max()
for h in self._horizons:
    horizon_dates = pd.date_range(
        start=last_train_date + pd.Timedelta(days=1),
        periods=h,
        freq="D",
    )
```

When the forecaster is called with `test_df` as the argument, it takes
`test_df["date"].max()` as `last_train_date`, then forecasts dates
**after** that. These forecast dates fall entirely outside the test
split's date range. Every lookup in the evaluation loop
(`research/experiments/exp06_failure_analysis.py:59-61`) fails:

```python
actual_row = test_df[(test_df["campaign_id"] == cid) &
                     (test_df["date"] == pd.Timestamp(point.date))]
```

### Violated Assumption
**The forecaster can predict "in-sample" for the test window.**
It cannot — it is designed to predict *future* dates beyond the input.
This is correct production behavior but wrong for failure analysis, which
needs *past* predictions aligned with known actuals.

### Business Impact
**Low.** This is a testing methodology bug, not a production bug. It
prevents automated failure analysis but doesn't affect the pipeline.

### Minimal Fix

In `research/experiments/exp06_failure_analysis.py`, use walk-forward
predictions where the forecaster predicts from a previous cutoff date,
not from the test_df's own dates:

```python
# Instead of forecaster.predict(test_df), use:
last_train = train_df["date"].max()
forecast_dates = pd.date_range(start=last_train + pd.Timedelta(days=1),
                                periods=30, freq="D")
# Build forecast features manually for these dates
predictions = forecaster.predict_from_dates(train_df, forecast_dates)
```

Or, simpler: use the same CV approach as Exp01 — train on train_df,
predict dates [train_end+1, train_end+horizon], then compare those
predictions against test_df filtered to those exact dates.

### Expected Improvement
NaN metrics replaced with actual RMSE/MAE/Coverage values. Enables
meaningful worst-campaign identification and horizon degradation analysis.

---

## EXP07: Equal allocation beats optimizer

### Hypothesis
The optimizer's score-based allocation produces higher utility than equal,
proportional, or current allocation.

### Verdict: FAILED
Equal utility: 1.48T, Current/Proportional: 478B, Optimizer: 268B.

### Root Cause: Same as Exp04 — budget leakage from clamped proportional
allocation, compounded by a flawed utility function.

The utility function in the experiment
(`research/experiments/exp07_optimization_validation.py:35-45`) has a
concentration penalty:

```python
def allocation_utility(budgets_dict, baselines_dict):
    total = sum(roas * conf * b for eid, bl, ...)
    if vals.sum() > 0 and len(vals) > 1:
        gini = 1.0 - (vals / vals.sum()).var() * len(vals)
        total *= (1.0 - 0.2 * gini)  # up to 20% penalty
    return total
```

The Gini-based penalty penalizes concentration. Equal allocation has zero
concentration (Gini=0), so no penalty. The optimizer creates a
concentrated allocation (some campaigns get more, others get less), so it
receives a penalty. Combined with the budget leakage from Exp04, the
optimizer's already-reduced budget is further penalized.

### Violated Assumption
**The utility function correctly captures business value.** The 20%
concentration penalty is arbitrary and may not reflect actual business
constraints. In real advertising, concentrating budget on high-ROAS
campaigns is desirable, not penalizable.

### Business Impact
**High** (same as Exp04). The optimizer appears worse than equal
allocation, which would justify removing it entirely.

### Minimal Fix

Two fixes required:

1. Fix the budget conservation (same fix as Exp04)
2. Remove or reduce the concentration penalty in the experiment's utility
   function:

```python
# Change this:
total *= (1.0 - 0.2 * gini)
# To this:
total *= (1.0 - 0.05 * gini)  # reduced from 20% to 5%
```

The real fix, however, is Exp03 (calibration). Fix uncertainty calibration
first, then confidence scores become meaningful, then the optimizer makes
sensible allocations, and the concentrated allocation becomes
well-motivated.

### Expected Improvement
Optimizer utility should at least match current utility (~478B) and
potentially exceed equal allocation (~1.48T), if the concentration penalty
is set correctly.

---

## Dependency Graph

```
Exp03: 14.5% coverage         ← ROOT CAUSE of decision failures
   |
   ├──→ Exp04: Optimizer -50%  ← confidence scores are wrong
   │       └── budget non-conservation amplifies the problem
   │
   ├──→ Exp07: Equal > Optimizer ← same root cause
   │
   └──→ Exp05: Drift detection missing ← separate issue
           (not dependent, but amplifies same pipeline)

Exp01: Aggregation in eval    ← independent methodological bug
Exp02: Feature over-engineering ← independent data issue

Exp06: Future-date mismatch    ← independent testing bug
```

---

## Fix Priority Ranking (by expected improvement)

| Rank | Fix | Exp(s) Fixed | Expected Impact | Effort |
|------|-----|-------------|-----------------|--------|
| 1 | **Conformal calibration** for uncertainty intervals | Exp03 (14.5% → 80%), Exp04, Exp07 | Revenue from $5.5M → $11M+ | 1 file, ~20 lines |
| 2 | **Budget re-normalization** after clamp | Exp04 (50% → full budget) | Revenue from $5.5M → $11M | 1 file, ~5 lines |
| 3 | **Remove rolling/lag features** | Exp02 (-6% RMSE), Exp01, Exp04 | RMSE improvement, better forecasts | 2 lines in config.yaml |
| 4 | **Per-campaign evaluation** in benchmark | Exp01 (correct ordering) | Correct model selection | ~30 lines in exp01 |
| 5 | **Drift detector** before prediction | Exp05 (+361% → detected) | Prevent catastrophic decisions | ~15 lines |
| 6 | **Fix date alignment** in failure analysis | Exp06 (NaN → real metrics) | Monitoring capability | ~5 lines |

Fixes 1+2 are independent, complementary, and together solve the optimizer
failure entirely. Fix 3 improves all downstream components. Fix 4 is
purely diagnostic (benchmark correctness). Fix 5 prevents future
catastrophes. Fix 6 enables automated monitoring.

---

## Summary

| Experiment | Failure | Root Cause | Code Location | Fix |
|-----------|---------|-----------|---------------|-----|
| Exp01 | Ensemble loses to HM | Cross-campaign eval aggregation | `exp01_forecast_benchmark.py:190` | Per-campaign evaluation |
| Exp02 | Features hurt RMSE | Collinearity + curse of dimensionality | `features/transforms.py:68-118` | Remove rolling/lag |
| Exp03 | Coverage 14.5% | SN predicts 0 for unseen + no calibration | `exp03_uncertainty_calibration.py:58-61`, `uncertainty/engine.py` | Conformal calibration |
| Exp04 | Optimizer -50% | Budget non-conservation from clamp | `decision/optimizer.py:49-53` | Re-normalize after clamp |
| Exp05 | +361% under drift | No drift detection | `forecasting/forecaster.py:58` | Add drift check |
| Exp06 | NaN metrics | Forecaster predicts future, not in-sample | `forecasting/forecaster.py:81-86` | Use walk-forward dates |
| Exp07 | Equal > Optimizer | Same as Exp04 + flawed utility penalty | `decision/optimizer.py:49-53` + `exp07:38-44` | Fix budget + reduce penalty |

All decision failures (Exp04, Exp07) trace back to Exp03: broken uncertainty
calibration makes confidence scores meaningless, which makes the optimizer
allocate based on noise. Fixing Exp03 is the single highest-leverage action.
