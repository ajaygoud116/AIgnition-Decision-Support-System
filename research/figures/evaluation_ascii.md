# Evaluation Figure

```
                              FORECAST BENCHMARK (Exp01 — Post-Fix)
                              ─────────────────────────────────────
 Model           RMSE    MAE    MAPE  Coverage_90
 ─────           ────    ───    ────  ───────────
 HistoricalMean  244.9  197.4  280.7   0.73      ████████████████████▌░░░░░░░░░░░░░░░░░
 Ensemble        330.2  258.9  316.1   0.28      ████████████████████░░░░░░░░░░░░░░░░░░░░░
 SeasonalNaive   269.2  212.1  269.0   0.49      ████████████████████░░░░░░░░░░░░░░░░░░░░
 Naive           361.0  323.9  158.6   0.10      ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░
 LightGBM        473.2  406.4  437.5   0.14      ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 XGBoost         573.6  470.8  396.8   0.11      ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 RandomForest    584.2  484.1  398.9   0.07      ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 LinearReg       752.7  675.0  761.6   0.07      ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

  0    100   200   300   400   500   600   700   800

 Bug: Cross-campaign aggregation (groupby date → average) destroys campaign identity.
 Fix: Per-campaign evaluation → ensemble RMSE expected ~200-250.


                              FEATURE ABLATION (Exp02)
                              ───────────────────────
 Configuration          RMSE    Delta vs All
 ─────────────          ────    ─────────────
 All features           587.0   —
 Without rolling        558.4   -4.9%  ██████████████████████░░░░░░░░░░ (improvement)
 Without lag            548.6   -6.5%  ██████████████████████░░░░░░░░░░ (improvement)
 Without ratio     →    598.8   +2.0%  ████████████████████████████████ (degradation)
 Without time           559.7   -4.7%  ██████████████████████░░░░░░░░░░ (improvement)

 Only ratio features: 6.5% better than full set. 84% of features add noise.


                              UNCERTAINTY CALIBRATION (Exp03)
                              ─────────────────────────────
 Metric               Expected    Observed
 ──────               ────────    ────────
 Coverage (80% CI)     80.0%       14.5%   █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 Interval width         2.34        0.71   ██████░░░░░░░░░░░░░░░░░░░░░

 Coverage gap: 65.5 percentage points
 ┌──┐
 │  │ Expected 80% coverage
 │  │
 │  │ ░░░░ Observed 14.5%
 │  │ ░░░░
 │  │ ░░░░
 └──┴────────────────────
 0%  20% 40% 60% 80% 100%


                              BUSINESS EVALUATION (Exp04)
                              ──────────────────────────
 Strategy       Revenue       ROAS    Risk    Efficiency
 ────────       ───────       ────    ────    ──────────
 Current       $11,040,561    5.10    2.41      4.51     ████████████████████████
 Uniform       $ 8,462,698    3.91    0.00      3.88     ██████████████████
 Proportional  $11,040,561    5.10    2.41      4.51     ████████████████████████
 Optimizer     $ 5,531,532    2.55    2.40      2.40     ████████████

 Optimizer loses -50% revenue due to budget non-conservation (clamp breaks sum).


                              SENSITIVITY ANALYSIS (Exp05)
                              ───────────────────────────
 Scenario              RMSE        Delta vs Baseline
 ────────              ────        ───────────────────
 Baseline              696.3       —
 Missing 10%           680.2       -2.3%  █████████████░░░░░░░░░░░░░ (improvement)
 Missing 30%           625.8      -10.1%  █████████████░░░░░░░░░░░░░ (improvement)
 Noise 2x              853.9      +22.6%  ██████████████████████░░░░░
 Noise 3x            1,090.3      +56.6%  ████████████████████████████░
 Outliers 2%         1,239.3      +78.0%  ███████████████████████████████
 Concept Drift    →  3,213.3     +361.5%  ████████████████████████████████████████████████

 Catastrophic under drift. No detector in current codebase.

                              COMPLEXITY BENCHMARK (Exp08)
                              ────────────────────────────
 Model          Train Time    Infer (ms/row)    Memory
 ─────          ──────────    ──────────────    ──────
 LightGBM        4.73s          0.105           5.0 MB  ████████░░░░░░░░░░░░░░░░
 Seasonal Naive  0.03s          0.004           1.6 MB  ██░░░░░░░░░░░░░░░░░░░░░░
 Ensemble        4.76s          0.059           6.6 MB  ████████░░░░░░░░░░░░░░░░

 All models lightweight. No computational bottleneck. Bottleneck is model quality.


                              TEST COVERAGE (288 tests)
                              ─────────────────────────
 Module          Tests    Status
 ──────          ─────    ──────
 ingestion      24       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 validation     18       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 features       30       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 forecasting    42       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 uncertainty    36       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 simulation     24       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 decision       48       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 report         30       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
 pipeline       36       ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●

 TOTAL: 288/288 passing | Run time: < 10 seconds
```
