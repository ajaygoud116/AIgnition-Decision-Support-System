# AIgnition — AI Decision Support System for Marketing Budget Allocation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-288%2F288-passing-green.svg)]()
[![Pipeline](https://img.shields.io/badge/pipeline-5%20min%2F102%20campaigns-success.svg)]()

Automate multi-channel budget planning. Forecast 102 campaigns with uncertainty bounds, simulate scenarios, and get per-campaign recommendations — all in under 5 minutes.

## Quick Start

```bash
# Place CSVs in data/ directory (Google Ads, Meta Ads, Bing Ads formats)
# Run the pipeline:
python -m src.pipeline.main

# Output appears in output/:
#   predictions.csv       — 30/60/90-day quantile predictions
#   uncertainty.csv     — per-campaign confidence, volatility, stability
#   simulations.csv     — baseline, +10%, -10% scenario projections
#   recommendations.csv — 131 budget recommendations with rationales
#   summary.json        — one-page executive summary
```

## Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Ingest   │→ │ Validate │→ │ Features │→ │ Forecast │→ │Uncertainty│→ │ Simulate  │→ │  Decide   │
│ (3 CSVs)  │  │(schema + │  │(71 feats)│  │(LGB quant)│  │(conformal)│  │(3 scen.)  │  │(optimizer)│
└──────────┘  │ quality)  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
              └──────────┘                                                    │
                                                                              ▼
                                                                       ┌──────────┐
                                                                       │  Report   │
                                                                       │ (CSV + JSON)│
                                                                       └──────────┘
```

### Stages

| Stage | Module | What it does |
|---|---|---|
| **Ingest** | `src/ingestion/` | Reads Google/Meta/Bing CSVs → unified 9-column schema |
| **Validate** | `src/validation/` | Schema, missing data, duplicates, date sanity |
| **Features** | `src/features/` | 71 features: rolling stats, lags, ratios, time features |
| **Forecast** | `src/forecasting/` | LightGBM quantile regression (p10/p50/p90 at 30/60/90d) |
| **Uncertainty** | `src/uncertainty/` | Conformal prediction calibration, volatility, stability trends |
| **Simulate** | `src/simulation/` | Baseline, +10%, -10% budget scenarios with diminishing returns |
| **Decide** | `src/decision/` | Campaign assessor (5 flags) + budget optimizer |
| **Report** | `src/report/` | CSV exports + JSON summary |

## Live Pipeline Output (102 campaigns)

| Metric | Value |
|---|---|
| Campaigns forecasted | 102 |
| Total forecast revenue (p50) | $1,958,590.80 |
| Campaigns flagged | 110 (207 total flags) |
| High uncertainty campaigns | 79 |
| Scenarios simulated | 3 |

### Scenario Analysis

| Scenario | Projected Revenue | Projected Spend | ROAS |
|---|---|---|---|
| Baseline | $2,019,561.10 | $2,166,531.67 | 0.93 |
| +10% Budget | $3,088,002.53 | $2,383,184.84 | **1.30** |
| -10% Budget | $951,119.68 | $1,949,878.51 | 0.49 |

### Flag Types

| Flag | Count | Description |
|---|---|---|
| high_uncertainty | 79 | Model confidence below threshold |
| below_roas_target | 42 | Projected ROAS < 3.0 |
| zero_revenue | 31 | No revenue for 45+ days |
| cost_inflation | 22 | Cost growth exceeding threshold |
| concentration_risk | 33 | Budget too concentrated |

## Usage

```bash
# Full retrain (train model from scratch)
python -m src.pipeline.main --force-retrain

# Custom data directory
python -m src.pipeline.main --data-dir ./my_campaigns

# Custom config
python -m src.pipeline.main --config ./my_config.yaml
```

## Configuration

Edit `config.yaml`:

- `forecast.horizons`: forecast horizons in days (default: [30, 60, 90])
- `decision.min_roas_target`: ROAS threshold for alerts (default: 3.0)
- `decision.volatility_threshold`: volatility flag threshold (default: 0.5)
- `validation.valid_campaign_types`: accepted campaign types per channel

## Testing

```bash
python -m pytest src/__tests__/ -v
```

**288 tests, all passing.** Coverage across all modules: ingestion, validation, features, forecasting, uncertainty, simulation, decision, report.

## Experimental Validation

8 controlled experiments validate the system:

| Experiment | Finding |
|---|---|
| Exp01: Forecast Benchmark | Historical Mean benchmark has aggregation bug — per-campaign eval needed |
| Exp02: Feature Ablation | Only ratio features help; rolling/lag features add noise (+6% RMSE) |
| Exp03: Uncertainty Calibration | Raw intervals cover 14.5% vs 80% target — conformal calibration fixes |
| Exp04: Business Evaluation | Budget non-conservation causes -50% revenue — re-normalization fixes |
| Exp05: Sensitivity | Concept drift catastrophic (+361%) — drift detector needed |
| Exp06: Failure Analysis | Date alignment bug — NaN metrics; walk-forward evaluation fixes |
| Exp07: Optimization | Concentration penalty too aggressive (20% → 5% recommended) |
| Exp08: Complexity | All models lightweight; no computational bottleneck |

## Data Schema

The pipeline accepts CSVs from three platforms with these normalized columns:

| Column | Type | Description |
|---|---|---|
| date | date | Campaign date (YYYY-MM-DD) |
| campaign_id | string | Unique campaign identifier |
| channel | string | google / meta / bing |
| spend | float | Daily ad spend |
| revenue | float | Daily attributed revenue |
| clicks | int | Daily clicks |
| impressions | int | Daily impressions |
| conversions | float | Daily conversions |
| daily_budget | float | Campaign daily budget |

## Limitations

1. **Forecast quality depends on data quality** — campaigns with <30 days history are flagged as high uncertainty
2. **No causal inference** — the system models correlation, not causation
3. **No drift detection in production** — concept drift can silently degrade predictions
4. **Budget conservation gap** — optimizer constraints may prevent full budget allocation
5. **No external factors** — holidays, competitor actions, market shifts are not modeled

## License

Hackathon submission — not for commercial use.

---

*AIgnition: Decisions with Certainty.*
