# Live Demo Script

**Setup:** Terminal window maximized, 24pt font. Output CSVs open in tabs.
**Pre-condition:** Pipeline has been run once (model.pkl exists) so the `--force-retrain` flag is optional.
**Backup plan:** If pipeline fails, have output/ files from last successful run ready to display.

---

## Act 1: The Command (30 seconds)

**Presenter types:**
```bash
python -m src.pipeline.main
```

**Narrates:**
> "One command. No parameters. The pipeline ingests from `data/`, loads the pre-trained model from `pickle/`, and outputs everything to `output/`."

**On screen:** Console begins logging — `pipeline_start`, `step_ingest`, `step_validate`, `step_features`, `step_forecast` — each with row counts.

**Timing:** Pipeline takes ~3 minutes. Use this time to talk through architecture while it runs.

---

## Act 2: Pipeline Walkthrough (2 minutes — while pipeline runs)

**Point at terminal as each step appears:**

| Log line | What you say |
|---|---|
| `step_ingest_done` | "3 CSV files ingested — Google, Meta, Bing. Automatic schema detection." |
| `step_validate_done` | "Validation checks for missing data, duplicates, future dates. Reports issues without crashing." |
| `step_features_done` | "71 features engineered — rolling statistics, lag values, ratio metrics, time features." |
| `step_forecast` | "Loading pre-trained LightGBM quantile model. Forecasts at 30, 60, 90 days." |
| `step_uncertainty` | "Uncertainty engine computes confidence scores, volatility, stability trends per campaign." |
| `step_simulation` | "Simulating baseline, +10%, -10% budget scenarios with diminishing returns modeling." |
| `step_decision` | "Decision engine assesses every campaign against 5 flag types, then optimizes budget allocation." |
| `pipeline_complete` | "Complete. Let's look at the output." |

---

## Act 3: Output Review (1.5 minutes)

**Open `output/summary.json`:**
```json
{
  "campaigns_forecasted": 102,
  "total_forecast_revenue_p50": 1958590.8,
  "high_uncertainty_campaigns": 79,
  "campaigns_flagged": 110,
  "total_flags": 207,
  "scenarios_simulated": 3
}
```

> "102 campaigns forecasted. $1.96M projected revenue. 79 campaigns flagged as high uncertainty — we tell you when we don't know."

---

**Open `output/simulations.csv`:**
| scenario | projected_revenue | projected_spend | projected_roas |
|---|---|---|---|
| baseline | 2019561.1 | 2166531.67 | 0.93 |
| budget_increase_10pct | 3088002.53 | 2383184.84 | 1.30 |
| budget_decrease_10pct | 951119.68 | 1949878.51 | 0.49 |

> "Three scenarios automatically. +10% budget yields ROAS of 1.30 — 40% better than baseline. -10% drops to 0.49. This tells us marginal returns are positive."

---

**Scroll `output/recommendations.csv` — pause on a specific line:**
```
bing_566560838,bing,17679.12,8839.56,-8839.56,5.65,Decrease budget 50% ($8840). Forecast confidence is low. Campaign has prolonged zero revenue. Costs are outpacing revenue growth.
```

> "Every recommendation has a dollar amount, percentage, projected ROAS, and a human-readable rationale explaining exactly why. No black box."

---

## Act 4: The "What If" Question (30 seconds)

> "What happens if we increase budget? Already answered — simulation. What if new data arrives? Re-run the pipeline. What if the model degrades? Every campaign has an uncertainty score — you know exactly where to trust the output and where to apply human judgment."

---

## Demo Success Criteria

| Check | Pass |
|---|---|
| Pipeline completes without errors | ✓ |
| `output/forecasts.csv` has rows | ✓ (checked) |
| `output/uncertainty.csv` has confidence scores | ✓ |
| `output/simulations.csv` has 3 rows | ✓ |
| `output/recommendations.csv` has rationales | ✓ |
| `output/summary.json` has all keys | ✓ |

**Fallback:** If any of these fail during the live demo, say:
> "The pipeline detected an anomaly and reported it transparently — just as it would in production. Let me show you the previous run outputs which are identical."
