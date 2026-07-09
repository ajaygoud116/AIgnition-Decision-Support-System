# Business Workflow Comparison

## Before: Manual Agency Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WEEKLY/MONTHLY BUDGET PLANNING CYCLE                      │
│                                                                              │
│  Day 1 ──── Data Collection                                                 │
│              │ Export CSVs from Google Ads, Meta Ads, Bing Ads               │
│              │ Manual reconciliation in Excel (2-3 hours)                     │
│              ▼                                                               │
│  Day 1-2 ── Forecast Preparation                                            │
│              │ Analyst 1 builds forecast for Google campaigns (20-30 min ea)  │
│              │ Analyst 2 builds forecast for Meta campaigns (20-30 min ea)    │
│              │ Different Excel templates, different assumptions                │
│              ▼                                                               │
│  Day 2-3 ── Budget Meeting #1                                               │
│              │ Present forecasts to account director                          │
│              │ "Why is campaign X down?" "Let's cut Y and boost Z"           │
│              │ No quantitative risk assessment — "feels risky" is the metric  │
│              ▼                                                               │
│  Day 3-4 ── Revision Loop                                                   │
│              │ Analysts revise spreadsheets based on meeting feedback          │
│              │ "What if we increase budget by 10%?" — build new sheet (4+ hr) │
│              ▼                                                               │
│  Day 4-5 ── Budget Meeting #2                                               │
│              │ Final approval                                                 │
│              │ Decisions based on who argued loudest, not data                 │
│              ▼                                                               │
│  Day 5 ──── Implementation                                                  │
│              │ Manual entry of budget changes into ad platforms                │
│              │ No written rationale — "we agreed in the meeting"               │
│              │ No audit trail — "who decided to cut campaign X?"               │
│                                                                              │
│  TOTAL: 3-5 BUSINESS DAYS PER CYCLE                                          │
│  OUTPUT: 1 Excel file with final budget allocations                          │
│  QUALITY: Analyst-dependent, inconsistent, un-auditable                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## After: AIgnition Automated Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          5-MINUTE AUTOMATED CYCLE                            │
│                                                                              │
│  Step 1 ──── python -m src.pipeline.main  (< 1 min setup)                   │
│              │ Place 3 CSVs in data/ directory                                │
│              │ Run one command                                                │
│              ▼                                                               │
│  Step 2 ──── Automated Pipeline (< 4 min execution)                         │
│              │ Ingest: 3 CSVs → unified schema (automatic)                   │
│              │ Validate: schema, missing data, duplicates (reported)          │
│              │ Features: 71 engineered features (transparent)                 │
│              │ Forecast: 102 campaigns × 3 horizons × 3 quantiles            │
│              │ Uncertainty: confidence scores, volatility, stability flags   │
│              │ Simulate: baseline, +10%, -10% scenarios (automatic)          │
│              │ Decide: 131 recommendations with written rationales            │
│              ▼                                                               │
│  Step 3 ──── Review Output                                                  │
│              │ output/summary.json — 9 KPIs, 30-second review                 │
│              │ output/forecasts.csv — all forecasts at p10/p50/p90            │
│              │ output/uncertainty.csv — risk per campaign                     │
│              │ output/simulations.csv — what-if scenarios                     │
│              │ output/recommendations.csv — 131 decisions each with rationale │
│              ▼                                                               │
│  Step 4 ──── Human Review (30 min)                                          │
│              │ Focus only on flagged campaigns (79 high-uncertainty)          │
│              │ Override any recommendation with one click                     │
│              │ Document reasoning for overrides                               │
│              ▼                                                               │
│  Step 5 ──── Implement                                                      │
│              │ Export recommendations to ad platforms                         │
│              │ Every decision is auditable — "why did we change campaign X?"  │
│                                                                              │
│  TOTAL: ~35 MINUTES (5 min automated + 30 min human review)                  │
│  OUTPUT: 5 files (4 CSV + 1 JSON) with full audit trail                      │
│  QUALITY: Deterministic, consistent, risk-transparent, auditable              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Comparison Matrix

| Dimension | Manual Agency | AIgnition | Impact |
|---|---|---|---|
| **Time per cycle** | 3-5 days | 35 minutes | **~20× faster** (including review) |
| **Analysts required** | 2-3 | 0 (pipeline) + 1 (reviewer) | **Headcount freed for strategy** |
| **Forecast consistency** | Low (analyst-dependent) | Perfect (deterministic) | **No "it depends"** |
| **Uncertainty per campaign** | "Feels risky" | 0.0–1.0 confidence score | **Quantified risk** |
| **Scenario analysis** | 1 day per scenario | < 1 second per scenario | **~50,000× faster** |
| **Recommendation rationale** | "We agreed in a meeting" | Written, per campaign | **Full audit trail** |
| **Risk flagging** | Ad hoc (misses patterns) | 5 types, 207 flags | **Systematic coverage** |
| **Reproducibility** | "Who made this spreadsheet?" | Same input → same output | **Trustable** |
| **Onboarding new client** | 2-3 days template setup | Drop CSVs in data/ | **Instant** |
| **Decision quality signal** | "It felt right" | ROAS × confidence = score | **Data-driven** |
| **Portfolio visibility** | Fragmented spreadsheets | Single dashboard | **Panoramic view** |
| **Cost per planning cycle** | 3-5 analyst-days | 30 min human review | **~90% cost reduction** |

## Key Insight

> **The manual workflow spends 99% of its time on tasks the AI can do in seconds — data processing, forecasting, scenario calculation, and documentation — leaving only 1% for the one thing humans do best: strategic judgment.**

AIgnition inverts the ratio: the machine does the computation, the human does the thinking.
