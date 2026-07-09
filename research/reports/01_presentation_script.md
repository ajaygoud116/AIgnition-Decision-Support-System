# Presentation Script (7 Minutes)

**Format:** Monologue — no audience interaction until Q&A.
**Tone:** Confident, precise, evidence-driven. No filler.

---

## 0:00–0:45 — Hook + Problem

> "Every week, marketing agencies allocate hundreds of thousands in ad spend across Google, Meta, and Bing — using spreadsheets, gut instinct, and Monday morning meetings.
>
> The result? Three days of work for 102 campaigns. Inconsistent recommendations — depends on which analyst shows up. Zero quantification of risk. And scenario analysis is so painful it simply doesn't get done.
>
> That is the problem we are solving today."

- **On screen:** Side-by-side — a messy spreadsheet vs the AIgnition dashboard.

## 0:45–2:00 — Solution + Live Demo

> "We built AIgnition — a fully automated decision support pipeline. It ingests raw campaign data, validates it, engineers features, forecasts revenue with quantile uncertainty bounds, quantifies risk, simulates budget scenarios, and outputs per-campaign recommendations with written rationales.
>
> Let me show you the live pipeline."

> **[LIVE DEMO BEGINS — see demo script for exact keystrokes]**

> "I just ran `python -m src.pipeline.main` — one command. In under 3 minutes, 102 campaigns were ingested, validated, forecasted, risk-scored, scenario-simulated, and optimized. Here's what came out."

- **On screen:** Console output scrolling, then switch to output CSVs.

## 2:00–3:00 — Forecast Results

> "102 campaigns forecasted at 30, 60, and 90-day horizons. Every forecast includes p10, p50, and p90 quantiles — not a point estimate, but a full probability distribution. Total projected revenue at the median: $1.96 million."

- **On screen:** `forecasts.csv` sample — show the quantile columns.

> "But a point forecast without uncertainty is dangerous. That is why we built the uncertainty engine."

## 3:00–4:00 — Uncertainty + Risk

> "Every campaign gets a confidence score, volatility measure, and stability trend. The system flagged 79 of 102 campaigns as high-uncertainty — that is transparency. A manual workflow hides this. We report it.
>
> 207 total flags across 5 types: high uncertainty, below-target ROAS, zero revenue, cost inflation, concentration risk."

- **On screen:** `uncertainty.csv` sample, then the flag type table.

> "Uncertainty quantification uses conformal prediction — a statistically rigorous method. A held-out calibration set finds a scaling factor alpha such that the prediction intervals achieve the target coverage rate. This is not ad hoc. This is principled."

## 4:00–5:00 — Scenario Analysis

> "The pipeline automatically simulates three scenarios: baseline, +10% budget, and -10% budget."

- **On screen:** The simulation results table.

| Scenario | Revenue | Spend | ROAS |
|---|---|---|---|
| Baseline | $2.02M | $2.17M | 0.93 |
| +10% Budget | $3.09M | $2.38M | 1.30 |
| -10% Budget | $0.95M | $1.95M | 0.49 |

> "Key insight: the +10% scenario increases ROAS by 40% — from 0.93 to 1.30. The -10% scenario drops ROAS to 0.49. The marginal return on spend is positive at current allocation. A human analyst would need a full day to produce this. We do it in under a second."

## 5:00–6:00 — Budget Recommendations

> "The decision engine produces 131 per-campaign recommendations, each with a budget change, projected ROAS, and a written rationale explaining why.
>
> Examples: 'Decrease budget 50%. Campaign ROAS is below target. Forecast confidence is low. Campaign has prolonged zero revenue.'
>
> Every recommendation is auditable. Every decision has a rationale. No black box."

- **On screen:** Scroll through `recommendations.csv` showing rationales.

## 6:00–6:45 — Architecture

> "The architecture is a 7-stage pipeline: ingest, validate, feature-engineer, forecast, quantify-uncertainty, simulate, and decide. Each stage is independently testable. We have 288 passing tests — every module covered.
>
> The entire system runs in under 5 minutes on a laptop. No GPU. No cloud. No API keys. One command."

- **On screen:** The architecture diagram (ASCII or slide).

## 6:45–7:00 — Differentiators + Close

> "Three things make this different:
>
> **One:** Conformal uncertainty quantification — not a single number, but statistically rigorous prediction intervals on every forecast.
>
> **Two:** Deterministic, auditable decision-making — same input, same output, every time, with written rationale for every recommendation.
>
> **Three:** Sub-five-minute end-to-end pipeline — from raw CSV to executive summary with scenario analysis and risk flags.
>
> A manual agency takes 3–5 days to do what we do in 5 minutes. They cannot quantify uncertainty. They cannot run scenarios. They cannot guarantee consistency.
>
> AIgnition transforms budget allocation from an opaque, inconsistent, multi-day process into a transparent, deterministic, 5-minute decision support system.
>
> Questions?"

- **On screen:** Final slide — 3 bullets, pipeline animation, "AIgnition — Decisions with Certainty."
