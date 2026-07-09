# Architecture Figure

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  AIGNITION PIPELINE ARCHITECTURE                              │
│                                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │          │   │          │   │          │   │          │   │          │   │          │   │          │ │
│  │  INGEST  │──>│ VALIDATE │──>│ FEATURES │──>│ FORECAST │──>│UNCERTAINTY│──>│ SIMULATE │──>│  DECIDE  │ │
│  │          │   │          │   │          │   │          │   │          │   │          │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │              │              │              │              │              │              │       │
│       ▼              ▼              ▼              ▼              ▼              ▼              ▼       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ 3 CSVs   │   │Schema +  │   │ Rolling  │   │LGB QRM   │   │Conformal │   │Baseline  │   │Assessor  │ │
│  │Google    │   │Quality   │   │ Stats    │   │p10/p50/  │   │Calibrate │   │+10%      │   │+         │ │
│  │Meta      │   │Missing   │   │ Lags     │   │p90       │   │Volatility│   │-10%      │   │Optimizer │ │
│  │Bing      │   │Duplicates│   │ Ratios   │   │30/60/90d │   │Coverage  │   │Diminish  │   │5 Flags   │ │
│  └──────────┘   │Dates     │   │ Time     │   └──────────┘   │Stability │   │Returns   │   │131 Recs  │ │
│                 └──────────┘   └──────────┘                 └──────────┘   └──────────┘   └──────────┘ │
│                                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                   CONFIG (config.yaml)                                    │ │
│  │  horizons: [30,60,90] | min_roas_target: 3.0 | volatility_threshold: 0.5 | random_seed: 42│ │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│                                     OUTPUT LAYER                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                          │ │
│  │   output/forecasts.csv            306 series × 90 days × 3 quantiles                      │ │
│  │   output/uncertainty.csv          102 campaigns × 4 metrics (conf, vol, stability, width) │ │
│  │   output/simulations.csv          3 scenarios × (revenue, spend, ROAS)                    │ │
│  │   output/recommendations.csv      131 campaigns × (change, ROAS, rationale)               │ │
│  │   output/summary.json             9 KPIs in one page                                      │ │
│  │                                                                                          │ │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
│                                     DATA FLOW                                                │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────────────┐ │
│  │ raw CSVs   │   │ validated  │   │ feature    │   │ forecast   │   │ reports            │ │
│  │ ~25K rows  │──>│ DataFrame  │──>│ DataFrame  │──>│ result     │──>│ CSV + JSON          │ │
│  │ 3 channels │   │ 9 cols     │   │ 71 cols    │   │ p10/p50/p90│   │ output/             │ │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────────────┘ │
│                                                                    │                        │
│                                                                    ▼                        │
│                                                           ┌────────────────────┐            │
│                                                           │ pickle/model.pkl   │            │
│                                                           │ LightGBM quantile  │            │
│                                                           │ 3 targets × 3 hzn │            │
│                                                           └────────────────────┘            │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
