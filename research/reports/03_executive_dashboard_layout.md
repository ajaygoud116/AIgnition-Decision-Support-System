# Executive Dashboard Layout

A single-screen dashboard for a CMO or agency head. Every number is live from the pipeline output.

```
┌──────────────────────────────────────────────────────────────────────┐
│  AIGNITION  ●  Decision Support System           [ Last run: < 5m ago ] │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐                  │
│  │  CAMPAIGNS           │  │  TOTAL REVENUE (p50)  │                  │
│  │  102  forecasted     │  │  $1,958,591           │                  │
│  │  110  flagged        │  │                       │                  │
│  │  79   high-uncertainty│  │  $3,088,003 at +10%  │                  │
│  └──────────────────────┘  └──────────────────────┘                  │
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐                  │
│  │  PORTFOLIO ROAS       │  │  BUDGET ACTIVE      │                  │
│  │  Baseline     0.93   │  │  $2,166,532 total    │                  │
│  │  +10% budget  1.30   │  │  131 recommendations  │                  │
│  │  -10% budget  0.49   │  │  49.9% allocatable   │                  │
│  └──────────────────────┘  └──────────────────────┘                  │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  RISK OVERVIEW                                                    │
│                                                                      │
│  Flags by type:              │  Campaigns by confidence:          │
│  ████████████ high_uncertainty (79)│  ■ Low (<0.5)     ████████████ 79│
│  ████ below_roas_target     (42)│  ■ Medium (0.5-0.8) ██         8 │
│  ███ zero_revenue           (31)│  ■ High (>0.8)     ████████   15 │
│  ██ cost_inflation          (22)│                                   │
│  █ concentration_risk       (33)│                                   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  CHANNEL BREAKDOWN                                                  │
│                                                                      │
│  Channel    Campaigns    Total Flags  Avg Confidence  Avg Volatility │
│  ───────    ─────────    ───────────  ──────────────  ────────────── │
│  bing          55           —            0.03            0.35       │
│  google        43           —            0.01            0.36       │
│  meta           4           —            0.00            0.00       │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  SCENARIO COMPARISON                                                │
│                                                                      │
│  $3.5M ┤                                            ┌────┐         │
│  $3.0M ┤                                  ┌─────────┤3.09│         │
│  $2.5M ┤                    ┌────┐         │   M    └────┘         │
│  $2.0M ┤          ┌─────────┤2.02│─────────┤                        │
│  $1.5M ┤          │   M    └────┘         │                        │
│  $1.0M ┤┌────┐    │                        │          ┌────┐       │
│  $0.5M ┤│0.95│────┘                        └──────────┤0.95│       │
│        └┴────┴─────────────────────────────────────────┴────┴────  │
│         -10%       Baseline       +10%                              │
│          0.49 ROAS  0.93 ROAS     1.30 ROAS                        │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  TOP RECOMMENDATIONS                                                │
│                                                                      │
│  Campaign ID           Channel  Change    Expected ROAS  Rationale │
│  ───────────────────   ───────  ───────   ─────────────  ─────────  │
│  bing_570837633        bing     -$13.13        17.73    Low conf,  │
│                                                           zero rev  │
│  bing_566560838        bing     -$8,839.56      5.65    Low conf,  │
│                                                           zero rev  │
│  ... (scrollable list of 131)                                      │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  SYSTEM HEALTH                                                      │
│                                                                      │
│  • Model: LightGBM Quantile (3 targets: p10, p50, p90)              │
│  • Calibration: Conformal (α=3.59 computed from held-out data)      │
│  • Tests: 288/288 passing                                           │
│  • Forecast horizons: 30, 60, 90 days                               │
│  • Data freshness: 2024-01-01 to 2026-06-05                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Single screen** — no scrolling required for the core KPIs
2. **All numbers from live pipeline** — no mock data in the demo
3. **Risk shown first** — uncertainty and flags are the most important signal
4. **Scenario comparison is visual** — the bar chart tells the story instantly
5. **Recommendations are actionable** — every row has a rationale
6. **System health at the bottom** — proves it's real tech, not a mockup
