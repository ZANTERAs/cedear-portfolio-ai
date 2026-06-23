# CEDEAR Heat Finder

Screens ~140 BYMA CEDEARs with a 5-factor fundamental **heat score** to surface
mid/short-term opportunities, then builds a score-weighted top-20 portfolio and
a 5-page PDF report. Data comes from Yahoo Finance via `yfinance` (no API key
required).

## Scoring factors (0–100)

| Factor | Pts | What it measures |
|-----------|----:|------------------|
| Valuación | 30  | PEG ratio, Forward P/E vs growth |
| Analistas | 25  | Consensus rating + target-price upside |
| Earnings  | 20  | Revenue growth, EPS growth, margins |
| Momentum  | 15  | 1m / 3m / 6m returns (percentile-ranked) |
| Técnico   | 10  | RSI + 50/200-day moving-average position |

The fundamentals table also flags **% below the 52-week high** (green = healthy
pullback, orange = pinned to the high) and **overbought RSI > 70** (red).

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python cedear_portfolio_ai.py
```

Outputs (regenerated each run, git-ignored by default):

- `cedear_heat_report.pdf` — 5-page visual report
- `cedear_heat_result.json` — scores + portfolio weights

## Disclaimer

For research/educational use only. Not investment advice.
