# bloomberg-research-strategies
Quantitative equity research using Bloomberg BQuant (BQL) to build multi-factor screeners and backtest systematic long-only strategies.

---

Pursuing excess returns can be both fun and daunting. Here I explore different equity screeners, portfolio construction ideas, and signal-driven strategies in a paper-trading environment.

---

## Strategy: Value + Technical (Tiered)

The core idea is simple — find fundamentally cheap names and use technicals to time the entry. Rather than buying everything that screens well, the entry bar adjusts based on how undervalued a name is. Deep value names get more room on the technical side because the fundamental case is strong enough to carry short-term noise. Moderate value names need stronger technical confirmation before entering.

### Universe

- Russell 3000 + Nasdaq 100, deduplicated on ticker
- Names must rank in the top 150 across all three composite models to make the cut
- Typically produces 30-50 names depending on market conditions

### Factor Groups

**Value** — FCF Yield, P/Sales, P/E, EV/EBITDA (lower multiples ranked higher)

**Profitability** — 3Y avg ROE, 3Y avg ROIC, operating margin change YoY

**Momentum** — 12-1, 6-1, and 3-1 month returns (skipping the most recent month to avoid reversal)

### Composite Ranking

Each factor group is scored three ways and a name needs to rank highly on all three to pass:
- Percentile rank across the universe
- Z-score composite
- First principal component (PCA)

### Relative Value Z-Score (Value Weighted Z)

Rather than relying solely on cross-sectional ranking, names are evaluated across three dimensions:

- **vs Self** — how cheap is the name relative to its own 2Y EWM history? Avoids penalizing structurally cheap sectors.
- **vs Sector** — how cheap relative to market-cap weighted sector peers? Corrects for cross-sector valuation bias.
- **vs Index** — standard cross-sectional z-score against the full universe.

These three z-scores are blended equally (0.33 each) into a single **Value Weighted Z**, which drives tier assignment. This replaces the raw percentile rank used in v1.

### Entry Tiers

| Tier | Value WZ Pctle | P/SMA50 | RSI | MACD |
|------|----------------|---------|-----|------|
| Deep Value | > 80 | < 1.25 | < 72 | — |
| Good Value | 60–80 | < 1.20 | < 65 | > 0 |
| Moderate | 50–60 | < 1.10 | < 58 | — |

During true stress periods (SPY below SMA200 **and** VIX above 25), all bars tighten by one notch.

### Pyramid Position Manager

Positions can scale from a base entry up to 2x capital through two add layers, gated by price action and regime:

| State | Size | Add Condition |
|-------|------|---------------|
| Base (1.0) | $1,000 | Entry bar met |
| Add 1 (1.5) | $1,500 | P/SMA50 1.05–1.25, RSI < 72, MACD > 0, 5+ days since base |
| Add 2 (2.0) | $2,000 | P/SMA50 1.12–1.28, RSI < 75, MACD accelerating, 5+ days since Add 1 |

Pyramiding is only permitted when SPY is above SMA200 and VIX ≤ 25. Step-down logic trims adds before a full exit.

### Exit Rules

- **Extended**: P/SMA50 > 1.35 — stock has run, take profit
- **Overbought**: RSI > 80
- **Stop loss**: P/SMA50 < 0.92 — trend has broken
- Exit thresholds tighten with each pyramid level (1.40/1.50 for Add1/Add2 states)

### Portfolio Construction

- Long only
- Fixed dollar sizing per trade (default $1,000 — adjust to your own sizing)
- No position cap — signals drive everything, cash when no signals

---

## What Changed in v2: Point-in-Time Universe

The original version ran the screener once on today's data and applied it backwards across the full backtest window. That introduced **universe selection bias** — the backtest was effectively trading names that looked good *today*, which naturally inflates historical performance since those names already proved themselves over the period being tested.

v2 fixes this by rebuilding the screener at each quarterly rebalance date using only data available at that point in time. The backtest only trades names that would have qualified *as of that date*, not names that qualify today. As expected, the Sharpe ratio came down from the v1 number — a more honest reflection of what the strategy would have actually produced.

The entry/exit signal timing logic itself is unchanged. What the backtest now measures is whether those signals add value on names that screened well on a forward-blind basis.

---

## Known Limitations & Roadmap

The current framework applies fixed technical thresholds uniformly across names, sectors, and regimes. It doesn't learn from the data it pulls — a name with high earnings volatility gets the same RSI bar as a stable compounder, and entry timing doesn't adapt to how a particular stock has historically behaved around those levels.

The next phase will address this by adding:

- **Logistic regression** — estimate the probability of a profitable entry from technical features, replacing hard thresholds with a probability score
- **Bayesian updating** — update prior beliefs about a name's signal quality as new trades resolve, allowing the model to differentiate between signals that have historically worked and those that haven't
- **Kelly sizing** — size positions based on estimated edge and variance rather than fixed dollar amounts
- **Sliding window / concept drift detection** — re-estimate parameters on a rolling basis so the model adapts when relationships change

---

## Technical Stack

- Bloomberg BQuant (Python notebook environment)
- BQL for all data access
- pandas, numpy, scikit-learn (PCA), scipy, matplotlib

---

## Contact

**Email:** jason.a.bustamante01@gmail.com

---

*For educational and research purposes only. Not investment advice.*
