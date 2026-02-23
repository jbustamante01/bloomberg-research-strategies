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

### Entry Tiers

| Tier | Value Pctle | P/SMA50 | RSI | MACD |
|------|------------|---------|-----|------|
| Deep Value | > 80 | < 1.25 | < 72 | — |
| Good Value | 60–80 | < 1.20 | < 65 | > 0 |
| Moderate | 50–60 | < 1.10 | < 58 | — |

During true stress periods (SPY below SMA200 **and** VIX above 25), all bars tighten by one notch.

### Exit Rules

- **Extended**: P/SMA50 > 1.35 — stock has run, take profit
- **Overbought**: RSI > 80
- **Stop loss**: P/SMA50 < 0.92 — trend has broken

### Portfolio Construction

- Long only
- Fixed dollar sizing per trade (default $1,000 — adjust to your own sizing)
- No position cap — signals drive everything, cash when no signals

### Backtest Notes

The backtest uses the current screener universe applied historically, so universe selection bias is present — names that rank well today generally performed well over the past year. What the backtest does validate is that the entry/exit signal timing works on fundamentally strong names. The current signals table is the primary output; the backtest validates the logic behind it.

---

## Technical Stack

- Bloomberg BQuant (Python notebook environment)
- BQL for all data access
- pandas, numpy, scikit-learn (PCA), matplotlib

---

## Contact

**Email:** jason.a.bustamante01@gmail.com

---

*For educational and research purposes only. Not investment advice.*
