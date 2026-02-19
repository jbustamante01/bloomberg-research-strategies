# bloomberg-research-strategies
Quantitative equity research project using Bloomberg BQuant (BQL) to design multi‑factor screeners and backtest systematic investment strategies.

---

Pursuing excess returns can be both fun and daunting. Here, we explore different equity screeners, portfolio construction and strategy ideas in a paper‑trading environment.

---

## Overview
Multi-factor investment research pipeline developed on Bloomberg BQuant using BQL (Adapation to public data WIP)

## Methodology

### 1. Universe Construction
- Pulled constituents from SPX, Russell 2000, Nasdaq-100
- Deduplicated across indices
- Final universe: ~800 unique securities

### 2. Factor Construction
**Value Factors:**
- FCF Yield
- Price/Sales (inverted)
- P/E Ratio (inverted)
- EV/EBITDA (inverted)

**Profitability Factors:**
- 3-Year Average ROE
- 3-Year Average ROIC
- Operating Margin Change

**Momentum Factors:**
- 12-1 month momentum
- 6-1 month momentum
- 3-1 month momentum

### 3. Composite Ranking Models
- **Percentile Composite:** Cross-sectional percentile ranks
- **Z-Score Composite:** Standardized factor scores
- **PCA Composite:** First principal component

### 4. Signal Generation
Combined fundamental and technical signals:
- Value + Profitability screens
- Technical confirmation (SMA, RSI, MACD)
- Entry/exit rules with mean-reversion logic

### 5. Backtesting
- Equal-weight long-only portfolio
- Daily rebalancing on signals
- 1-year lookback period
- Performance vs SPY benchmark

## Results Summary

## Technical Implementation

**Tools:**
- Bloomberg BQuant (Python environment)
- BQL for data access
- pandas, numpy, scikit-learn
- matplotlib for visualization

## Key Learnings

- Factor combination improves signal quality
- Technical confirmation reduces false positives
- Equal-weight outperforms cap-weight in this universe
- Mean-reversion works better than momentum in current regime

---

## Contact


**Email:** jason.a.bustamante01@gmail.com


## Note on Code Access

This project was developed within Bloomberg's BQuant environment. Due to Bloomberg's data licensing and security policies, the code and data cannot be shared outside the platform.

## Disclaimer

This repository is for educational and research purposes only.
All data is sourced from Bloomberg Terminal via BQL.
This does not constitute investment advice.
