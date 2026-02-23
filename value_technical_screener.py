"""
Value + Technical Signal Strategy

Multi-factor screener across Russell 3000 and Nasdaq 100.
Combines fundamental value rankings with technical entry/exit signals
to identify long opportunities on a daily basis.

Names are ranked across value, profitability, and momentum factor groups.
Entry bars are tiered by how undervalued a name is — deeper value gets
more lenient technical requirements, moderate value needs stronger confirmation.

Long only, fixed dollar sizing, defined exits on extension, overbought RSI,
or a stop loss when the trend breaks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import bql

bq = bql.Service()
as_of_date = __import__('datetime').date.today()
params = {'dates': as_of_date, 'fill': 'prev'}
range_3y = bq.func.range(-2, 0)


# =============================================================================
# SECTION 1 — FACTOR DEFINITIONS
# Multiplier of -1 flips the sign on metrics where lower is better
# (PE, EV/EBITDA, P/S) so everything ranks in the same direction.
# =============================================================================

value = {
    'FCF Yield': {
        'item': bq.data.fcf_ev_yield(**params),
        'col_name': 'FCF Yield'
    },
    'Px/Sales': {
        'item': bq.data.px_to_sales_ratio(**params),
        'multiplier': -1,
        'col_name': 'Px/Sales'
    },
    'PE': {
        'item': bq.data.pe_ratio(**params),
        'multiplier': -1,
        'col_name': 'PE'
    },
    'EV/EBITDA': {
        'item': bq.data.ev_to_ebitda(**params),
        'multiplier': -1,
        'col_name': 'EV/EBITDA'
    },
}

profitability = {
    'Avg ROE 3Y': {
        'item': bq.data.return_com_eqy(**params, fa_period_offset=range_3y).avg(),
        'col_name': 'Avg ROE 3Y'
    },
    'Avg ROIC 3Y': {
        'item': bq.data.RETURN_ON_INV_CAPITAL(**params, fa_period_offset=range_3y).avg(),
        'col_name': 'Avg ROIC 3Y'
    },
    'Op Mrg Chg': {
        'item': bq.data.OPER_MARGIN(**params, fpo='1') - bq.data.OPER_MARGIN(**params, fpo='0'),
        'col_name': 'Op Mrg Chg'
    },
}

momentum = {
    'Mom 12-1': {
        'item': bq.data.total_return(calc_interval=bq.func.range(f'{as_of_date}-12m', f'{as_of_date}-1m')),
        'col_name': 'Mom 12-1'
    },
    'Mom 6-1': {
        'item': bq.data.total_return(calc_interval=bq.func.range(f'{as_of_date}-6m', f'{as_of_date}-1m')),
        'col_name': 'Mom 6-1'
    },
    'Mom 3-1': {
        'item': bq.data.total_return(calc_interval=bq.func.range(f'{as_of_date}-3m', f'{as_of_date}-1m')),
        'col_name': 'Mom 3-1'
    },
}

display_items = {
    'Name': bq.data.name(),
    'Ticker': bq.data.ticker(),
    'Id': bq.data.id(),
    'Sector': bq.data.classification_name(),
}


# =============================================================================
# SECTION 2 — UNIVERSE FETCH & FACTOR SCORING
# Russell 3000 for breadth, Nasdaq 100 to ensure large cap tech is represented.
# Deduplicate on ticker after combining. A name needs to rank in the top 150
# across all three composites (percentile, z-score, PCA) to make the final cut.
# =============================================================================

def raw_expr(d):
    return d['item'] * d.get('multiplier', 1)

raw_items = {d['col_name']: raw_expr(d) for d in list(value.values()) + list(profitability.values()) + list(momentum.values())}
data_items = {**display_items, **raw_items}

index_univ = {
    'Russell': bq.univ.members('RAY Index', dates=as_of_date),
    'Nasdaq100': bq.univ.members('NDX Index', dates=as_of_date),
}

frames = []
for lab, univ in index_univ.items():
    req = bql.Request(univ, data_items, with_params={'mode': 'cached'})
    rsp = bq.execute(req)
    df_x = pd.concat([it.df()[it.name] for it in rsp], axis=1)
    df_x['Index'] = lab
    frames.append(df_x)

df = pd.concat(frames).reset_index(drop=True)
before = len(df)
df = df.drop_duplicates(subset='Ticker', keep='first')
print(f"Universe: {len(df)} unique tickers ({before - len(df)} duplicates removed)")

raw_cols = list(raw_items.keys())

def pctle(series):
    return series.rank(pct=True) * 100

def pctle_group(comp_dicts):
    return pd.concat([pctle(df[d['col_name']] * d.get('multiplier', 1)) for d in comp_dicts], axis=1).mean(axis=1)

df['Value Pctle'] = pctle_group(value.values())
df['Profitability Pctle'] = pctle_group(profitability.values())
df['Momentum Pctle'] = pctle_group(momentum.values())
df['Composite Pctle'] = df[['Value Pctle', 'Profitability Pctle', 'Momentum Pctle']].mean(axis=1)
df['Rank Pctle'] = df['Composite Pctle'].rank(ascending=False)

z_frame = (df[raw_cols] - df[raw_cols].mean()) / df[raw_cols].std(ddof=0)
df['Z Composite'] = z_frame.mean(axis=1)
df['Rank Z'] = df['Z Composite'].rank(ascending=False)

pc1 = PCA(1).fit_transform(z_frame.fillna(0))[:, 0]
df['PCA Composite'] = pc1
df['Rank PCA'] = df['PCA Composite'].rank(ascending=False)

mask = (df['Rank Pctle'] <= 150) & (df['Rank Z'] <= 150) & (df['Rank PCA'] <= 150)
top100 = df.loc[mask]
id_list = top100['Id'].tolist()
print(f"Screener output: {len(top100)} names in top 150 across all three composites")

rec_items = {
    'Tot Buy': bq.data.tot_buy_rec(dates=as_of_date),
    'Tot Hold': bq.data.tot_hold_rec(dates=as_of_date),
    'Tot Sell': bq.data.tot_sell_rec(dates=as_of_date),
    'Tot Recs': bq.data.tot_analyst_rec(dates=as_of_date),
}
rec_rsp = bq.execute(bql.Request(bq.univ.list(id_list), rec_items, with_params={'mode': 'cached'}))
rec_df = pd.concat([it.df()[it.name] for it in rec_rsp], axis=1)
rec_df.index.name = 'Id'

top100 = top100.set_index('Id', drop=False).join(rec_df).reset_index(drop=True)
display(top100)


# =============================================================================
# SECTION 3 — PRICE HISTORY & TECHNICAL INDICATORS
# 365 days of daily closes. Indicators calculated once and reused
# across both strategies and the current signal table.
# =============================================================================

price_item = bq.data.px_last(dates=bq.func.range('-365D', '0D'), fill='NA')
response = bq.execute(bql.Request(bq.univ.list(id_list), price_item))
price = (
    response[0].df().dropna().reset_index()
    .pivot(index='DATE', columns='ID', values=response[0].df().columns[-1])
    .sort_index()
    .ffill()
)

sma50 = price.rolling(50).mean()
sma100 = price.rolling(100).mean()
sma200 = price.rolling(200).mean()
std50 = price.rolling(50).std()

def rsi(series, n=14):
    d = series.diff()
    rs = d.clip(lower=0).rolling(n).mean() / (-d).clip(lower=0).rolling(n).mean()
    return 100 - 100 / (1 + rs)

rsi14 = price.apply(rsi, n=14)
ema12 = price.ewm(span=12, adjust=False).mean()
ema26 = price.ewm(span=26, adjust=False).mean()
macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

price_to_sma50 = price / sma50
price_to_sma200 = price / sma200

latest_px = price.iloc[-1]
latest_sma_50 = sma50.iloc[-1]
latest_sma_100 = sma100.iloc[-1]
latest_rsi = rsi14.iloc[-1]
latest_macd_hist = macd_hist.iloc[-1]
latest_price_to_sma50 = price_to_sma50.iloc[-1]

sma_cross_up_hist = (sma50 > sma100) & (sma50.shift() <= sma100.shift())
sma_cross_down_hist = (sma50 < sma100) & (sma50.shift() >= sma100.shift())
sma_cross_up_now = (latest_sma_50 > latest_sma_100) & (sma50.iloc[-2] <= sma100.iloc[-2])
sma_cross_down_now = (latest_sma_50 < latest_sma_100) & (sma50.iloc[-2] >= sma100.iloc[-2])

ret = price.pct_change().fillna(0)


# =============================================================================
# SECTION 4 — BENCHMARK & REGIME
# SPY for benchmarking and the trend check.
# True stress = SPY below SMA200 AND VIX above 25 (both conditions required).
# Elevated VIX alone (20-25) only tightens the Tier 3 entry bar slightly.
# =============================================================================

start = price.index[0].strftime('%Y-%m-%d')
end = price.index[-1].strftime('%Y-%m-%d')

px_item = bq.data.px_last(dates=bq.func.range(start, end), fill='PREV')

def fetch_series(ticker):
    rsp = bq.execute(bql.Request(bq.univ.list([ticker]), px_item))
    return (rsp[0].df().reset_index()
            .pivot(index='DATE', columns='ID', values=rsp[0].df().columns[-1])
            .squeeze().sort_index().ffill())

spy_ser = fetch_series('SPY US Equity')
vix_ser = fetch_series('VIX Index')
rf_ser = fetch_series('USGG10YR Index')

rf_annual = rf_ser.iloc[-1] / 100  # Bloomberg gives yield as e.g. 4.3, convert to decimal

spy_aligned = spy_ser.reindex(price.index).ffill()
vix_aligned = vix_ser.reindex(price.index).ffill()

spy_sma200 = spy_aligned.rolling(200).mean()
spy_below = spy_aligned < spy_sma200

true_stress = spy_below & (vix_aligned > 25)
mild_stress = (vix_aligned > 20) & (vix_aligned <= 25)
normal_days = ~true_stress & ~mild_stress

spy_eq = (1 + spy_ser.pct_change().fillna(0)).cumprod()


# =============================================================================
# SECTION 5 — PERFORMANCE STATS
# Sharpe uses daily mean/std * sqrt(252).
# CAGR uses actual start value so it works even if equity curve doesn't start at 1.
# =============================================================================

def perf_stats(eq, label='', freq=252):
    daily = eq.pct_change().dropna()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (freq / len(eq)) - 1
    vol = daily.std(ddof=1) * np.sqrt(freq)
    rf_daily = rf_annual / freq
    sharpe = ((daily.mean() - rf_daily) / daily.std(ddof=1)) * np.sqrt(freq) if daily.std(ddof=1) > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    win_r = (daily > 0).mean()
    if label:
        print(f"  [{label}] days={len(eq)} | end={eq.iloc[-1]:.4f} | daily_mean={daily.mean():.5f} daily_std={daily.std(ddof=1):.5f}")
    return pd.Series({'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': mdd, 'Win Rate': win_r})

spy_stats = perf_stats(spy_eq, label='SPY')


# =============================================================================
# SECTION 6 — STRATEGY 1: TECHNICAL ONLY (BASELINE)
# Pure price-action on the screener universe. Entry on oversold pullback
# or SMA50/100 golden cross. Exit on extension or death cross.
# Used as a baseline to compare against the value-tiered strategy.
# =============================================================================

STD_MULT = 0.3
RSI_ENTRY = 35
RSI_EXIT = 65

tech_entries = (((price < sma50 - STD_MULT * std50) & (rsi14 < RSI_ENTRY)) | sma_cross_up_hist)
tech_exits = (((price > sma50 + STD_MULT * std50) & (rsi14 > RSI_EXIT)) | sma_cross_down_hist)
tech_entries = tech_entries & ~tech_entries.shift().fillna(False)
tech_exits = tech_exits & ~tech_exits.shift().fillna(False)

pos_tech = pd.DataFrame(False, index=price.index, columns=price.columns)
pos_tech[tech_entries] = True
pos_tech = pos_tech.where(~tech_exits, False).ffill().fillna(False).astype(int)

weights_tech = pos_tech.div(pos_tech.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
port_ret_tech = (weights_tech.shift() * ret).sum(axis=1)
equity_tech = (1 + port_ret_tech).cumprod()
stats_tech = perf_stats(equity_tech, label='Tech Only')

tech_buy_now = (((latest_px < latest_sma_50 - STD_MULT * std50.iloc[-1]) & (latest_rsi < RSI_ENTRY)) | sma_cross_up_now)
tech_sell_now = (((latest_px > latest_sma_50 + STD_MULT * std50.iloc[-1]) & (latest_rsi > RSI_EXIT)) | sma_cross_down_now)
tech_signal = pd.Series('NEUTRAL', index=latest_px.index)
tech_signal[tech_buy_now] = 'BUY'
tech_signal[tech_sell_now] = 'SELL'

top100_tech = top100.copy()
top100_tech['Signal'] = top100_tech['Id'].map(tech_signal)
top100_tech['Price'] = top100_tech['Id'].map(latest_px)
top100_tech['SMA50'] = top100_tech['Id'].map(latest_sma_50)
top100_tech['RSI14'] = top100_tech['Id'].map(latest_rsi)

print(f"\nTechnical Only — BUY: {tech_buy_now.sum()}  SELL: {tech_sell_now.sum()}  NEUTRAL: {(tech_signal == 'NEUTRAL').sum()}")
display(top100_tech[['Ticker', 'Name', 'Sector', 'Value Pctle', 'Price', 'SMA50', 'RSI14', 'Signal']])


# =============================================================================
# SECTION 7 — STRATEGY 2: VALUE + TECHNICAL (TIERED)
# Entry bar is tiered by Value Pctle. Higher conviction value names get
# a looser technical requirement. Moderate value needs stronger confirmation.
#
# Tier 1 — Value Pctle > 80  : P/SMA50 < 1.25, RSI < 72
# Tier 2 — Value Pctle 60-80 : P/SMA50 < 1.20, RSI < 65, MACD > 0
# Tier 3 — Value Pctle 50-60 : P/SMA50 < 1.10, RSI < 58
#
# Bars tighten during true stress (SPY < SMA200 and VIX > 25):
# Tier 1 : P/SMA50 < 1.15, RSI < 65
# Tier 2 : P/SMA50 < 1.10, RSI < 58, MACD > 0
# Tier 3 : P/SMA50 < 1.00, RSI < 50
#
# Exit: P/SMA50 > 1.35 (extended), RSI > 80 (overbought), P/SMA50 < 0.92 (stop loss)
# Sizing: fixed dollar per trade
# =============================================================================

POSITION_SIZE = 1000

deep_ids = [x for x in top100.loc[top100['Value Pctle'] > 80, 'Id'] if x in price.columns]
good_ids = [x for x in top100.loc[(top100['Value Pctle'] > 60) & (top100['Value Pctle'] <= 80), 'Id'] if x in price.columns]
mod_ids = [x for x in top100.loc[(top100['Value Pctle'] > 50) & (top100['Value Pctle'] <= 60), 'Id'] if x in price.columns]

print(f"\nTier 1 (Deep Value >80):   {len(deep_ids)} names")
print(f"Tier 2 (Good Value 60-80): {len(good_ids)} names")
print(f"Tier 3 (Moderate 50-60):   {len(mod_ids)} names")

t1_std = (price_to_sma50 < 1.25) & (rsi14 < 72)
t2_std = (price_to_sma50 < 1.20) & (rsi14 < 65) & (macd_hist > 0)
t3_std = (price_to_sma50 < 1.10) & (rsi14 < 58)
t1_tight = (price_to_sma50 < 1.15) & (rsi14 < 65)
t2_tight = (price_to_sma50 < 1.10) & (rsi14 < 58) & (macd_hist > 0)
t3_tight = (price_to_sma50 < 1.00) & (rsi14 < 50)

exit_cond = (price_to_sma50 > 1.35) | (rsi14 > 80) | (price_to_sma50 < 0.92)

pos_val = pd.DataFrame(0, index=price.index, columns=price.columns)

for ids, t_std, t_tight in [(deep_ids, t1_std, t1_tight), (good_ids, t2_std, t2_tight), (mod_ids, t3_std, t3_tight)]:
    if not ids:
        continue
    entry_raw = (
        (normal_days.to_frame().values * t_std[ids].values.astype(int)) |
        (true_stress.to_frame().values * t_tight[ids].values.astype(int))
    ).astype(bool)
    entry_df = pd.DataFrame(entry_raw, index=price.index, columns=ids)
    state = pd.DataFrame(np.nan, index=price.index, columns=ids)
    state[entry_df] = 1
    state[exit_cond[ids]] = 0
    pos_val[ids] = state.ffill().fillna(0).astype(int)

capital_deployed = pos_val.shift().sum(axis=1) * POSITION_SIZE
dollar_pnl = (pos_val.shift() * ret * POSITION_SIZE).sum(axis=1)
port_ret_val = (dollar_pnl / capital_deployed.replace(0, np.nan)).fillna(0)
equity_val = (1 + port_ret_val).cumprod()
stats_val = perf_stats(equity_val, label='Value+Tech')

active_days = (pos_val.sum(axis=1) > 0).sum()
print(f"\nActive position days: {active_days} of {len(price)} ({active_days / len(price):.1%})")
print(f"Avg positions held: {pos_val.sum(axis=1).mean():.1f}")
print(f"Peak capital deployed: ${int(pos_val.sum(axis=1).max() * POSITION_SIZE):,}")
print(f"\nBacktest Performance:")
print(stats_val.round(4))


# =============================================================================
# SECTION 8 — TODAY'S SIGNALS
# Same tiered logic applied to the latest snapshot.
# Outputs a clean action list sorted by signal type.
# =============================================================================

current_vix_val = float(vix_aligned.iloc[-1])
current_spy_ok = bool(spy_aligned.iloc[-1] > spy_sma200.iloc[-1])
true_stress_now = (not current_spy_ok) and (current_vix_val > 25)

tech_snap = pd.DataFrame({
    'Price': latest_px,
    'SMA50': latest_sma_50,
    'RSI14': latest_rsi,
    'MACD_Hist': latest_macd_hist,
    'Price_to_SMA50': latest_price_to_sma50,
}).rename_axis('Id')

results = top100.merge(tech_snap, how='left', on='Id')

def get_signal(row):
    vp = row['Value Pctle']
    p2s = row['Price_to_SMA50']
    rsi_v = row['RSI14']
    macd = row['MACD_Hist']

    if pd.isna(p2s) or pd.isna(rsi_v):
        return 'NO DATA'

    if p2s > 1.35:
        return 'EXIT — extended'
    if rsi_v > 80:
        return 'EXIT — overbought'
    if p2s < 0.92:
        return 'EXIT — stop loss'

    stress = ' [stress]' if true_stress_now else ''

    if vp > 80:
        bar = (p2s < 1.15 and rsi_v < 65) if true_stress_now else (p2s < 1.25 and rsi_v < 72)
        if bar:
            return f'BUY — Tier 1 Deep Value{stress}'

    if 60 < vp <= 80:
        bar = (p2s < 1.10 and rsi_v < 58 and macd > 0) if true_stress_now else (p2s < 1.20 and rsi_v < 65 and macd > 0)
        if bar:
            return f'BUY — Tier 2 Good Value{stress}'

    if 50 < vp <= 60:
        bar = (p2s < 1.00 and rsi_v < 50) if true_stress_now else (p2s < 1.10 and rsi_v < 58)
        if bar:
            return f'BUY — Tier 3 Moderate{stress}'

    return 'HOLD / WATCH'

results['Signal'] = results.apply(get_signal, axis=1)
results['P/SMA50'] = results['Price_to_SMA50'].round(3)

buys = results[results['Signal'].str.startswith('BUY', na=False)].sort_values('Value Pctle', ascending=False)
exits = results[results['Signal'].str.startswith('EXIT', na=False)].sort_values('Value Pctle', ascending=False)

print(f"\n{'='*70}")
print(f"TODAY'S SIGNALS  —  {pd.Timestamp.today().strftime('%B %d, %Y')}")
print(f"Regime: SPY {'ABOVE' if current_spy_ok else 'BELOW'} SMA200  |  VIX {current_vix_val:.1f}  |  {'STRESS' if true_stress_now else 'NORMAL'}")
print(f"{'='*70}")

if not buys.empty:
    print("\n  BUY SIGNALS:")
    for _, r in buys.iterrows():
        print(f"    {r['Ticker']:<8}  {r['Name']:<35}  Value: {r['Value Pctle']:.0f}p  P/SMA50: {r['P/SMA50']:.3f}  RSI: {r['RSI14']:.0f}  |  {r['Signal']}")

if not exits.empty:
    print("\n  EXIT SIGNALS:")
    for _, r in exits.iterrows():
        print(f"    {r['Ticker']:<8}  {r['Name']:<35}  Value: {r['Value Pctle']:.0f}p  P/SMA50: {r['P/SMA50']:.3f}  RSI: {r['RSI14']:.0f}  |  {r['Signal']}")

print("\nFull Table:")
display(results[['Ticker', 'Name', 'Sector', 'Value Pctle', 'Profitability Pctle', 'Price', 'SMA50', 'P/SMA50', 'RSI14', 'Signal']].sort_values('Signal'))
print("\nBreakdown:")
print(results['Signal'].value_counts().to_string())


# =============================================================================
# SECTION 9 — PERFORMANCE COMPARISON & PLOT
# =============================================================================

comparison = pd.DataFrame({
    'Technical Only': stats_tech,
    'Value + Tech': stats_val,
    'SPY': spy_stats,
}).T

print(f"\n{'='*70}")
print("PERFORMANCE COMPARISON")
print(f"{'='*70}")
print(comparison.round(4))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(equity_tech.index, equity_tech.values, label='Technical Only', linewidth=2, color='tab:blue')
ax1.plot(equity_val.index, equity_val.values, label='Value + Tech (Tiered)', linewidth=2, color='tab:green')
ax1.plot(spy_eq.index, spy_eq.values, label='SPY', linewidth=2, color='tab:orange', linestyle='--')

for d, s in true_stress.items():
    if s:
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.12, color='red')

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles + [Patch(facecolor='red', alpha=0.3, label='Stress Period')],
           labels + ['Stress Period'], fontsize=10, loc='upper left')

ax1.set_title('Strategy Performance vs SPY', fontsize=14)
ax1.set_ylabel('Cumulative Return')
ax1.grid(True, alpha=0.3)
ax1.text(0.01, 0.84,
    f"Value+Tech : Sharpe={stats_val['Sharpe']:.2f}  CAGR={stats_val['CAGR']:.1%}  MaxDD={stats_val['MaxDD']:.1%}\n"
    f"Tech Only  : Sharpe={stats_tech['Sharpe']:.2f}  CAGR={stats_tech['CAGR']:.1%}  MaxDD={stats_tech['MaxDD']:.1%}\n"
    f"SPY        : Sharpe={spy_stats['Sharpe']:.2f}  CAGR={spy_stats['CAGR']:.1%}  MaxDD={spy_stats['MaxDD']:.1%}",
    transform=ax1.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='left',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.plot(vix_aligned.index, vix_aligned.values, color='purple', linewidth=1.2, label='VIX')
ax2.axhline(25, color='red', linestyle='--', linewidth=1, label='VIX 25')
ax2.axhline(20, color='orange', linestyle='--', linewidth=1, label='VIX 20')
ax2.fill_between(vix_aligned.index, 25, vix_aligned.values, where=(vix_aligned > 25), alpha=0.2, color='red')
ax2.set_ylabel('VIX')
ax2.set_xlabel('Date')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 10 — TRADE LOG
# Every entry/exit pair with realized return and dollar P&L.
# Open positions show mark-to-market vs last available price.
# =============================================================================

def build_trade_log(pos_df, price_df):
    records = []
    for col in pos_df.columns:
        pos_s = pos_df[col]
        px_s = price_df[col]
        entries = pos_s.index[(pos_s == 1) & (pos_s.shift(1).fillna(0) == 0)]
        exits = pos_s.index[(pos_s == 0) & (pos_s.shift(1).fillna(0) == 1)]
        for e in entries:
            future = exits[exits > e]
            x = future[0] if len(future) else pos_s.index[-1]
            status = 'Closed' if len(future) else 'Open'
            epx = px_s.get(e, np.nan)
            xpx = px_s.get(x, np.nan)
            ret_v = (xpx / epx - 1) if (not np.isnan(epx) and epx > 0) else np.nan
            records.append({
                'Ticker': col,
                'Entry Date': e.date(),
                'Exit Date': x.date() if status == 'Closed' else '—',
                'Days Held': (x - e).days,
                'Entry Px': round(epx, 2) if not np.isnan(epx) else np.nan,
                'Current/Exit Px': round(xpx, 2) if not np.isnan(xpx) else np.nan,
                'Return %': round((ret_v or 0) * 100, 2),
                'Status': status,
            })
    return pd.DataFrame(records).sort_values('Entry Date').reset_index(drop=True) if records else pd.DataFrame()

tlog = build_trade_log(pos_val, price)

if not tlog.empty:
    closed = tlog[tlog['Status'] == 'Closed']
    open_pos = tlog[tlog['Status'] == 'Open']

    print(f"\n{'='*70}")
    print("TRADE LOG")
    print(f"{'='*70}")
    print(f"Total: {len(tlog)} trades  ({len(closed)} closed  /  {len(open_pos)} open)")

    if not closed.empty:
        cw = closed[closed['Return %'] > 0]
        print(f"Win Rate   : {len(cw) / len(closed):.1%}")
        print(f"Avg Return : {closed['Return %'].mean():.2f}%  (${closed['Return %'].mean() * POSITION_SIZE / 100:,.0f} at ${POSITION_SIZE:,}/trade)")
        print(f"Avg Hold   : {closed['Days Held'].mean():.0f} days")
        print(f"Best       : {closed['Return %'].max():.2f}%  (${closed['Return %'].max() * POSITION_SIZE / 100:,.0f})  —  {closed.loc[closed['Return %'].idxmax(), 'Ticker']}")
        print(f"Worst      : {closed['Return %'].min():.2f}%  (${closed['Return %'].min() * POSITION_SIZE / 100:,.0f})  —  {closed.loc[closed['Return %'].idxmin(), 'Ticker']}")
        print(f"Total P&L  : ${(closed['Return %'] * POSITION_SIZE / 100).sum():,.0f}")

        closed = closed.copy()
        closed[f'P&L (${POSITION_SIZE:,})'] = (closed['Return %'] * POSITION_SIZE / 100).round(0).astype(int)
        print("\nClosed Trades:")
        display(closed[['Ticker', 'Entry Date', 'Exit Date', 'Days Held', 'Entry Px', 'Current/Exit Px', 'Return %', f'P&L (${POSITION_SIZE:,})']])

    if not open_pos.empty:
        open_pos = open_pos.copy()
        open_pos[f'Unreal P&L (${POSITION_SIZE:,})'] = (open_pos['Return %'] * POSITION_SIZE / 100).round(0).astype(int)
        print(f"\nOpen Positions  (mark-to-market  {price.index[-1].date()}):")
        display(open_pos[['Ticker', 'Entry Date', 'Days Held', 'Entry Px', 'Current/Exit Px', 'Return %', f'Unreal P&L (${POSITION_SIZE:,})']])