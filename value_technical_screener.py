"""
Value + Technical Signal Strategy  |  v2.1
Multi-factor screener across Russell 3000.
Combines fundamental value rankings with technical entry/exit signals
to identify long opportunities on a daily basis.

Factor groups: Value, Profitability, Momentum
Composite models: Percentile, Z-score, PCA
Relative value: vs Self (2Y EWM), vs Sector (mkt-cap weighted), vs Index

Universe rebalanced quarterly using point-in-time snapshots to eliminate
forward-looking bias. Entry at top-150 rank; exit buffer at 200.

Entry bars tiered by Value Weighted Z. Pyramid position manager:
states 0 / 0.5 / 1.0 / 1.5 / 2.0 with regime overlay.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from scipy.stats import zscore as scipy_zscore
import bql

bq = bql.Service()
as_of_date = __import__('datetime').date.today()
params = {'dates': as_of_date, 'fill': 'prev'}
range_3y = bq.func.range(-2, 0)


# =============================================================================
# SECTION 1 — FACTOR DEFINITIONS
# =============================================================================

value = {
    'FCF Yield': {'item': bq.data.fcf_ev_yield(**params), 'col_name': 'FCF Yield'},
    'Px/Sales':  {'item': bq.data.px_to_sales_ratio(**params), 'multiplier': -1, 'col_name': 'Px/Sales'},
    'PE':        {'item': bq.data.pe_ratio(**params), 'multiplier': -1, 'col_name': 'PE'},
    'EV/EBITDA': {'item': bq.data.ev_to_ebitda(**params), 'multiplier': -1, 'col_name': 'EV/EBITDA'},
}

profitability = {
    'Avg ROE 3Y':  {'item': bq.data.return_com_eqy(**params, fa_period_offset=range_3y).avg(), 'col_name': 'Avg ROE 3Y'},
    'Avg ROIC 3Y': {'item': bq.data.RETURN_ON_INV_CAPITAL(**params, fa_period_offset=range_3y).avg(), 'col_name': 'Avg ROIC 3Y'},
    'Op Mrg Chg':  {'item': bq.data.OPER_MARGIN(**params, fpo='1') - bq.data.OPER_MARGIN(**params, fpo='0'), 'col_name': 'Op Mrg Chg'},
}

momentum = {
    'Mom 12-1': {'item': bq.data.total_return(calc_interval=bq.func.range(f'{as_of_date}-12m', f'{as_of_date}-1m')), 'col_name': 'Mom 12-1'},
    'Mom 6-1':  {'item': bq.data.total_return(calc_interval=bq.func.range(f'{as_of_date}-6m',  f'{as_of_date}-1m')), 'col_name': 'Mom 6-1'},
    'Mom 3-1':  {'item': bq.data.total_return(calc_interval=bq.func.range(f'{as_of_date}-3m',  f'{as_of_date}-1m')), 'col_name': 'Mom 3-1'},
}

display_items = {
    'Name':   bq.data.name(),
    'Ticker': bq.data.ticker(),
    'Id':     bq.data.id(),
    'Sector': bq.data.classification_name(),
}


# =============================================================================
# SECTION 2 — QUARTERLY UNIVERSE ENGINE (POINT-IN-TIME)
# =============================================================================

REBALANCE_FREQ  = 'QS'
BACKTEST_START  = (pd.Timestamp.today() - pd.DateOffset(days=365)).strftime('%Y-%m-%d')
BACKTEST_END    = pd.Timestamp.today().strftime('%Y-%m-%d')

rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq=REBALANCE_FREQ)
if pd.Timestamp(BACKTEST_START).normalize() not in rebalance_dates:
    rebalance_dates = pd.DatetimeIndex([pd.Timestamp(BACKTEST_START)]).append(rebalance_dates)
if pd.Timestamp.today().normalize() not in rebalance_dates:
    rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([pd.Timestamp.today().normalize()]))


def run_screener(as_of):
    date_str = as_of.strftime('%Y-%m-%d') if hasattr(as_of, 'strftime') else as_of
    p    = {'dates': date_str, 'fill': 'prev'}
    r3y  = bq.func.range(-2, 0)

    val = {
        'FCF Yield':  {'item': bq.data.fcf_ev_yield(**p),                                               'col_name': 'FCF Yield'},
        'Px/Sales':   {'item': bq.data.px_to_sales_ratio(**p),          'multiplier': -1,               'col_name': 'Px/Sales'},
        'PE':         {'item': bq.data.pe_ratio(**p),                   'multiplier': -1,               'col_name': 'PE'},
        'EV/EBITDA':  {'item': bq.data.ev_to_ebitda(**p),               'multiplier': -1,               'col_name': 'EV/EBITDA'},
    }
    prof = {
        'Avg ROE 3Y':  {'item': bq.data.return_com_eqy(**p, fa_period_offset=r3y).avg(),                'col_name': 'Avg ROE 3Y'},
        'Avg ROIC 3Y': {'item': bq.data.RETURN_ON_INV_CAPITAL(**p, fa_period_offset=r3y).avg(),         'col_name': 'Avg ROIC 3Y'},
        'Op Mrg Chg':  {'item': bq.data.OPER_MARGIN(**p, fpo='1') - bq.data.OPER_MARGIN(**p, fpo='0'), 'col_name': 'Op Mrg Chg'},
    }
    mom = {
        'Mom 12-1': {'item': bq.data.total_return(calc_interval=bq.func.range(f'{date_str}-12m', f'{date_str}-1m')), 'col_name': 'Mom 12-1'},
        'Mom 6-1':  {'item': bq.data.total_return(calc_interval=bq.func.range(f'{date_str}-6m',  f'{date_str}-1m')), 'col_name': 'Mom 6-1'},
        'Mom 3-1':  {'item': bq.data.total_return(calc_interval=bq.func.range(f'{date_str}-3m',  f'{date_str}-1m')), 'col_name': 'Mom 3-1'},
    }
    disp = {'Name': bq.data.name(), 'Ticker': bq.data.ticker(), 'Id': bq.data.id(), 'Sector': bq.data.classification_name()}

    def raw_expr(d):
        return d['item'] * d.get('multiplier', 1)

    all_factors = list(val.values()) + list(prof.values()) + list(mom.values())
    raw_items   = {d['col_name']: raw_expr(d) for d in all_factors}
    data_items  = {**disp, **raw_items}
    raw_cols    = list(raw_items.keys())

    univ = bq.univ.members('RAY Index', dates=date_str)
    req  = bql.Request(univ, data_items, with_params={'mode': 'cached'})
    rsp  = bq.execute(req)
    df   = pd.concat([it.df()[it.name] for it in rsp], axis=1).reset_index(drop=True)

    def pctle(s):
        return s.rank(pct=True) * 100

    def pctle_group(comp_dicts):
        return pd.concat([pctle(df[d['col_name']] * d.get('multiplier', 1)) for d in comp_dicts], axis=1).mean(axis=1)

    df['Value Pctle']         = pctle_group(val.values())
    df['Profitability Pctle'] = pctle_group(prof.values())
    df['Momentum Pctle']      = pctle_group(mom.values())
    df['Composite Pctle']     = df[['Value Pctle', 'Profitability Pctle', 'Momentum Pctle']].mean(axis=1)
    df['Rank Pctle']          = df['Composite Pctle'].rank(ascending=False)

    z_frame          = (df[raw_cols] - df[raw_cols].mean()) / df[raw_cols].std(ddof=0)
    df['Z Composite'] = z_frame.mean(axis=1)
    df['Rank Z']      = df['Z Composite'].rank(ascending=False)

    pc1                  = PCA(1).fit_transform(z_frame.fillna(0))[:, 0]
    df['PCA Composite']  = pc1
    df['Rank PCA']       = df['PCA Composite'].rank(ascending=False)

    df['Screener Date'] = date_str
    return df


def apply_buffer(df_new, df_current, entry_rank=150, exit_rank=200):
    entry_mask = (
        (df_new['Rank Pctle'] <= entry_rank) &
        (df_new['Rank Z']     <= entry_rank) &
        (df_new['Rank PCA']   <= entry_rank)
    )
    exit_mask = (
        (df_new['Rank Pctle'] > exit_rank) |
        (df_new['Rank Z']     > exit_rank) |
        (df_new['Rank PCA']   > exit_rank)
    )
    new_entrants = df_new[entry_mask]
    if df_current is not None and len(df_current) > 0:
        current_ids = set(df_current['Id'].tolist())
        retained    = df_new[df_new['Id'].isin(current_ids) & ~exit_mask]
        combined    = pd.concat([new_entrants, retained]).drop_duplicates(subset='Id', keep='first')
    else:
        combined = new_entrants
    return combined.reset_index(drop=True)


print(f"Running quarterly screener across {len(rebalance_dates)} rebalance dates...")
universe_snapshots = {}
prev_snapshot      = None

for rb_date in rebalance_dates:
    date_str  = rb_date.strftime('%Y-%m-%d')
    print(f"  Scoring {date_str}...", end=' ')
    df_scored = run_screener(rb_date)
    snapshot  = apply_buffer(df_scored, prev_snapshot)
    universe_snapshots[date_str] = snapshot
    prev_snapshot = snapshot
    print(f"{len(snapshot)} names")

latest_snapshot_date = sorted(universe_snapshots.keys())[-1]
top100   = universe_snapshots[latest_snapshot_date]
id_list  = top100['Id'].tolist()
all_ids  = list(set(id for snap in universe_snapshots.values() for id in snap['Id'].tolist()))

print(f"\nToday's universe: {len(top100)} names")
print(f"All-time unique IDs: {len(all_ids)}")

date_range_full        = pd.date_range(BACKTEST_START, BACKTEST_END, freq='B')
active_universe_by_date = {}
sorted_rb_dates         = sorted(universe_snapshots.keys())

for d in date_range_full:
    d_str      = d.strftime('%Y-%m-%d')
    applicable = [rb for rb in sorted_rb_dates if rb <= d_str]
    active_universe_by_date[d] = set(universe_snapshots[applicable[-1]]['Id'].tolist()) if applicable else set()

print(f"Daily universe lookup built across {len(date_range_full)} trading days")

rec_items = {
    'Tot Buy':  bq.data.tot_buy_rec(dates=as_of_date),
    'Tot Hold': bq.data.tot_hold_rec(dates=as_of_date),
    'Tot Sell': bq.data.tot_sell_rec(dates=as_of_date),
    'Tot Recs': bq.data.tot_analyst_rec(dates=as_of_date),
}
rec_rsp = bq.execute(bql.Request(bq.univ.list(id_list), rec_items, with_params={'mode': 'cached'}))
rec_df  = pd.concat([it.df()[it.name] for it in rec_rsp], axis=1)
rec_df.index.name = 'Id'

top100 = top100.set_index('Id', drop=False).join(rec_df).reset_index(drop=True)
display(top100)


# =============================================================================
# SECTION 2B — RELATIVE VALUE Z-SCORES
# vs Self (2Y EWM), vs Sector (mkt-cap weighted), vs Index (cross-sectional)
# Blended equally into Value Weighted Z for tier assignment.
# =============================================================================

value_col_names = [d['col_name'] for d in value.values()]

val_history_params = {'dates': bq.func.range('-730D', '0D'), 'fill': 'PREV'}
val_history_items  = {
    'FCF Yield':   bq.data.fcf_ev_yield(**val_history_params),
    'Px/Sales':    bq.data.px_to_sales_ratio(**val_history_params) * -1,
    'PE':          bq.data.pe_ratio(**val_history_params) * -1,
    'EV/EBITDA':   bq.data.ev_to_ebitda(**val_history_params) * -1,
    'Cur Mkt Cap': bq.data.cur_mkt_cap(**val_history_params, currency='USD'),
}

print("Pulling 2Y value metric history...")
hist_rsp    = bq.execute(bql.Request(bq.univ.list(all_ids), val_history_items, with_params={'mode': 'cached'}))
hist_frames = {}
for item in hist_rsp:
    df_raw   = item.df().reset_index()
    date_col = 'DATE' if 'DATE' in df_raw.columns else 'AS_OF_DATE'
    hist_frames[item.name] = (
        df_raw.pivot(index=date_col, columns='ID', values=item.name)
        .sort_index().ffill()
    )

vs_self_z = {}
for col in value_col_names:
    series_df = hist_frames[col]
    ewm_mean  = series_df.ewm(span=252, min_periods=60).mean()
    ewm_std   = series_df.ewm(span=252, min_periods=60).std()
    z         = (series_df - ewm_mean) / ewm_std.replace(0, np.nan)
    vs_self_z[col] = z.iloc[-1]

vs_self_df        = pd.DataFrame(vs_self_z); vs_self_df.index.name = 'Id'
vs_self_composite = vs_self_df.mean(axis=1).rename('VS Self Z')

sector_map      = top100.set_index('Id')['Sector'].to_dict()
latest_mktcap   = hist_frames['Cur Mkt Cap'].iloc[-1].rename('Mkt Cap')

vs_sector_z = {}
for col in value_col_names:
    latest_vals = hist_frames[col].iloc[-1].rename(col)
    tmp         = pd.DataFrame({'Value': latest_vals, 'Mkt Cap': latest_mktcap})
    tmp['Sector'] = tmp.index.map(sector_map)
    tmp = tmp.dropna(subset=['Sector', 'Value', 'Mkt Cap'])

    def wavg(g):
        return (g['Value'] * g['Mkt Cap']).sum() / g['Mkt Cap'].sum()

    sector_avg      = tmp.groupby('Sector').apply(wavg).rename('Sector Avg')
    tmp             = tmp.join(sector_avg, on='Sector')
    tmp['vs Sector'] = tmp['Value'] / tmp['Sector Avg']
    tmp['vs Sector Z'] = tmp.groupby('Sector')['vs Sector'].transform(
        lambda x: scipy_zscore(x, nan_policy='omit')
    )
    vs_sector_z[col] = tmp['vs Sector Z']

vs_sector_df        = pd.DataFrame(vs_sector_z); vs_sector_df.index.name = 'Id'
vs_sector_composite = vs_sector_df.mean(axis=1).rename('VS Sector Z')

vs_index_composite = top100.set_index('Id')['Z Composite'].rename('VS Index Z')

VS_SELF_WT, VS_SECTOR_WT, VS_INDEX_WT = 0.33, 0.33, 0.33

rel_val = pd.DataFrame({
    'VS Self Z':   vs_self_composite,
    'VS Sector Z': vs_sector_composite,
    'VS Index Z':  vs_index_composite,
}).reindex(top100['Id'].values)

rel_val['Value Weighted Z'] = (
    rel_val['VS Self Z'].fillna(0)   * VS_SELF_WT   +
    rel_val['VS Sector Z'].fillna(0) * VS_SECTOR_WT +
    rel_val['VS Index Z'].fillna(0)  * VS_INDEX_WT
)

top100 = top100.merge(rel_val.reset_index(), on='Id', how='left')
top100['Value WZ Pctle'] = top100['Value Weighted Z'].rank(pct=True) * 100

print(f"\nRelative value z-scores: {rel_val['Value Weighted Z'].notna().sum()} names")
display(top100[['Ticker', 'Name', 'Sector', 'Value Pctle', 'Value Weighted Z', 'Value WZ Pctle',
                'VS Self Z', 'VS Sector Z', 'VS Index Z']].sort_values('Value Weighted Z', ascending=False))


# =============================================================================
# SECTION 3 — PRICE HISTORY & TECHNICAL INDICATORS
# =============================================================================

price_item = bq.data.px_last(dates=bq.func.range('-365D', '0D'), fill='NA')
response   = bq.execute(bql.Request(bq.univ.list(all_ids), price_item))
price = (
    response[0].df().dropna().reset_index()
    .pivot(index='DATE', columns='ID', values=response[0].df().columns[-1])
    .sort_index().ffill()
)

sma50  = price.rolling(50).mean()
sma100 = price.rolling(100).mean()
sma200 = price.rolling(200).mean()
std50  = price.rolling(50).std()

def rsi(series, n=14):
    d  = series.diff()
    rs = d.clip(lower=0).rolling(n).mean() / (-d).clip(lower=0).rolling(n).mean()
    return 100 - 100 / (1 + rs)

rsi14     = price.apply(rsi, n=14)
ema12     = price.ewm(span=12, adjust=False).mean()
ema26     = price.ewm(span=26, adjust=False).mean()
macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

price_to_sma50  = price / sma50
price_to_sma200 = price / sma200

latest_px           = price.iloc[-1]
latest_sma_50       = sma50.iloc[-1]
latest_sma_100      = sma100.iloc[-1]
latest_rsi          = rsi14.iloc[-1]
latest_macd_hist    = macd_hist.iloc[-1]
latest_price_to_sma50 = price_to_sma50.iloc[-1]

sma_cross_up_hist   = (sma50 > sma100) & (sma50.shift() <= sma100.shift())
sma_cross_down_hist = (sma50 < sma100) & (sma50.shift() >= sma100.shift())
sma_cross_up_now    = (latest_sma_50 > latest_sma_100) & (sma50.iloc[-2] <= sma100.iloc[-2])
sma_cross_down_now  = (latest_sma_50 < latest_sma_100) & (sma50.iloc[-2] >= sma100.iloc[-2])

ret = price.pct_change().fillna(0)


# =============================================================================
# SECTION 4 — BENCHMARK & REGIME
# =============================================================================

start = price.index[0].strftime('%Y-%m-%d')
end   = price.index[-1].strftime('%Y-%m-%d')
px_item = bq.data.px_last(dates=bq.func.range(start, end), fill='PREV')

def fetch_series(ticker):
    rsp = bq.execute(bql.Request(bq.univ.list([ticker]), px_item))
    return (rsp[0].df().reset_index()
            .pivot(index='DATE', columns='ID', values=rsp[0].df().columns[-1])
            .squeeze().sort_index().ffill())

spy_ser = fetch_series('SPY US Equity')
vix_ser = fetch_series('VIX Index')
rf_ser  = fetch_series('USGG10YR Index')

rf_annual   = rf_ser.iloc[-1] / 100
spy_aligned = spy_ser.reindex(price.index).ffill()
vix_aligned = vix_ser.reindex(price.index).ffill()
spy_sma200  = spy_aligned.rolling(200).mean()
spy_below   = spy_aligned < spy_sma200

true_stress  = spy_below & (vix_aligned > 25)
mild_stress  = (vix_aligned > 20) & (vix_aligned <= 25)
normal_days  = ~true_stress & ~mild_stress

spy_eq = (1 + spy_ser.pct_change().fillna(0)).cumprod()


# =============================================================================
# SECTION 5 — PERFORMANCE STATS
# =============================================================================

def perf_stats(eq, label='', freq=252):
    daily    = eq.pct_change().dropna()
    rf_daily = rf_annual / freq
    cagr     = (eq.iloc[-1] / eq.iloc[0]) ** (freq / len(eq)) - 1
    vol      = daily.std(ddof=1) * np.sqrt(freq)
    sharpe   = ((daily.mean() - rf_daily) / daily.std(ddof=1)) * np.sqrt(freq) if daily.std(ddof=1) > 0 else np.nan
    mdd      = (eq / eq.cummax() - 1).min()
    win_r    = (daily > 0).mean()
    if label:
        print(f"  [{label}] days={len(eq)} | end={eq.iloc[-1]:.4f} | daily_mean={daily.mean():.5f} daily_std={daily.std(ddof=1):.5f}")
    return pd.Series({'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': mdd, 'Win Rate': win_r})

spy_stats = perf_stats(spy_eq, label='SPY')


# =============================================================================
# SECTION 6 — STRATEGY 1: TECHNICAL ONLY (BASELINE)
# =============================================================================

STD_MULT  = 0.3
RSI_ENTRY = 35
RSI_EXIT  = 65

tech_entries = (((price < sma50 - STD_MULT * std50) & (rsi14 < RSI_ENTRY)) | sma_cross_up_hist)
tech_exits   = (((price > sma50 + STD_MULT * std50) & (rsi14 > RSI_EXIT)) | sma_cross_down_hist)
tech_entries = tech_entries & ~tech_entries.shift().fillna(False)
tech_exits   = tech_exits   & ~tech_exits.shift().fillna(False)

pos_tech     = pd.DataFrame(False, index=price.index, columns=price.columns)
pos_tech[tech_entries] = True
pos_tech     = pos_tech.where(~tech_exits, False).ffill().fillna(False).astype(int)

weights_tech  = pos_tech.div(pos_tech.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
port_ret_tech = (weights_tech.shift() * ret).sum(axis=1)
equity_tech   = (1 + port_ret_tech).cumprod()
stats_tech    = perf_stats(equity_tech, label='Tech Only')

tech_buy_now  = (((latest_px < latest_sma_50 - STD_MULT * std50.iloc[-1]) & (latest_rsi < RSI_ENTRY)) | sma_cross_up_now)
tech_sell_now = (((latest_px > latest_sma_50 + STD_MULT * std50.iloc[-1]) & (latest_rsi > RSI_EXIT)) | sma_cross_down_now)
tech_signal   = pd.Series('NEUTRAL', index=latest_px.index)
tech_signal[tech_buy_now]  = 'BUY'
tech_signal[tech_sell_now] = 'SELL'

top100_tech = top100.copy()
top100_tech['Signal'] = top100_tech['Id'].map(tech_signal)
top100_tech['Price']  = top100_tech['Id'].map(latest_px)
top100_tech['SMA50']  = top100_tech['Id'].map(latest_sma_50)
top100_tech['RSI14']  = top100_tech['Id'].map(latest_rsi)

print(f"\nTechnical Only — BUY: {tech_buy_now.sum()}  SELL: {tech_sell_now.sum()}  NEUTRAL: {(tech_signal == 'NEUTRAL').sum()}")
display(top100_tech[['Ticker', 'Name', 'Sector', 'Value WZ Pctle', 'Price', 'SMA50', 'RSI14', 'Signal']])


# =============================================================================
# SECTION 7 — STRATEGY 2: VALUE + TECHNICAL (TIERED) — PYRAMID POSITION MANAGER
#
# Position states: 0=flat, 0.5=warning/partial, 1=base, 1.5=add1, 2=add2
# Dollar sizes:    $0 / $500 / $1000 / $1500 / $2000  (UNIT_SIZE = $1000)
#
# Regime:
#   Bull   (SPY > SMA200, VIX < 15)   — pyramid allowed, extended exit thresholds
#   Normal (SPY > SMA200, VIX 15-25)  — pyramid allowed, standard thresholds
#   Stress (SPY < SMA200 OR VIX > 25) — no new adds, tighten exits
# =============================================================================

UNIT_SIZE = 1000

all_snapshot_df = pd.concat(universe_snapshots.values(), ignore_index=True)
all_snapshot_df = all_snapshot_df.drop_duplicates(subset='Id', keep='last')

wz_map    = top100.set_index('Id')['Value WZ Pctle'].to_dict()
pctle_map = all_snapshot_df.set_index('Id')['Value Pctle'].to_dict()

def get_tier_score(id_):
    return wz_map.get(id_, pctle_map.get(id_, 0))

all_universe_ids = [x for x in all_snapshot_df['Id'].tolist() if x in price.columns]
tier_scores      = {id_: get_tier_score(id_) for id_ in all_universe_ids}

deep_ids = [id_ for id_, sc in tier_scores.items() if sc > 80]
good_ids = [id_ for id_, sc in tier_scores.items() if 60 < sc <= 80]
mod_ids  = [id_ for id_, sc in tier_scores.items() if 50 < sc <= 60]

print(f"\nTier 1 (Deep Value WZ >80):   {len(deep_ids)} names")
print(f"Tier 2 (Good Value WZ 60-80): {len(good_ids)} names")
print(f"Tier 3 (Moderate WZ 50-60):   {len(mod_ids)} names")

t1_std   = (price_to_sma50 < 1.25) & (rsi14 < 72)
t2_std   = (price_to_sma50 < 1.20) & (rsi14 < 65) & (macd_hist > 0)
t3_std   = (price_to_sma50 < 1.10) & (rsi14 < 58)
t1_tight = (price_to_sma50 < 1.15) & (rsi14 < 65)
t2_tight = (price_to_sma50 < 1.10) & (rsi14 < 58) & (macd_hist > 0)
t3_tight = (price_to_sma50 < 1.00) & (rsi14 < 50)

add1_cond = (
    (price_to_sma50 >= 1.05) & (price_to_sma50 <= 1.25) &
    (rsi14 < 72) & (macd_hist > 0)
)
macd_accel = macd_hist > macd_hist.shift(1)
add2_cond  = (
    (price_to_sma50 >= 1.12) & (price_to_sma50 <= 1.28) &
    (rsi14 < 75) & (macd_hist > 0) & macd_accel
)

bull_regime   = (spy_aligned > spy_sma200) & (vix_aligned < 15)
can_pyramid   = (spy_aligned > spy_sma200) & (vix_aligned <= 25)
stress_regime = ~can_pyramid

universe_gate = pd.DataFrame(False, index=price.index, columns=price.columns)
sorted_rb_dates = sorted(universe_snapshots.keys())

for price_date in price.index:
    d_str      = price_date.strftime('%Y-%m-%d')
    applicable = [rb for rb in sorted_rb_dates if rb <= d_str]
    if applicable:
        active_ids  = set(universe_snapshots[applicable[-1]]['Id'].tolist())
        cols_active = [c for c in active_ids if c in universe_gate.columns]
        universe_gate.loc[price_date, cols_active] = True

all_tier_ids = list(set(deep_ids + good_ids + mod_ids))
ids_in_price = [x for x in all_tier_ids if x in price.columns]

entry_signals = pd.DataFrame(False, index=price.index, columns=price.columns)
for ids, t_std, t_tight in [
    (deep_ids, t1_std, t1_tight),
    (good_ids, t2_std, t2_tight),
    (mod_ids,  t3_std, t3_tight)
]:
    iip = [x for x in ids if x in price.columns]
    if not iip:
        continue
    raw = (
        (normal_days.to_frame().values * t_std[iip].values.astype(int)) |
        (true_stress.to_frame().values * t_tight[iip].values.astype(int))
    ).astype(bool)
    gate = universe_gate[iip].values
    entry_signals[iip] = raw & gate

pos_val  = pd.DataFrame(0.0, index=price.index, columns=price.columns)
dates    = price.index.tolist()
n_dates  = len(dates)

for col in ids_in_price:
    p2s  = price_to_sma50[col].values
    rsi  = rsi14[col].values
    entr = entry_signals[col].values
    a1   = add1_cond[col].values
    a2   = add2_cond[col].values
    pyr  = can_pyramid.values
    st   = stress_regime.values

    state            = 0.0
    days_since_base  = 0
    days_since_add1  = 0
    col_pos          = pos_val[col].values

    for i in range(n_dates):
        ps = p2s[i]
        r  = rsi[i]

        if state >= 1.0: days_since_base += 1
        if state >= 1.5: days_since_add1 += 1

        if state == 2.0:
            if ps > 1.50 or r > 85 or ps < 1.08:
                state = 0.0; days_since_base = 0; days_since_add1 = 0
            elif ps < 1.12:
                state = 1.5; days_since_add1 = 0

        if state == 1.5:
            step_down_lvl = 1.02 if st[i] else 1.05
            full_exit_lvl = 0.92 if st[i] else 1.00
            if ps > 1.40 or r > 82 or ps < full_exit_lvl:
                state = 0.0; days_since_base = 0; days_since_add1 = 0
            elif ps < step_down_lvl:
                state = 1.0; days_since_add1 = 0

        if state == 1.0:
            if ps > 1.35 or r > 80 or ps < 0.92:
                state = 0.0; days_since_base = 0
            elif ps < 0.96:
                state = 0.5

        if state == 0.5:
            if ps < 0.90:
                state = 0.0; days_since_base = 0
            elif ps > 0.98:
                state = 1.0

        if state == 0.0 and entr[i]:
            state = 1.0; days_since_base = 1; days_since_add1 = 0
        elif state == 1.0 and pyr[i] and days_since_base >= 5 and a1[i]:
            state = 1.5; days_since_add1 = 1
        elif state == 1.5 and pyr[i] and days_since_add1 >= 5 and a2[i]:
            state = 2.0

        col_pos[i] = state

    pos_val[col] = col_pos

capital_deployed = pos_val.shift().sum(axis=1) * UNIT_SIZE
dollar_pnl       = (pos_val.shift() * ret * UNIT_SIZE).sum(axis=1)
port_ret_val     = (dollar_pnl / capital_deployed.replace(0, np.nan)).fillna(0)
equity_val       = (1 + port_ret_val).cumprod()
stats_val        = perf_stats(equity_val, label='Value+Tech')

state_counts = pos_val.stack().value_counts().sort_index()
print(f"\nPosition state distribution:")
for s, c in state_counts.items():
    labels = {0.0: 'Flat', 0.5: 'Warning', 1.0: 'Base', 1.5: 'Add1', 2.0: 'Add2'}
    print(f"  State {s} ({labels.get(s,'?')}): {int(c):,} name-days")

active_days = (pos_val.sum(axis=1) > 0).sum()
print(f"\nActive position days: {active_days} of {len(price)} ({active_days/len(price):.1%})")
print(f"Avg units deployed:   {pos_val.sum(axis=1).mean():.2f}")
print(f"Peak capital deployed: ${pos_val.sum(axis=1).max() * UNIT_SIZE:,.0f}")
print(f"\nBacktest Performance:")
print(stats_val.round(4))


# =============================================================================
# SECTION 8 — TODAY'S SIGNALS
# =============================================================================

current_vix_val  = float(vix_aligned.iloc[-1])
current_spy_ok   = bool(spy_aligned.iloc[-1] > spy_sma200.iloc[-1])
true_stress_now  = (not current_spy_ok) and (current_vix_val > 25)
can_pyr_now      = current_spy_ok and (current_vix_val <= 25)

tech_snap = pd.DataFrame({
    'Price':          latest_px,
    'SMA50':          latest_sma_50,
    'RSI14':          latest_rsi,
    'MACD_Hist':      latest_macd_hist,
    'Price_to_SMA50': latest_price_to_sma50,
}).rename_axis('Id')

results = top100.merge(tech_snap, how='left', on='Id')

def get_signal(row):
    vp    = row['Value WZ Pctle']
    p2s   = row['Price_to_SMA50']
    rsi_v = row['RSI14']
    macd  = row['MACD_Hist']

    if pd.isna(p2s) or pd.isna(rsi_v):
        return 'NO DATA'

    stress = ' [stress]' if true_stress_now else ''

    if p2s > 1.50 and rsi_v > 85: return 'EXIT — extended (Add2 level)'
    if p2s > 1.40 and rsi_v > 82: return 'EXIT — extended (Add1 level)'
    if p2s > 1.35:                 return 'EXIT — extended'
    if rsi_v > 80:                 return 'EXIT — overbought'
    if p2s < 0.90:                 return 'EXIT — stop loss (full)'
    if p2s < 0.92:                 return 'EXIT — stop loss'
    if p2s < 0.96:                 return 'TRIM — warning (partial exit)'

    if can_pyr_now and 1.12 <= p2s <= 1.28 and rsi_v < 75 and macd > 0:
        return 'ADD — Pyramid 2 (if held at Add1)'
    if can_pyr_now and 1.05 <= p2s <= 1.25 and rsi_v < 72 and macd > 0:
        return 'ADD — Pyramid 1 (if held at base)'

    if vp > 80:
        bar = (p2s < 1.15 and rsi_v < 65) if true_stress_now else (p2s < 1.25 and rsi_v < 72)
        if bar: return f'BUY — Tier 1 Deep Value{stress}'

    if 60 < vp <= 80:
        bar = (p2s < 1.10 and rsi_v < 58 and macd > 0) if true_stress_now else (p2s < 1.20 and rsi_v < 65 and macd > 0)
        if bar: return f'BUY — Tier 2 Good Value{stress}'

    if 50 < vp <= 60:
        bar = (p2s < 1.00 and rsi_v < 50) if true_stress_now else (p2s < 1.10 and rsi_v < 58)
        if bar: return f'BUY — Tier 3 Moderate{stress}'

    return 'HOLD / WATCH'

results['Signal'] = results.apply(get_signal, axis=1)
results['P/SMA50'] = results['Price_to_SMA50'].round(3)

buys  = results[results['Signal'].str.startswith('BUY',  na=False)].sort_values('Value WZ Pctle', ascending=False)
adds  = results[results['Signal'].str.startswith('ADD',  na=False)].sort_values('Value WZ Pctle', ascending=False)
trims = results[results['Signal'].str.startswith('TRIM', na=False)].sort_values('Value WZ Pctle', ascending=False)
exits = results[results['Signal'].str.startswith('EXIT', na=False)].sort_values('Value WZ Pctle', ascending=False)

print(f"\n{'='*70}")
print(f"TODAY'S SIGNALS  —  {pd.Timestamp.today().strftime('%B %d, %Y')}")
print(f"Regime: SPY {'ABOVE' if current_spy_ok else 'BELOW'} SMA200  |  VIX {current_vix_val:.1f}  |  {'STRESS' if true_stress_now else 'NORMAL'}")
print(f"{'='*70}")

if not buys.empty:
    print("\n  BUY SIGNALS:")
    for _, r in buys.iterrows():
        print(f"    {r['Ticker']:<8}  {r['Name']:<35}  Value WZ: {r['Value WZ Pctle']:.0f}p  P/SMA50: {r['P/SMA50']:.3f}  RSI: {r['RSI14']:.0f}  |  {r['Signal']}")

if not adds.empty:
    print("\n  PYRAMID ADD SIGNALS:")
    for _, r in adds.iterrows():
        print(f"    {r['Ticker']:<8}  {r['Name']:<35}  Value WZ: {r['Value WZ Pctle']:.0f}p  P/SMA50: {r['P/SMA50']:.3f}  RSI: {r['RSI14']:.0f}  |  {r['Signal']}")

if not trims.empty:
    print("\n  TRIM / WARNING SIGNALS:")
    for _, r in trims.iterrows():
        print(f"    {r['Ticker']:<8}  {r['Name']:<35}  Value WZ: {r['Value WZ Pctle']:.0f}p  P/SMA50: {r['P/SMA50']:.3f}  RSI: {r['RSI14']:.0f}  |  {r['Signal']}")

if not exits.empty:
    print("\n  EXIT SIGNALS:")
    for _, r in exits.iterrows():
        print(f"    {r['Ticker']:<8}  {r['Name']:<35}  Value WZ: {r['Value WZ Pctle']:.0f}p  P/SMA50: {r['P/SMA50']:.3f}  RSI: {r['RSI14']:.0f}  |  {r['Signal']}")

print("\nFull Table:")
display(results[['Ticker', 'Name', 'Sector', 'Value WZ Pctle', 'VS Self Z', 'VS Sector Z',
                 'Profitability Pctle', 'Price', 'SMA50', 'P/SMA50', 'RSI14', 'Signal']].sort_values('Signal'))
print("\nBreakdown:")
print(results['Signal'].value_counts().to_string())


# =============================================================================
# SECTION 9 — PERFORMANCE COMPARISON & PLOT
# =============================================================================

comparison = pd.DataFrame({
    'Technical Only': stats_tech,
    'Value + Tech':   stats_val,
    'SPY':            spy_stats,
}).T

print(f"\n{'='*70}")
print("PERFORMANCE COMPARISON")
print(f"{'='*70}")
print(comparison.round(4))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(equity_tech.index, equity_tech.values, label='Technical Only',     linewidth=2, color='tab:blue')
ax1.plot(equity_val.index,  equity_val.values,  label='Value + Tech (Tiered)', linewidth=2, color='tab:green')
ax1.plot(spy_eq.index,      spy_eq.values,      label='SPY',                linewidth=2, color='tab:orange', linestyle='--')

for d, s in true_stress.items():
    if s:
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.12, color='red')

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles + [Patch(facecolor='red', alpha=0.3, label='Stress Period')],
           labels  + ['Stress Period'], fontsize=10, loc='upper left')

ax1.set_title('Strategy Performance vs SPY', fontsize=14)
ax1.set_ylabel('Cumulative Return')
ax1.grid(True, alpha=0.3)
ax1.text(0.98, 0.02,
    f"Value+Tech : Sharpe={stats_val['Sharpe']:.2f}  CAGR={stats_val['CAGR']:.1%}  MaxDD={stats_val['MaxDD']:.1%}\n"
    f"Tech Only  : Sharpe={stats_tech['Sharpe']:.2f}  CAGR={stats_tech['CAGR']:.1%}  MaxDD={stats_tech['MaxDD']:.1%}\n"
    f"SPY        : Sharpe={spy_stats['Sharpe']:.2f}  CAGR={spy_stats['CAGR']:.1%}  MaxDD={spy_stats['MaxDD']:.1%}",
    transform=ax1.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.plot(vix_aligned.index, vix_aligned.values, color='purple', linewidth=1.2, label='VIX')
ax2.axhline(25, color='red',    linestyle='--', linewidth=1, label='VIX 25')
ax2.axhline(20, color='orange', linestyle='--', linewidth=1, label='VIX 20')
ax2.fill_between(vix_aligned.index, 25, vix_aligned.values, where=(vix_aligned > 25), alpha=0.2, color='red')
ax2.set_ylabel('VIX')
ax2.set_xlabel('Date')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 10 — TRADE JOURNAL
# A) EVENT LOG  — every state transition per trade lifecycle
# B) POSITION SUMMARY — one row per trade (entry to full exit)
# =============================================================================

def classify_event(prev_state, new_state, ret_pct):
    if new_state > prev_state:
        return {1.0: 'Base Entry', 1.5: 'Pyramid Add 1', 2.0: 'Pyramid Add 2'}.get(new_state, f'Add->{new_state}')
    if new_state == 0.5 and prev_state >= 1.0:
        return 'Trim — Warning (50% exit)'
    if new_state == 0.0 and prev_state > 0:
        if ret_pct > 8:  return 'Full Exit — Profit Taken'
        if ret_pct < -5: return 'Full Exit — Stop Loss'
        return 'Full Exit — Scratch'
    if new_state < prev_state and new_state > 0:
        return f'Step Down ({prev_state:.1f}->{new_state:.1f})'
    return 'Unknown'


def situation_status(state, p2s, ret_pct):
    if state == 0:               return 'Closed'
    if p2s < 0.90:               return 'STOP LOSS IMMINENT'
    if p2s < 0.96:               return 'Warning — Trimmed'
    if p2s > 1.40:               return 'Extended — Near Exit'
    if p2s > 1.30:               return 'Extended — Watch'
    if state >= 2.0 and ret_pct > 20: return 'Pyramid Working'
    if state >= 1.5 and ret_pct > 10: return 'Add Working'
    if ret_pct > 5:              return 'Working'
    if ret_pct < -8:             return 'Under Water'
    return 'Neutral'


def next_action_guidance(state, p2s, entry_px, current_px, regime_ok):
    sma50_est = current_px / p2s if p2s > 0 else current_px

    if state == 0:
        re_px = round(current_px * 0.98, 2)
        return 'Re-entry if recovers', f'Above ${re_px} + entry bar conditions'

    if state == 0.5:
        rec_px  = round(sma50_est * 0.98, 2)
        stop_px = round(sma50_est * 0.90, 2)
        return 'Recover or Stop', f'Rebuild >${rec_px} | Full stop <${stop_px}'

    if state == 1.0:
        trim_px = round(sma50_est * 0.96, 2)
        exit_px = round(sma50_est * 1.35, 2)
        if regime_ok:
            add_lo = round(sma50_est * 1.05, 2)
            add_hi = round(sma50_est * 1.25, 2)
            return 'Add1 / Trim / Exit', f'Add ${add_lo}-${add_hi}+RSI<72+MACD>0 | Trim <${trim_px} | Exit >${exit_px}'
        return 'Hold / Trim / Exit', f'No adds (stress) | Trim <${trim_px} | Exit >${exit_px}'

    if state == 1.5:
        step_px = round(sma50_est * 1.05, 2)
        exit_px = round(sma50_est * 1.40, 2)
        add_lo  = round(sma50_est * 1.12, 2)
        add_hi  = round(sma50_est * 1.28, 2)
        return 'Add2 / Step Down / Exit', f'Add2 ${add_lo}-${add_hi}+MACD accel | Step down <${step_px} | Exit >${exit_px}'

    if state == 2.0:
        step_px = round(sma50_est * 1.12, 2)
        exit_px = round(sma50_est * 1.50, 2)
        stop_px = round(sma50_est * 1.08, 2)
        return 'Hold / Step Down / Exit', f'Step down <${step_px} | Exit >${exit_px} or <${stop_px}'

    return 'Monitor', 'No specific trigger'


def build_event_journal(pos_df, price_df, unit_size=1000):
    event_rows   = []
    summary_rows = []
    running_pnl  = 0.0

    first_entries = {c: pos_df[c].index[pos_df[c] > 0][0]
                     for c in pos_df.columns if (pos_df[c] > 0).any()}
    sorted_cols   = sorted(first_entries, key=lambda c: first_entries[c])

    for col in sorted_cols:
        pos_s       = pos_df[col]
        px_s        = price_df[col]
        transitions = pos_s[pos_s != pos_s.shift(1)].iloc[1:]

        trade_start    = None
        trade_entry_px = None
        trade_events   = []
        prev_state     = 0.0

        for date, new_state in transitions.items():
            px = px_s.get(date, np.nan)
            if pd.isna(px):
                prev_state = new_state
                continue

            ret_pct    = (px / trade_entry_px - 1) * 100 if (trade_entry_px and trade_entry_px > 0) else 0.0
            event_type = classify_event(prev_state, new_state, ret_pct)

            if prev_state == 0 and new_state > 0:
                trade_start    = date
                trade_entry_px = px
                trade_events   = []
                ret_pct        = 0.0

            try:    p2s_val = float(price_to_sma50.loc[date, col])
            except: p2s_val = 1.0
            try:    rsi_val = float(rsi14.loc[date, col])
            except: rsi_val = 50.0
            try:    reg_ok  = bool(can_pyramid.loc[date])
            except: reg_ok  = True

            pnl_before = round(running_pnl, 0)

            if new_state < prev_state and trade_entry_px:
                units_reduced = (prev_state - new_state) * unit_size
                realized_pnl  = round(units_reduced * (px / trade_entry_px - 1), 0)
            else:
                realized_pnl = 0.0

            if new_state == 0 and trade_entry_px:
                total_trade_pnl = round(prev_state * unit_size * (px / trade_entry_px - 1), 0)
                running_pnl += total_trade_pnl
            elif realized_pnl:
                running_pnl += realized_pnl

            pnl_after  = round(running_pnl, 0)
            status_now = situation_status(new_state, p2s_val, ret_pct)
            action, tgt = next_action_guidance(new_state, p2s_val, trade_entry_px or px, px, reg_ok)

            ev = {
                'Ticker':               col,
                'Date':                 date.date(),
                'Event':                event_type,
                'State':                f'{prev_state:.1f}->{new_state:.1f}',
                'Price':                round(px, 2),
                'P/SMA50':              round(p2s_val, 3),
                'RSI':                  round(rsi_val, 1),
                'Position Ret %':       round(ret_pct, 2),
                'Running P&L Before $': pnl_before,
                'Event P&L $':          realized_pnl if new_state <= prev_state else '—',
                'Running P&L After $':  pnl_after,
                'Situation':            status_now,
                'Next Action':          action,
                'Target / Trigger':     tgt,
            }
            event_rows.append(ev)
            trade_events.append(ev)

            if new_state == 0 and trade_start is not None:
                total_ret = (px / trade_entry_px - 1) * 100 if trade_entry_px else 0
                peak_st   = max(float(e['State'].split('->')[1]) for e in trade_events)
                peak_lbl  = {1.0: 'Base', 1.5: 'Add1', 2.0: 'Add2', 0.5: 'Warning'}.get(peak_st, str(peak_st))
                summary_rows.append({
                    'Ticker':         col,
                    'Entry Date':     trade_start.date(),
                    'Exit Date':      date.date(),
                    'Days Held':      (date - trade_start).days,
                    'Entry Px':       round(trade_entry_px, 2),
                    'Exit Px':        round(px, 2),
                    'Total Return %': round(total_ret, 2),
                    'P&L $':          round(total_ret * unit_size / 100, 0),
                    'Peak State':     peak_lbl,
                    'Had Pyramid':    peak_st >= 1.5,
                    'Had Trim':       any('Trim' in e['Event'] or 'Step' in e['Event'] for e in trade_events),
                    'Num Events':     len(trade_events),
                    'Outcome':        'Win' if total_ret > 0 else 'Loss',
                    'Status':         'Closed',
                })
                trade_start = None; trade_entry_px = None; trade_events = []

            prev_state = new_state

        if trade_start is not None:
            last_px   = px_s.iloc[-1]
            last_st   = pos_s.iloc[-1]
            total_ret = (last_px / trade_entry_px - 1) * 100 if trade_entry_px else 0
            peak_st   = max((float(e['State'].split('->')[1]) for e in trade_events), default=last_st)
            peak_lbl  = {1.0: 'Base', 1.5: 'Add1', 2.0: 'Add2', 0.5: 'Warning'}.get(peak_st, str(peak_st))
            try:    p2s_last = float(price_to_sma50.iloc[-1][col])
            except: p2s_last = 1.0
            try:    rsi_last = float(rsi14.iloc[-1][col])
            except: rsi_last = 50.0
            reg_ok_last = bool(can_pyramid.iloc[-1])
            status_now  = situation_status(last_st, p2s_last, total_ret)
            action, tgt = next_action_guidance(last_st, p2s_last, trade_entry_px, last_px, reg_ok_last)
            summary_rows.append({
                'Ticker':           col,
                'Entry Date':       trade_start.date(),
                'Exit Date':        '—',
                'Days Held':        (price_df.index[-1] - trade_start).days,
                'Entry Px':         round(trade_entry_px, 2),
                'Exit Px':          round(last_px, 2),
                'Total Return %':   round(total_ret, 2),
                'P&L $':            round(total_ret * unit_size / 100, 0),
                'Peak State':       peak_lbl,
                'Had Pyramid':      peak_st >= 1.5,
                'Had Trim':         any('Trim' in e['Event'] or 'Step' in e['Event'] for e in trade_events),
                'Num Events':       len(trade_events),
                'Outcome':          'Open',
                'Situation':        status_now,
                'Next Action':      action,
                'Target / Trigger': tgt,
                'Status':           'Open',
            })

    events_df  = pd.DataFrame(event_rows)  if event_rows   else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values('Entry Date').reset_index(drop=True) if summary_rows else pd.DataFrame()
    return events_df, summary_df


events_df, summary_df = build_event_journal(pos_val, price, unit_size=UNIT_SIZE)

if not summary_df.empty:
    closed   = summary_df[summary_df['Status'] == 'Closed']
    open_pos = summary_df[summary_df['Status'] == 'Open']

    print(f"\n{'='*70}")
    print("TRADE JOURNAL")
    print(f"{'='*70}")
    print(f"Total: {len(summary_df)} trades  ({len(closed)} closed  /  {len(open_pos)} open)")

    if not closed.empty:
        wins       = closed[closed['Total Return %'] > 0]
        pyr_trades = closed[closed['Had Pyramid']]
        trim_trades = closed[closed['Had Trim']]
        print(f"Win Rate          : {len(wins)/len(closed):.1%}  ({len(wins)} wins / {len(closed)-len(wins)} losses)")
        print(f"Avg Return        : {closed['Total Return %'].mean():.2f}%  (${closed['P&L $'].mean():,.0f} per trade)")
        print(f"Avg Hold          : {closed['Days Held'].mean():.0f} days")
        print(f"Pyramided trades  : {len(pyr_trades)} ({len(pyr_trades)/max(len(closed),1):.0%})  | Avg when pyramided: {pyr_trades['Total Return %'].mean():.2f}%")
        print(f"Trimmed trades    : {len(trim_trades)} ({len(trim_trades)/max(len(closed),1):.0%})  | Avg when trimmed:   {trim_trades['Total Return %'].mean():.2f}%")
        print(f"Best              : {closed['Total Return %'].max():.2f}%  —  {closed.loc[closed['Total Return %'].idxmax(),'Ticker']}")
        print(f"Worst             : {closed['Total Return %'].min():.2f}%  —  {closed.loc[closed['Total Return %'].idxmin(),'Ticker']}")
        print(f"Total Closed P&L  : ${closed['P&L $'].sum():,.0f}")

    if not open_pos.empty:
        print(f"Open Unrealized   : ${open_pos['P&L $'].sum():,.0f}  across {len(open_pos)} positions")

    if not closed.empty:
        print(f"\n{'─'*70}")
        print("CLOSED TRADE SUMMARY")
        print(f"{'─'*70}")
        display(closed[['Ticker','Entry Date','Exit Date','Days Held',
                         'Entry Px','Exit Px','Total Return %','P&L $',
                         'Peak State','Had Pyramid','Had Trim','Num Events','Outcome']])

    if not open_pos.empty:
        print(f"\n{'─'*70}")
        print(f"OPEN POSITIONS  (mark-to-market  {price.index[-1].date()})")
        print(f"{'─'*70}")
        display(open_pos[['Ticker','Entry Date','Days Held',
                           'Entry Px','Exit Px','Total Return %','P&L $',
                           'Peak State','Situation','Next Action','Target / Trigger']])

    if not events_df.empty:
        print(f"\n{'─'*70}")
        print("FULL EVENT LOG  (every state transition)")
        print(f"{'─'*70}")
        display(events_df[['Ticker','Date','Event','State',
                            'Price','P/SMA50','RSI',
                            'Position Ret %',
                            'Running P&L Before $','Event P&L $','Running P&L After $',
                            'Situation','Next Action','Target / Trigger']])
