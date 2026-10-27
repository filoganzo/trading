"""Generate portfolio manager charts — US liquid universe, vol-tier optimisation, leverage analysis."""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'path/to/repo/hft')
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

from stocks.backtest.simulation.portfolio_sim import run_portfolio_sim
from stocks.backtest.simulation.param_grid import TRADE_VARIANTS

# ── Paths ─────────────────────────────────────────────────────────────────────
tmp_us  = Path('path/to/repo/hft/stocks/data/backtest_results/_tmp_us')
tmp_liq = Path('path/to/repo/hft/stocks/data/backtest_results/_tmp_liq')
res     = Path('path/to/repo/hft/stocks/data/backtest_results')
out     = Path('path/to/output/FiloHFT_Charts_US_VolTier')
out.mkdir(exist_ok=True)

BENCHMARKS = {'DE40','IT40','UK100','US500','F40','SWI20','ES35','STOXX50','NETH25'}
ACCOUNT    = 100_000
RISK_PCT   = 0.01
LEVERAGE   = 20        # IC Markets stock CFDs (typical 1:20 for US majors)
AVG_SL_PCT = 0.018     # median SL distance as fraction of stock price (from backtest)

# Notional per trade at 1% risk: risk_amt / SL%
NOTIONAL_PER_TRADE = ACCOUNT * RISK_PCT / AVG_SL_PCT   # ≈ $55,556

STYLE_BG = '#0d1117'

def dark_ax(ax):
    ax.set_facecolor(STYLE_BG)
    ax.tick_params(colors='white')
    ax.grid(color='#222', lw=0.5)
    for sp in ax.spines.values():
        sp.set_color('#333')

# ── Load bars ─────────────────────────────────────────────────────────────────
bars_pkl = tmp_us / 'all_bars.pkl' if tmp_us.exists() else tmp_liq / 'all_bars.pkl'
print(f"Loading all_bars from {bars_pkl}...", flush=True)
all_bars = pickle.load(open(bars_pkl, 'rb'))

# ── Find best entry config (most candidates in _tmp_us, else _tmp_liq) ────────
cands_dir = tmp_us if tmp_us.exists() else tmp_liq
best_eid, best_n = 34, 0
for p in sorted(cands_dir.glob('cands_*.parquet')):
    try:
        n = len(pd.read_parquet(p))
        if n > best_n:
            best_n  = n
            best_eid = int(p.stem.split('_')[1])
    except Exception:
        pass
print(f"Best entry_id = {best_eid} ({best_n} candidates)", flush=True)

cands = pd.read_parquet(cands_dir / f'cands_{best_eid}.parquet')
print(f"Candidates loaded: {len(cands)} rows", flush=True)

# ── Latest sweep CSV (US liquid preferred) ────────────────────────────────────
csvs_us  = sorted(res.glob('trade_metrics_*_liq_us.csv'))
csvs_liq = sorted(res.glob('trade_metrics_*_liq.csv'))
sweep_csv = (csvs_us[-1] if csvs_us else csvs_liq[-1] if csvs_liq else None)
sweep_df  = pd.read_csv(sweep_csv).drop_duplicates() if sweep_csv else pd.DataFrame()
print(f"Sweep CSV: {sweep_csv.name if sweep_csv else 'none'}, {len(sweep_df)} rows", flush=True)

# ── Variant 6 — global winner ─────────────────────────────────────────────────
variant6 = next(v for v in TRADE_VARIANTS if v['variant_id'] == 6)

# ── Simulations by vol tier ───────────────────────────────────────────────────
tiers = ['low_vol', 'mid_vol', 'high_vol']
TIER_COLORS = {'low_vol': '#4fc3f7', 'mid_vol': '#ffd54f', 'high_vol': '#ef5350', 'ALL': '#81c784'}
TIER_LABELS = {'low_vol': 'Low Vol', 'mid_vol': 'Mid Vol', 'high_vol': 'High Vol'}

has_vol_tier = 'vol_tier' in cands.columns
sim_results = {}

print("Running portfolio sims...", flush=True)
cands_all_sim = cands.copy()
trades_all, equity_all = run_portfolio_sim(cands_all_sim, all_bars, variant6, max_concurrent=5)
sim_results['ALL'] = (trades_all, equity_all)
print(f"  ALL: {len(trades_all)} trades", flush=True)

if has_vol_tier:
    for tier in tiers:
        sub = cands[cands['vol_tier'] == tier].copy()
        if sub.empty:
            continue
        t, e = run_portfolio_sim(sub, all_bars, variant6, max_concurrent=5)
        sim_results[tier] = (t, e)
        print(f"  {tier}: {len(t)} trades", flush=True)


def make_equity_from_trades(trades, account=ACCOUNT, risk_pct=RISK_PCT):
    t = trades.copy()
    t['entry_date'] = pd.to_datetime(t['entry_date'])
    t = t.sort_values('entry_date').reset_index(drop=True)
    t['pnl_usd'] = t['pnl_r'] * (account * risk_pct)
    t['cum_pnl'] = t['pnl_usd'].cumsum()
    t['account'] = account + t['cum_pnl']
    return t


def drawdown_series(acct_series):
    s = pd.Series(acct_series)
    peak = s.expanding().max()
    return (s - peak) / peak * 100


def compute_metrics(trades, account=ACCOUNT, risk_pct=RISK_PCT):
    if trades.empty:
        return {}
    t = make_equity_from_trades(trades, account, risk_pct)
    r = trades['pnl_r']
    wins   = r[r > 0]
    losses = r[r < 0]
    years  = (t['entry_date'].max() - t['entry_date'].min()).days / 365.25
    cagr   = ((account + t['pnl_usd'].sum()) / account) ** (1 / years) - 1 if years > 0 else float('nan')
    dd     = drawdown_series(t['account'])
    return {
        'n':        len(trades),
        'win_rate': (r > 0).mean(),
        'avg_R':    r.mean(),
        'sharpe':   r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else float('nan'),
        'pf':       wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('nan'),
        'max_dd':   dd.min(),
        'cagr':     cagr,
        'calmar':   (cagr * 100) / abs(dd.min()) if dd.min() < 0 else float('nan'),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Equity Curves by Vol Tier
# ═══════════════════════════════════════════════════════════════════════════════
print("Chart 1...", flush=True)
fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor(STYLE_BG)
ax1, ax2 = axes
for ax in axes:
    dark_ax(ax)

plot_items = [('ALL', '#81c784')] + [(t, TIER_COLORS[t]) for t in tiers if t in sim_results]
for key, color in plot_items:
    trades, _ = sim_results[key]
    if trades.empty:
        continue
    t = make_equity_from_trades(trades)
    m = compute_metrics(trades)
    lbl = TIER_LABELS.get(key, 'All US Liquid')
    lw  = 2.5 if key == 'ALL' else 1.5
    ax1.plot(t['entry_date'], t['account'], color=color, lw=lw,
             label=f"{lbl}  (n={m['n']}  |  CAGR {m['cagr']*100:.1f}%  |  Sharpe {m['sharpe']:.2f})")
    dd = drawdown_series(t['account'])
    ax2.fill_between(range(len(dd)), dd, 0, color=color, alpha=0.35)

ax1.axhline(ACCOUNT, color='#555', lw=0.8, ls='--')
ax1.set_ylabel('Account Value ($)', color='white', fontsize=11)
ax1.set_title(
    'LRC Mean-Reversion — US Liquid Universe by Volatility Tier\n'
    'lev+0.3atr|rr5|none  |  $100k  |  1% risk/trade  |  max 5 concurrent',
    color='white', fontsize=13, pad=10)
ax1.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax2.set_ylabel('Drawdown %', color='white', fontsize=10)
ax2.set_ylim(None, 2)
plt.tight_layout()
fig.savefig(out / 'chart1_equity_by_vol_tier.png', dpi=150, bbox_inches='tight', facecolor=STYLE_BG)
plt.close()
print("  saved", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — Trade Distribution by Vol Tier
# ═══════════════════════════════════════════════════════════════════════════════
print("Chart 2...", flush=True)
tier_keys = [k for k in ['low_vol', 'mid_vol', 'high_vol'] if k in sim_results]
ncols = max(len(tier_keys), 1)
fig, axes = plt.subplots(1, ncols, figsize=(14, 6), sharey=False)
fig.patch.set_facecolor(STYLE_BG)
if ncols == 1:
    axes = [axes]

for ax, tier in zip(axes, tier_keys):
    dark_ax(ax)
    trades, _ = sim_results[tier]
    m = compute_metrics(trades)
    r = trades['pnl_r']
    bins = np.linspace(-1.5, 5.5, 30)
    ax.hist(r, bins=bins, color=TIER_COLORS[tier], alpha=0.8, edgecolor='#111')
    ax.axvline(0,       color='white',  lw=1.0, ls='--')
    ax.axvline(r.mean(), color='yellow', lw=1.5, label=f"Mean R = {r.mean():.3f}")
    ax.set_title(
        f"{TIER_LABELS[tier]}  (n={m['n']})\n"
        f"Win {m['win_rate']:.1%}  |  Avg R {m['avg_R']:.3f}  |  Sharpe {m['sharpe']:.2f}",
        color='white', fontsize=11)
    ax.set_xlabel('Trade P&L (R multiples)', color='white')
    ax.set_ylabel('# Trades', color='white')
    ax.legend(facecolor='#1a1f2e', labelcolor='white', fontsize=9)

fig.suptitle('Trade Return Distribution by Volatility Tier — lev+0.3atr|rr5|none', color='white', fontsize=13)
plt.tight_layout()
fig.savefig(out / 'chart2_distribution_by_tier.png', dpi=150, bbox_inches='tight', facecolor=STYLE_BG)
plt.close()
print("  saved", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Monthly Return Heatmap (All US liquid)
# ═══════════════════════════════════════════════════════════════════════════════
print("Chart 3...", flush=True)
trades_hm, _ = sim_results['ALL']
t = make_equity_from_trades(trades_hm)
t['month'] = t['entry_date'].dt.to_period('M')
monthly = t.groupby('month')['pnl_usd'].sum() / ACCOUNT * 100
idx     = pd.period_range(t['entry_date'].min().to_period('M'),
                          t['entry_date'].max().to_period('M'), freq='M')
monthly = monthly.reindex(idx, fill_value=0)
years   = sorted(set(p.year for p in monthly.index))
grid    = pd.DataFrame(index=years, columns=list(range(1, 13)), dtype=float)
for p, v in monthly.items():
    grid.loc[p.year, p.month] = v

vals_flat = grid.values[~np.isnan(grid.values)]
vmax = max(abs(vals_flat.max()), abs(vals_flat.min())) if len(vals_flat) else 1

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(STYLE_BG)
dark_ax(ax)
cmap = LinearSegmentedColormap.from_list('rg', ['#c62828', '#1b1b1b', '#2e7d32'])
im   = ax.imshow(grid.values.astype(float), cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.set_xticks(range(12)); ax.set_xticklabels(mnames, color='white')
ax.set_yticks(range(len(years))); ax.set_yticklabels(years, color='white')
for i, yr in enumerate(years):
    for j in range(12):
        val = grid.loc[yr, j + 1]
        if not (np.isnan(val) or val == 0):
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    color='white' if abs(val) > vmax * 0.4 else 'lightgray',
                    fontsize=9, fontweight='bold')
cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.ax.tick_params(colors='white')
cb.set_label('Monthly Return %', color='white')
ax.set_title('Monthly Returns — US Liquid Universe  (% of $100k, 1% risk/trade, 5 concurrent)',
             color='white', fontsize=12)
plt.tight_layout()
fig.savefig(out / 'chart3_monthly_heatmap.png', dpi=150, bbox_inches='tight', facecolor=STYLE_BG)
plt.close()
print("  saved", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Vol Tier Performance Comparison (metrics grid)
# ═══════════════════════════════════════════════════════════════════════════════
print("Chart 4...", flush=True)
tier_metrics = {k: compute_metrics(sim_results[k][0]) for k in sim_results if sim_results[k][0] is not None}

metric_defs = [
    ('n',        'Total Trades',    '',    False),
    ('win_rate', 'Win Rate',        '%',   False),
    ('avg_R',    'Avg R / Trade',   'R',   False),
    ('sharpe',   'Sharpe R',        '',    False),
    ('pf',       'Profit Factor',   'x',   False),
    ('cagr',     'CAGR',            '%',   False),
    ('max_dd',   'Max Drawdown',    '%',   True),
    ('calmar',   'Calmar Ratio',    '',    False),
]

tier_order = [k for k in ['ALL', 'low_vol', 'mid_vol', 'high_vol'] if k in tier_metrics]
tier_lbls  = [TIER_LABELS.get(k, 'All US') for k in tier_order]
tier_cols  = [TIER_COLORS[k] for k in tier_order]

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
fig.patch.set_facecolor(STYLE_BG)

for ax, (key, name, unit, neg) in zip(axes.flat, metric_defs):
    dark_ax(ax)
    vals = []
    for tk in tier_order:
        m   = tier_metrics[tk]
        raw = m.get(key, float('nan'))
        if key in ('win_rate', 'cagr') and not np.isnan(raw):
            raw *= 100
        vals.append(raw)

    bars = ax.bar(tier_lbls, vals, color=tier_cols, edgecolor='#111', width=0.5)
    for bar, val in zip(bars, vals):
        if np.isnan(val):
            continue
        fmt = f'{int(val)}' if key == 'n' else f'{val:.2f}{unit}'
        ypos = bar.get_height() * (0.92 if neg else 1.05)
        va   = 'top' if neg else 'bottom'
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, fmt,
                ha='center', va=va, color='white', fontsize=10, fontweight='bold')
    ax.set_title(name, color='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=8)
    if neg:
        ax.set_ylim(min(v for v in vals if not np.isnan(v)) * 1.5, 0)
    else:
        valid = [v for v in vals if not np.isnan(v)]
        ax.set_ylim(0, max(valid) * 1.4 if valid else 1)

fig.suptitle('Performance by Volatility Tier — lev+0.3atr|rr5|none  |  US Liquid  |  5 concurrent',
             color='white', fontsize=13)
plt.tight_layout()
fig.savefig(out / 'chart4_vol_tier_metrics.png', dpi=150, bbox_inches='tight', facecolor=STYLE_BG)
plt.close()
print("  saved", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5 — Leverage Analysis: Concurrency × CAGR × Drawdown
# ═══════════════════════════════════════════════════════════════════════════════
print("Chart 5...", flush=True)

# Simulate at different max_concurrent levels
concurrent_levels = [1, 3, 5, 10]
lev_rows = []
for mc in concurrent_levels:
    t_mc, _ = run_portfolio_sim(cands_all_sim, all_bars, variant6, max_concurrent=mc)
    m = compute_metrics(t_mc)
    # Margin at 1:LEVERAGE
    margin_per_slot  = NOTIONAL_PER_TRADE / LEVERAGE
    total_margin     = margin_per_slot * mc
    margin_pct_acct  = total_margin / ACCOUNT * 100
    lev_rows.append({
        'concurrent': mc,
        'trades':     m.get('n', 0),
        'cagr':       m.get('cagr', float('nan')) * 100,
        'sharpe':     m.get('sharpe', float('nan')),
        'max_dd':     m.get('max_dd', float('nan')),
        'calmar':     m.get('calmar', float('nan')),
        'margin_pct': margin_pct_acct,
    })
    print(f"  mc={mc}: CAGR={m.get('cagr',0)*100:.1f}%  dd={m.get('max_dd',0):.2f}%  margin={margin_pct_acct:.1f}%", flush=True)

lev_df = pd.DataFrame(lev_rows)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(STYLE_BG)
xlabels = [f'{r["concurrent"]} slots\n({r["margin_pct"]:.1f}% margin)' for _, r in lev_df.iterrows()]
bar_colors = ['#4fc3f7', '#29b6f6', '#0277bd', '#01579b']

# CAGR
ax = axes[0]
dark_ax(ax)
bars = ax.bar(xlabels, lev_df['cagr'], color=bar_colors[:len(lev_df)], edgecolor='#111', width=0.5)
for bar, val in zip(bars, lev_df['cagr']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
            f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
ax.set_title('CAGR by Concurrent Slots', color='white', fontsize=11)
ax.set_ylabel('CAGR %', color='white')
ax.set_ylim(0, lev_df['cagr'].max() * 1.4)

# Sharpe
ax = axes[1]
dark_ax(ax)
bars = ax.bar(xlabels, lev_df['sharpe'], color=bar_colors[:len(lev_df)], edgecolor='#111', width=0.5)
for bar, val in zip(bars, lev_df['sharpe']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
            f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
ax.set_title('Sharpe R by Concurrent Slots', color='white', fontsize=11)
ax.set_ylabel('Sharpe R', color='white')
ax.set_ylim(0, lev_df['sharpe'].max() * 1.4)

# Max Drawdown
ax = axes[2]
dark_ax(ax)
dd_vals = lev_df['max_dd'].tolist()
bars = ax.bar(xlabels, dd_vals, color=['#ef5350']*len(lev_df), edgecolor='#111', width=0.5)
for bar, val in zip(bars, dd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.92,
            f'{val:.2f}%', ha='center', va='top', color='white', fontsize=11, fontweight='bold')
ax.set_title('Max Drawdown by Concurrent Slots', color='white', fontsize=11)
ax.set_ylabel('Max Drawdown %', color='white')
ax.set_ylim(min(dd_vals) * 1.5, 0)

# Annotation block
margin_note = (
    f"IC Markets 1:{LEVERAGE} leverage on stock CFDs\n"
    f"Notional/trade ≈ ${NOTIONAL_PER_TRADE:,.0f}  |  Margin/trade ≈ ${NOTIONAL_PER_TRADE/LEVERAGE:,.0f}\n"
    f"$100k account  |  1% risk/trade"
)
fig.text(0.5, -0.03, margin_note, ha='center', color='#aaa', fontsize=9)
fig.suptitle('Leverage & Concurrency Impact — lev+0.3atr|rr5|none  |  US Liquid',
             color='white', fontsize=13)
plt.tight_layout()
fig.savefig(out / 'chart5_leverage_concurrency.png', dpi=150, bbox_inches='tight', facecolor=STYLE_BG)
plt.close()
print("  saved", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6 — Best Variant per Vol Tier + Sector (from sweep CSV)
# ═══════════════════════════════════════════════════════════════════════════════
print("Chart 6...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(STYLE_BG)

def _seg_bar(ax, df, seg_type, title, min_trades=15):
    dark_ax(ax)
    sub = df[
        (df['segment_type'] == seg_type) &
        (df['total_trades'] >= min_trades) &
        df['segment_value'].notna() &
        (df['segment_value'].astype(str).str.strip() != '') &
        (df['variant_id'] == 6)
    ].copy()
    if sub.empty:
        ax.set_title(title + ' (no data)', color='white')
        return
    sub = sub.sort_values('sharpe_R', ascending=True)
    colors = ['#ef5350' if v < 4 else '#ffa726' if v < 6 else '#66bb6a' for v in sub['sharpe_R']]
    bars = ax.barh(sub['segment_value'].tolist(), sub['sharpe_R'].tolist(),
                   color=colors, edgecolor='#111')
    for bar, wr, n in zip(bars, sub['win_rate'], sub['total_trades']):
        ax.text(bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.2f}  ({wr:.0%} wr, n={int(n)})',
                va='center', color='white', fontsize=9)
    ax.set_title(title, color='white', fontsize=11)
    ax.set_xlabel('Sharpe R', color='white')
    xlim = sub['sharpe_R'].max() * 1.6 if not sub.empty else 10
    ax.set_xlim(0, xlim)

if not sweep_df.empty:
    _seg_bar(axes[0], sweep_df, 'vol_tier',  'Best Variant by Vol Tier  (variant_id=6)')
    _seg_bar(axes[1], sweep_df, 'sector',    'Best Variant by Sector    (variant_id=6)')
else:
    for ax in axes:
        dark_ax(ax)
        ax.set_title('No sweep CSV found', color='white')

fig.suptitle('Segment Sharpe — lev+0.3atr|rr5|none  |  US Liquid Universe',
             color='white', fontsize=13)
plt.tight_layout()
fig.savefig(out / 'chart6_segments.png', dpi=150, bbox_inches='tight', facecolor=STYLE_BG)
plt.close()
print("  saved", flush=True)

print(f"\nAll charts saved to {out}", flush=True)
