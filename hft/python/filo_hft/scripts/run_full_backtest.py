"""Full-fidelity backtest driver — GOLD ONLY.

The live strategy trades XAUUSD exclusively. XAGUSD and DXY_M6 are
*confirmation feeds only*, consumed by the correlation gate inside the
engine. This driver therefore runs a single `run_from_disk` call on gold
with `load_correlations=True`, then evaluates ~70 TMLab shadow variants on
every base trade and dumps a comprehensive PNG report to the Desktop.

Output bundle (~/Desktop/filohft_backtest_20260415/):
  base backtest parquets      — trades, candidates, equity, metrics
  variants parquet            — long-form (base_trade x variant) table
  SUMMARY.txt                 — full tables + recommended configs
  01..16 PNG plots            — equity, families, sessions, hours, 5-min
                                buckets, TMLab leaderboards, cross-tables
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from filo_hft.backtest.engine import BacktestResult
from filo_hft.backtest.replay import run_from_disk
from filo_hft.backtest.tmlab_variants import (
    default_variants,
    evaluate_variants_for_trades,
)
from filo_hft.config import FiloConfig, load_config


ROOT = Path(__file__).resolve().parents[3]  # FINAL HFT/
CONFIG_PATH = ROOT / "config" / "default.ini"
DATA_ROOT = ROOT / "data"
DESKTOP = Path.home() / "Desktop"
OUT_DIR = DESKTOP / "filohft_backtest_20260415"

WINDOW_FROM = pd.Timestamp("2026-02-01", tz="UTC")
WINDOW_TO = pd.Timestamp("2026-04-13", tz="UTC")

# Live telemetry — used for the reality-check haircut.
LIVE_TELEMETRY = Path(
    "C:/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/telemetry"
)
LIVE_DAYS = ("20260413", "20260414", "20260415")
PIP_PRICE = 0.10
LOT = 0.10
USD_PER_PIP_PER_LOT = 1.0 * 10.0  # $1/pt/lot × 10 points/pip

# Families → which TM bucket the engine uses for their stop/target.
# Mirror of _REVERSION / _CONTINUATION sets in engine.py, plus fakeout →
# reversion bucket in live.
FAMILY_TM_BUCKET: dict[str, str] = {
    "breakout": "breakout",
    "retest": "breakout",
    "engulfing_accel": "breakout",
    "rejection": "rejection",
    "fakeout": "rejection",
    "mean_reversion": "reversion",
    "failed_auction": "reversion",
}


# ======================================================================
# Time-bucket helpers
# ======================================================================


def _classify_session(h: int) -> str:
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "overlap_ln_ny"
    if 16 <= h < 21:
        return "ny"
    return "offhours"


def _augment_time_columns(df: pd.DataFrame, ts_col: str = "ts_open") -> pd.DataFrame:
    if len(df) == 0:
        return df
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)
    out["hour"] = ts.dt.hour.astype(int)
    out["minute"] = ts.dt.minute.astype(int)
    out["bucket_5m"] = ((out["hour"] * 60 + out["minute"]) // 5 * 5).astype(int)
    out["day_of_week"] = ts.dt.dayofweek.astype(int)
    out["session"] = out["hour"].map(_classify_session)
    return out


# ======================================================================
# Formatting helpers
# ======================================================================


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}"


def _pct(num: float, den: float) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


# ======================================================================
# Base backtest
# ======================================================================


def _run_base(cfg: FiloConfig) -> BacktestResult:
    print("[run] loading gold bars + correlation feeds (XAGUSD, DXY_M6)")
    res = run_from_disk(
        data_root=DATA_ROOT,
        cfg=cfg,
        symbol=cfg.symbols.primary,
        from_ts=WINDOW_FROM,
        to_ts=WINDOW_TO,
        load_correlations=True,
    )
    trades = res.trades
    n = len(trades)
    if n:
        gross = float(trades.get("gross_usd", trades["usd"]).sum())
        comm = float(trades.get("commission_usd", pd.Series([0.0])).sum())
        net = float(trades["usd"].sum())
        wr = 100.0 * float((trades["usd"] > 0).sum()) / n
        print(
            f"[run] XAUUSD: {n} trades, gross {_fmt_usd(gross)}, "
            f"commission {_fmt_usd(comm)}, net {_fmt_usd(net)}, WR {wr:.1f}%"
        )
    else:
        print("[run] XAUUSD: zero trades — cannot proceed")
    return res


# ======================================================================
# Variant evaluation (gold only)
# ======================================================================


def _run_variants(cfg: FiloConfig, trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()

    from filo_hft.data.loaders import load_bars
    m1 = load_bars(DATA_ROOT, cfg.symbols.primary, "M1")
    m1 = m1[(m1["ts"] >= WINDOW_FROM) & (m1["ts"] <= WINDOW_TO)].reset_index(drop=True)

    variants = default_variants()
    print(f"[tmlab] evaluating {len(variants)} variants across "
          f"{len(trades)} base trades...")
    vf = evaluate_variants_for_trades(trades, m1, cfg, variants=variants)
    vf = _augment_time_columns(vf, ts_col="ts_open")
    vf["session"] = vf["hour"].map(_classify_session)
    return vf


# ======================================================================
# Aggregation tables
# ======================================================================


def _variant_leaderboard(vf: pd.DataFrame) -> pd.DataFrame:
    g = vf.groupby("variant").agg(
        n=("usd", "size"),
        gross=("gross_usd", "sum"),
        commission=("commission_usd", "sum"),
        net=("usd", "sum"),
        wr=("usd", lambda s: 100.0 * (s > 0).mean()),
        avg_pips=("pips", "mean"),
        avg_bars=("bars_held", "mean"),
    )
    g["expectancy_usd"] = g["net"] / g["n"]
    return g.sort_values("net", ascending=False)


def _variant_by_family(vf: pd.DataFrame) -> pd.DataFrame:
    return (
        vf.groupby(["family", "variant"])["usd"]
        .sum()
        .unstack("variant")
        .fillna(0.0)
    )


def _variant_by_session(vf: pd.DataFrame) -> pd.DataFrame:
    return (
        vf.groupby(["session", "variant"])["usd"]
        .sum()
        .unstack("variant")
        .fillna(0.0)
    )


def _family_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    g = trades.groupby(["family", "direction"]).agg(
        n=("usd", "size"),
        gross=("gross_usd", "sum"),
        commission=("commission_usd", "sum"),
        net=("usd", "sum"),
        wr=("usd", lambda s: 100.0 * (s > 0).mean()),
        avg_pips=("pips", "mean"),
    )
    g["expectancy_usd"] = g["net"] / g["n"]
    return g.sort_values("net", ascending=False)


def _session_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    if "session" not in t.columns:
        t["session"] = t["hour"].map(_classify_session)
    g = t.groupby("session").agg(
        n=("usd", "size"),
        net=("usd", "sum"),
        wr=("usd", lambda s: 100.0 * (s > 0).mean()),
        avg_pips=("pips", "mean"),
    )
    g["expectancy_usd"] = g["net"] / g["n"]
    return g.sort_values("net", ascending=False)


def _hour_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    g = trades.groupby("hour").agg(
        n=("usd", "size"),
        net=("usd", "sum"),
        wr=("usd", lambda s: 100.0 * (s > 0).mean()),
        avg_pips=("pips", "mean"),
    )
    g["expectancy_usd"] = g["net"] / g["n"]
    return g


def _bucket_5m_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    t = _augment_time_columns(trades, ts_col="ts_open")
    g = t.groupby("bucket_5m").agg(
        n=("usd", "size"),
        net=("usd", "sum"),
        wr=("usd", lambda s: 100.0 * (s > 0).mean()),
    )
    g["expectancy_usd"] = g["net"] / g["n"]
    return g


def _family_session_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    if "session" not in t.columns:
        t["session"] = t["hour"].map(_classify_session)
    return (
        t.groupby(["family", "session"])["usd"].sum().unstack("session").fillna(0.0)
    )


# ======================================================================
# Live telemetry reality check
# ======================================================================


def _load_live_trades() -> pd.DataFrame:
    """Reconstruct live trades from telemetry/trades.csv files.

    Dedupes by ticket (first open / last close) to avoid the repeat-open
    bug we observed in 04-15's telemetry where one ticket was logged 290
    times. Commission = LOT × $7/lot round-turn, flat per trade.
    """
    rows: list[pd.DataFrame] = []
    for day in LIVE_DAYS:
        f = LIVE_TELEMETRY / day / "trades.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f, on_bad_lines="skip", engine="python")
        except Exception as exc:
            print(f"[live] failed to read {f}: {exc}")
            continue
        opens = (
            df[df["event"] == "open"]
            .sort_values("ts_ms")
            .drop_duplicates("ticket", keep="first")[
                ["ticket", "direction", "price", "ts_ms"]
            ]
            .rename(columns={"price": "entry", "ts_ms": "ts_open_ms"})
        )
        closes = (
            df[df["event"] == "close"]
            .sort_values("ts_ms")
            .drop_duplicates("ticket", keep="last")[["ticket", "price", "ts_ms"]]
            .rename(columns={"price": "exit", "ts_ms": "ts_close_ms"})
        )
        t = opens.merge(closes, on="ticket", how="inner")
        if len(t) == 0:
            continue
        t["day"] = day
        t["ts_open"] = pd.to_datetime(t["ts_open_ms"], unit="ms", utc=True)
        t["ts_close"] = pd.to_datetime(t["ts_close_ms"], unit="ms", utc=True)
        t["pip_diff"] = (t["exit"] - t["entry"]) / PIP_PRICE
        t.loc[t["direction"] == "short", "pip_diff"] *= -1
        t["gross_usd"] = t["pip_diff"] * LOT * USD_PER_PIP_PER_LOT
        t["commission_usd"] = LOT * 7.0
        t["net_usd"] = t["gross_usd"] - t["commission_usd"]
        rows.append(t)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _live_summary(live: pd.DataFrame) -> dict:
    if len(live) == 0:
        return {}
    out: dict = {}
    for day, grp in live.groupby("day"):
        n = len(grp)
        net = float(grp["net_usd"].sum())
        wr = 100.0 * float((grp["net_usd"] > 0).mean())
        avgp = float(grp["pip_diff"].mean())
        out[str(day)] = {
            "trades": int(n),
            "net_usd": round(net, 2),
            "wr_pct": round(wr, 2),
            "avg_pips": round(avgp, 2),
        }
    out["_TOTAL_"] = {
        "trades": int(len(live)),
        "net_usd": round(float(live["net_usd"].sum()), 2),
        "wr_pct": round(100.0 * float((live["net_usd"] > 0).mean()), 2),
        "avg_pips": round(float(live["pip_diff"].mean()), 2),
    }
    return out


def _compute_haircut(live: pd.DataFrame, bt_trades: pd.DataFrame) -> dict:
    """Derive the per-trade pip haircut to apply to backtest forecasts so
    that the backtest base-config expectancy matches observed live.

    Result: {"pips_per_trade": float, "bt_expectancy_pips": float,
             "live_expectancy_pips": float, "haircut_pips": float}
    """
    if len(live) == 0 or len(bt_trades) == 0:
        return {}
    bt_exp = float(bt_trades["pips"].mean())
    live_exp = float(live["pip_diff"].mean())
    haircut = bt_exp - live_exp  # pips per trade, positive = optimism
    return {
        "bt_expectancy_pips": round(bt_exp, 3),
        "live_expectancy_pips": round(live_exp, 3),
        "haircut_pips": round(haircut, 3),
        "bt_wr_pct": round(100.0 * float((bt_trades["usd"] > 0).mean()), 2),
        "live_wr_pct": round(100.0 * float((live["net_usd"] > 0).mean()), 2),
    }


def _apply_haircut_to_leaderboard(
    vf: pd.DataFrame, haircut_pips: float
) -> pd.DataFrame:
    """Subtract haircut_pips from every trade's pip, recompute USD and
    rebuild the leaderboard. This gives a 'realistic' forecast conditioned
    on the most recent 3 days of live performance."""
    if len(vf) == 0 or haircut_pips == 0:
        return _variant_leaderboard(vf) if len(vf) else pd.DataFrame()
    adj = vf.copy()
    adj["pips_adj"] = adj["pips"] - haircut_pips
    adj["gross_usd_adj"] = adj["pips_adj"] * LOT * USD_PER_PIP_PER_LOT
    adj["usd_adj"] = adj["gross_usd_adj"] - adj["commission_usd"]
    g = adj.groupby("variant").agg(
        n=("usd_adj", "size"),
        net_haircut=("usd_adj", "sum"),
        wr_haircut=("usd_adj", lambda s: 100.0 * (s > 0).mean()),
        avg_pips_haircut=("pips_adj", "mean"),
    )
    g["expectancy_haircut"] = g["net_haircut"] / g["n"]
    return g.sort_values("net_haircut", ascending=False)


# ======================================================================
# Recommended-config extraction
# ======================================================================


def _variant_params(name: str) -> dict:
    """Resolve a TMLab variant name to its concrete parameters."""
    from filo_hft.backtest.tmlab_variants import default_variants
    for v in default_variants():
        if v.name == name:
            return {
                "stop_pips": v.stop_pips,
                "target_pips": v.target_pips,
                "be_trigger_pips": v.be_trigger_pips,
                "be_buffer_pips": v.be_buffer_pips,
                "trail_start_pips": v.trail_start_pips,
                "trail_distance_pips": v.trail_distance_pips,
                "atr_stop_mult": v.atr_stop_mult,
                "atr_window": v.atr_window,
                "adverse_cut_pips": v.adverse_cut_pips,
                "max_bars_hold": v.max_bars_hold,
            }
    return {}


def _render_per_family_config_block(
    best_per_family: dict[str, dict],
) -> str:
    """Render a per-family config INI block using the engine's 3 TM
    buckets (breakout / reversion / rejection). Continuation families
    (breakout, retest, engulfing_accel) → breakout bucket. Reversal
    families → reversion or rejection bucket per FAMILY_TM_BUCKET.
    If two families collide on one bucket with different winners, we
    pick the one with the higher net USD.
    """
    if not best_per_family:
        return ""
    bucket_winners: dict[str, tuple[str, str, float, dict]] = {}
    for fam, info in best_per_family.items():
        bucket = FAMILY_TM_BUCKET.get(fam)
        if bucket is None:
            continue
        name = info.get("variant", "base")
        net = float(info.get("net_usd", 0.0))
        params = _variant_params(name)
        if not params:
            continue
        if bucket not in bucket_winners or net > bucket_winners[bucket][2]:
            bucket_winners[bucket] = (fam, name, net, params)

    lines = ["[trade_management]",
             "; Per-family winners from TMLab (74 variants × 7604 base trades)",
             "; Families map to TM buckets: breakout/retest/engulfing_accel→breakout,",
             "; mean_reversion/failed_auction→reversion, rejection/fakeout→rejection"]
    for bucket, (fam, name, net, params) in sorted(bucket_winners.items()):
        lines.append("")
        lines.append(f"; === {bucket} bucket (driven by {fam}, variant '{name}', net ${net:,.0f}) ===")
        if params["atr_stop_mult"] > 0:
            lines.append(f"; NOTE: variant '{name}' uses atr_stop_mult={params['atr_stop_mult']}")
            lines.append(f"; MQL5 does not yet implement ATR-scaled stops — fall back to a")
            lines.append(f"; wide fixed stop of ~{max(25, params['stop_pips']):.0f} pips until the ATR port ships.")
            eff_stop = max(25.0, params['stop_pips'])
        else:
            eff_stop = params['stop_pips']
        lines.append(f"stop_{bucket}_pips    = {eff_stop:.1f}")
        lines.append(f"target_{bucket}_pips  = {params['target_pips']:.1f}")
    lines.append("")
    lines.append("; Global BE / trail — same across buckets (engine does not separate them)")
    # pick the most-common BE/trail across the winners
    be_vals = [p[3]["be_trigger_pips"] for p in bucket_winners.values()]
    ts_vals = [p[3]["trail_start_pips"] for p in bucket_winners.values()]
    td_vals = [p[3]["trail_distance_pips"] for p in bucket_winners.values()]
    if be_vals:
        lines.append(f"be_trigger_pips    = {float(np.median(be_vals)):.1f}")
    if ts_vals:
        lines.append(f"trail_start_pips   = {float(np.median(ts_vals)):.1f}")
    if td_vals:
        lines.append(f"trail_distance_pips = {float(np.median(td_vals)):.1f}")
    return "\n".join(lines)


def _recommend_configs(vf: pd.DataFrame, trades: pd.DataFrame) -> dict:
    rec: dict = {}
    if len(vf) == 0:
        return rec

    lb = _variant_leaderboard(vf)
    rec["best_variant_overall"] = {
        "name": str(lb.index[0]),
        "net_usd": float(lb.iloc[0]["net"]),
        "vs_base_usd": float(lb.iloc[0]["net"] - lb.loc["base", "net"])
        if "base" in lb.index else None,
        "wr": float(lb.iloc[0]["wr"]),
        "n": int(lb.iloc[0]["n"]),
    }

    # Best variant per family
    by_fam: dict[str, dict] = {}
    for fam, sub in vf.groupby("family"):
        g = sub.groupby("variant")["usd"].sum()
        if len(g):
            best = g.idxmax()
            by_fam[fam] = {
                "variant": str(best),
                "net_usd": float(g.loc[best]),
                "n_trades_in_family": int(len(sub[sub["variant"] == best])),
            }
    rec["best_variant_per_family"] = by_fam

    # Best variant per session
    by_sess: dict[str, dict] = {}
    for sess, sub in vf.groupby("session"):
        g = sub.groupby("variant")["usd"].sum()
        if len(g):
            best = g.idxmax()
            by_sess[str(sess)] = {
                "variant": str(best),
                "net_usd": float(g.loc[best]),
            }
    rec["best_variant_per_session"] = by_sess

    # Family whitelist — only families with positive net AND expectancy
    fam_br = _family_breakdown(trades)
    keep = []
    drop = []
    if len(fam_br):
        fam_totals = fam_br.groupby(level="family").agg(
            n=("n", "sum"), net=("net", "sum")
        )
        for fam, row in fam_totals.iterrows():
            if row["net"] > 0 and (row["net"] / max(row["n"], 1)) > 0:
                keep.append(str(fam))
            else:
                drop.append(str(fam))
    rec["family_whitelist"] = {"enable": keep, "disable": drop}

    # Hour-of-day gate — hours with positive net expectancy
    hr = _hour_breakdown(trades)
    if len(hr):
        profitable_hours = sorted(hr[hr["net"] > 0].index.tolist())
        loss_hours = sorted(hr[hr["net"] <= 0].index.tolist())
        rec["hour_whitelist"] = {
            "trade": profitable_hours,
            "skip": loss_hours,
        }

    # Session gate
    ses = _session_breakdown(trades)
    if len(ses):
        profitable_sessions = ses[ses["net"] > 0].index.tolist()
        rec["session_whitelist"] = list(profitable_sessions)

    # Best 5-min buckets (top 20)
    b5 = _bucket_5m_breakdown(trades)
    if len(b5):
        top = b5.sort_values("net", ascending=False).head(20)
        rec["top_5m_buckets"] = [
            {"bucket_5m": int(idx), "net_usd": float(row["net"]), "n": int(row["n"])}
            for idx, row in top.iterrows()
        ]

    return rec


# ======================================================================
# Plots
# ======================================================================


def _save_fig(fig, path: Path, dpi: int = 130) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path.name}")


def _plot_equity(eq: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(eq) == 0:
        ax.text(0.5, 0.5, "no trades", ha="center", va="center")
    else:
        ax.plot(eq["ts"], eq["equity_usd"], color="#0b6", linewidth=1.5)
        ax.axhline(0.0, color="#888", linewidth=0.5)
        ax.set_ylabel("Net equity (USD)")
        ax.grid(alpha=0.3)
    ax.set_title("XAUUSD — net equity curve (commissions applied)")
    _save_fig(fig, out)


def _plot_drawdown(eq: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    if "drawdown_pct" in eq.columns and len(eq):
        ax.fill_between(eq["ts"], -eq["drawdown_pct"], 0, color="#c33", alpha=0.5)
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)
    ax.set_title("XAUUSD — drawdown curve")
    _save_fig(fig, out)


def _plot_family_pnl(trades: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(trades):
        g = trades.groupby(["family", "direction"])["usd"].sum().unstack("direction").fillna(0.0)
        g.plot(kind="bar", ax=ax, width=0.75, color=["#0b6", "#c33"])
        ax.axhline(0.0, color="#888", linewidth=0.5)
        ax.set_ylabel("Net USD")
        ax.legend(title="direction")
    ax.set_title("Per family × direction — net USD (after commissions)")
    _save_fig(fig, out)


def _plot_session(trades: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(trades):
        g = _session_breakdown(trades)
        colors = ["#0b6" if v > 0 else "#c33" for v in g["net"]]
        ax1.bar(g.index, g["net"], color=colors)
        ax1.set_ylabel("Net USD")
        ax1.set_title("Per-session net USD")
        ax1.axhline(0.0, color="#555", linewidth=0.5)
        ax2.bar(g.index, g["n"], color="#46c")
        ax2.set_ylabel("Trade count")
        ax2.set_title("Per-session trade count")
        for a in (ax1, ax2):
            a.tick_params(axis="x", rotation=30)
    _save_fig(fig, out)


def _plot_hour(trades: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    if len(trades):
        g = _hour_breakdown(trades).reindex(range(24)).fillna(0.0)
        colors = ["#0b6" if v > 0 else "#c33" for v in g["net"]]
        ax1.bar(g.index, g["net"], color=colors)
        ax1.set_ylabel("Net USD")
        ax1.set_title("XAUUSD — per-hour net PnL (UTC)")
        ax1.axhline(0.0, color="#555", linewidth=0.5)
        ax1.set_xticks(range(24))
        ax2.bar(g.index, g["n"], color="#46c")
        ax2.set_ylabel("Trade count")
        ax2.set_title("XAUUSD — per-hour trade count")
        ax2.set_xlabel("UTC hour")
        ax2.set_xticks(range(24))
    _save_fig(fig, out)


def _plot_bucket_5m(trades: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    if len(trades):
        t = _augment_time_columns(trades, ts_col="ts_open")
        g = t.groupby("bucket_5m")["usd"].agg(["sum", "size"]).reindex(
            range(0, 24 * 60, 5)
        ).fillna(0.0)
        colors = ["#0b6" if v > 0 else "#c33" for v in g["sum"]]
        ax1.bar(g.index, g["sum"], width=4, color=colors)
        ax1.set_ylabel("Net USD")
        ax1.set_title(
            "XAUUSD — 5-minute slot net PnL (minute-of-day UTC, 288 buckets)"
        )
        ax1.axhline(0.0, color="#555", linewidth=0.5)
        ax1.set_xlim(0, 24 * 60)
        ax1.set_xticks(range(0, 24 * 60 + 1, 60))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(25)])
        ax2.bar(g.index, g["size"], width=4, color="#46c")
        ax2.set_ylabel("Trade count")
        ax2.set_xlabel("UTC time")
        ax2.set_xlim(0, 24 * 60)
        ax2.set_xticks(range(0, 24 * 60 + 1, 60))
        ax2.set_xticklabels([f"{h:02d}:00" for h in range(25)])
    _save_fig(fig, out)


def _plot_hour_family_heatmap(trades: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    if len(trades):
        p = (
            trades.groupby(["family", "hour"])["usd"]
            .sum()
            .unstack("hour")
            .reindex(columns=range(24))
            .fillna(0.0)
        )
        vmax = max(abs(p.values.min()), abs(p.values.max()), 1.0)
        im = ax.imshow(
            p.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax
        )
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        ax.set_yticks(range(len(p.index)))
        ax.set_yticklabels(p.index)
        ax.set_xlabel("UTC hour")
        ax.set_title("Family × hour net PnL (USD)")
        fig.colorbar(im, ax=ax, label="Net USD")
    _save_fig(fig, out)


def _plot_block_reasons(candidates: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(candidates):
        blocked = candidates[
            candidates["block_reason"].notna() & (candidates["block_reason"] != "")
        ]
        # normalize "score X < Y" variants into one bucket
        norm = blocked["block_reason"].str.replace(
            r"score \d+\.\d+", "score <0.45", regex=True
        ).str.replace(
            r"spread \d+\.\d+", "spread > cap", regex=True
        ).str.replace(
            r"kl eff-weight \d+\.\d+", "kl sliced", regex=True
        ).str.replace(
            r"exhaustion \d+ bars / [\d\.]+ pips", "exhaustion", regex=True
        )
        top = norm.value_counts().head(12)
        ax.barh(top.index[::-1], top.values[::-1], color="#c55")
        ax.set_xlabel("Candidates blocked")
        ax.set_title("Top block reasons (gold) — candidates that did not fire")
    _save_fig(fig, out)


def _plot_variant_leaderboard(vf: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    if len(vf):
        lb = _variant_leaderboard(vf)
        colors = ["#0b6" if v > 0 else "#c33" for v in lb["net"]]
        ax.barh(lb.index[::-1], lb["net"][::-1], color=colors[::-1])
        ax.set_xlabel("Net USD")
        ax.set_title("TMLab variants — net USD across all gold trades")
        ax.axvline(0.0, color="#555", linewidth=0.5)
        if "base" in lb.index:
            base_val = lb.loc["base", "net"]
            ax.axvline(
                base_val, color="#06c", linewidth=1.2, linestyle="--",
                label=f"base = {_fmt_usd(base_val)}",
            )
            ax.legend(loc="lower right")
    _save_fig(fig, out)


def _plot_variant_by_family(vf: pd.DataFrame, out: Path) -> None:
    if len(vf) == 0:
        fig, ax = plt.subplots(); _save_fig(fig, out); return
    families = sorted(vf["family"].unique())
    n = len(families)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)
    for ax, fam in zip(axes, families):
        sub = vf[vf["family"] == fam]
        g = sub.groupby("variant")["usd"].sum().sort_values(ascending=False)
        top = g.head(15)
        colors = ["#0b6" if v > 0 else "#c33" for v in top.values]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        ax.set_title(f"{fam} — top 15 variants (net USD)")
        ax.axvline(0.0, color="#555", linewidth=0.5)
    for ax in axes[len(families):]:
        ax.axis("off")
    _save_fig(fig, out)


def _plot_variant_by_session(vf: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    if len(vf):
        tbl = (
            vf.groupby(["session", "variant"])["usd"]
            .sum()
            .unstack("session")
            .fillna(0.0)
        )
        top20 = tbl.sum(axis=1).sort_values(ascending=False).head(20).index
        tbl = tbl.loc[top20]
        tbl.plot(kind="barh", stacked=True, ax=ax, colormap="tab10")
        ax.set_xlabel("Net USD")
        ax.set_title("Top 20 variants — stacked by session")
        ax.axvline(0.0, color="#555", linewidth=0.5)
    _save_fig(fig, out)


def _plot_close_reasons(trades: pd.DataFrame, vf: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if len(trades) and "close_reason" in trades.columns:
        cr = trades["close_reason"].value_counts()
        ax1.bar(cr.index, cr.values, color="#46c")
        ax1.set_title("Base trades — close reasons")
        ax1.tick_params(axis="x", rotation=30)
    if len(vf):
        cr2 = vf["close_reason"].value_counts()
        ax2.bar(cr2.index, cr2.values, color="#a63")
        ax2.set_title("All variant exits — close reasons")
        ax2.tick_params(axis="x", rotation=30)
    _save_fig(fig, out)


def _plot_commission_impact(trades: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(trades):
        gross = float(trades.get("gross_usd", trades["usd"]).sum())
        comm = float(trades.get("commission_usd", pd.Series([0.0])).sum())
        net = float(trades["usd"].sum())
        vals = [gross, comm, net]
        labels = ["Gross", "Commission", "Net"]
        colors = ["#06c", "#c33", "#0b6"]
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v,
                _fmt_usd(v), ha="center", va="bottom" if v >= 0 else "top",
            )
        ax.axhline(0.0, color="#555", linewidth=0.5)
        ax.set_ylabel("USD")
        ax.set_title("Gross → commission → net (gold, full window)")
    _save_fig(fig, out)


def _plot_signal_funnel(candidates: pd.DataFrame, trades: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(candidates):
        total_cands = len(candidates)
        accepted = int(candidates["accepted"].sum()) if "accepted" in candidates.columns else 0
        filled = len(trades)
        labels = ["Candidates", "Accepted", "Filled"]
        vals = [total_cands, accepted, filled]
        bars = ax.bar(labels, vals, color=["#46c", "#ec6", "#0b6"])
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom",
            )
        ax.set_ylabel("Count")
        ax.set_title("Signal funnel — gold")
    _save_fig(fig, out)


def _plot_family_session_stacked(trades: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(trades):
        p = _family_session_breakdown(trades)
        p.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("Net USD")
        ax.set_title("Family × session — stacked net USD")
        ax.axhline(0.0, color="#555", linewidth=0.5)
    _save_fig(fig, out)


def _plot_reality_check(live: pd.DataFrame, trades: pd.DataFrame,
                        haircut: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if len(live) and "day" in live.columns:
        per_day = (
            live.groupby("day")
            .agg(n=("net_usd", "size"),
                 net=("net_usd", "sum"),
                 pips=("pip_diff", "mean"))
            .reset_index()
        )
        colors = ["#0b6" if v > 0 else "#c33" for v in per_day["net"]]
        ax1.bar(per_day["day"], per_day["net"], color=colors)
        for i, r in per_day.iterrows():
            ax1.text(i, r["net"], f"{r['n']} tr\n{_fmt_usd(r['net'])}",
                     ha="center", va="bottom" if r["net"] >= 0 else "top",
                     fontsize=9)
        ax1.set_ylabel("Net USD")
        ax1.set_title("LIVE telemetry — per-day net PnL (Apr 13 PM → Apr 15)")
        ax1.axhline(0.0, color="#555", linewidth=0.5)
        ax1.tick_params(axis="x", rotation=30)

    if haircut:
        labels = [
            f"backtest\n{haircut['bt_expectancy_pips']:+.2f} pips\nWR {haircut['bt_wr_pct']:.1f}%",
            f"live\n{haircut['live_expectancy_pips']:+.2f} pips\nWR {haircut['live_wr_pct']:.1f}%",
            f"GAP\n{haircut['haircut_pips']:+.2f} pips",
        ]
        vals = [
            haircut["bt_expectancy_pips"],
            haircut["live_expectancy_pips"],
            haircut["haircut_pips"],
        ]
        colors2 = ["#0b6", "#c33", "#fa0"]
        ax2.bar(labels, vals, color=colors2)
        ax2.axhline(0.0, color="#555", linewidth=0.5)
        ax2.set_ylabel("Avg pips / trade")
        ax2.set_title("Backtest vs live expectancy (haircut = optimism)")
    _save_fig(fig, out)


def _plot_haircut_leaderboard(lb_hair: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    if len(lb_hair):
        lb = lb_hair.sort_values("net_haircut", ascending=True).tail(30)
        colors = ["#0b6" if v > 0 else "#c33" for v in lb["net_haircut"]]
        ax.barh(lb.index, lb["net_haircut"], color=colors)
        ax.set_xlabel("Net USD (after live-calibrated haircut)")
        ax.set_title(
            "HAIRCUT-ADJUSTED leaderboard — realistic forecasts, top 30"
        )
        ax.axvline(0.0, color="#555", linewidth=0.5)
    _save_fig(fig, out)


def _plot_top_variant_5m(vf: pd.DataFrame, base_name: str, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    if len(vf):
        sub = vf[vf["variant"] == base_name]
        g = (
            sub.groupby("bucket_5m")["usd"].agg(["sum", "size"])
            .reindex(range(0, 24 * 60, 5))
            .fillna(0.0)
        )
        colors = ["#0b6" if v > 0 else "#c33" for v in g["sum"]]
        ax1.bar(g.index, g["sum"], width=4, color=colors)
        ax1.set_title(f"Variant '{base_name}' — net USD per 5-min slot")
        ax1.axhline(0.0, color="#555", linewidth=0.5)
        ax1.set_xticks(range(0, 24 * 60 + 1, 60))
        ax1.set_xticklabels([f"{h:02d}" for h in range(25)])
        ax2.bar(g.index, g["size"], width=4, color="#46c")
        ax2.set_xticks(range(0, 24 * 60 + 1, 60))
        ax2.set_xticklabels([f"{h:02d}" for h in range(25)])
        ax2.set_xlabel("UTC hour (5-min buckets inside)")
        ax2.set_title(f"Variant '{base_name}' — trade count per 5-min slot")
    _save_fig(fig, out)


# ======================================================================
# SUMMARY writer
# ======================================================================


def _write_summary(
    out_dir: Path,
    trades: pd.DataFrame,
    candidates: pd.DataFrame,
    vf: pd.DataFrame,
    recommended: dict,
    live: pd.DataFrame,
    haircut: dict,
    leaderboard_haircut: pd.DataFrame,
) -> None:
    lines: list[str] = []
    w = lines.append

    # Actual data span (the filename lies — data ends 2026-04-10).
    actual_end = pd.to_datetime(trades["ts_close"]).max() if len(trades) else WINDOW_TO
    w("=" * 78)
    w("FiloHFT — full-fidelity gold backtest")
    w(f"Backtest window: {WINDOW_FROM.date()} → {actual_end.date()}  "
      f"(data ends {actual_end.date()}, filename implies {WINDOW_TO.date()})")
    w(f"Config:  {CONFIG_PATH}")
    w("Primary tradeable symbol: XAUUSD")
    w("Confirmation feeds: XAGUSD, DXY_M6 (correlation only, NOT traded)")
    w("Commission: $7 / lot round-turn, applied per closed trade")
    w("Fidelity: M1 bars, worst-first intrabar walk, slicing penalty active")
    w("=" * 78)
    w("")

    # -----------------------------------------------------------------
    # REALITY CHECK — backtest vs live telemetry for the 3 most recent days
    # -----------------------------------------------------------------
    if len(live):
        w("")
        w("#" * 78)
        w("# REALITY CHECK — backtest vs LIVE telemetry (04-13..04-15)")
        w("#" * 78)
        live_sum = _live_summary(live)
        w("LIVE per day:")
        for day in LIVE_DAYS:
            if day in live_sum:
                r = live_sum[day]
                w(f"  {day}: {r['trades']:4d} trades  "
                  f"net {_fmt_usd(r['net_usd'])}  WR {r['wr_pct']:.1f}%  "
                  f"avg_pips {r['avg_pips']:+.2f}")
        if "_TOTAL_" in live_sum:
            r = live_sum["_TOTAL_"]
            w(f"  TOTAL    : {r['trades']:4d} trades  "
              f"net {_fmt_usd(r['net_usd'])}  WR {r['wr_pct']:.1f}%  "
              f"avg_pips {r['avg_pips']:+.2f}")
        w("")
        if haircut:
            w("Backtest vs live (base config, pip expectancy):")
            w(f"  backtest expectancy : {haircut['bt_expectancy_pips']:+.2f} pips/trade, "
              f"WR {haircut['bt_wr_pct']:.1f}%")
            w(f"  live     expectancy : {haircut['live_expectancy_pips']:+.2f} pips/trade, "
              f"WR {haircut['live_wr_pct']:.1f}%")
            w(f"  ==> HAIRCUT          : {haircut['haircut_pips']:+.2f} pips/trade "
              "(backtest optimism)")
            w("")
            w("NOTE: backtest window (Feb 01..Apr 10) does NOT overlap with live")
            w("telemetry (Apr 13 PM..Apr 15). Haircut is derived purely from")
            w("out-of-sample expectancy gap and applied uniformly to every")
            w("variant in the leaderboard below.")
            w("")
            w("Likely sources of divergence (in descending likelihood):")
            w("  1. Bar-close-only signal detection — backtest takes a")
            w("     selection-biased subset of signals that fire at M1 bar")
            w("     close. Live fires mid-bar and takes more (and worse)")
            w("     signals. Live frequency is ~280/day vs ~150/day backtest.")
            w("  2. Unmodeled slippage and spread variance at entry — backtest")
            w("     uses the M1 bar's reported spread; real ticks can be wider.")
            w("  3. Regime change — backtest days (Feb..Apr 10) had structurally")
            w("     different volatility/trend conditions than 04-13..04-15.")
            w("  4. Correlation confirmation latency — live sees noisier")
            w("     real-time silver/DXY streams than the backtest's bar-close")
            w("     snapshots.")
        w("")

    # -----------------------------------------------------------------
    # Backtest raw headline
    # -----------------------------------------------------------------

    if len(trades):
        gross = float(trades["gross_usd"].sum())
        comm = float(trades["commission_usd"].sum())
        net = float(trades["usd"].sum())
        wr = 100.0 * float((trades["usd"] > 0).mean())
        avg_pips = float(trades["pips"].mean())
        n = len(trades)
        w("HEADLINE (base config, gold)")
        w(f"  trades       : {n}")
        w(f"  gross        : {_fmt_usd(gross)}")
        w(f"  commission   : {_fmt_usd(comm)}")
        w(f"  net          : {_fmt_usd(net)}")
        w(f"  win rate     : {wr:.2f}%")
        w(f"  expectancy   : {avg_pips:+.2f} pips / trade")
        w(f"  comm drag    : {_pct(-comm, gross)} of gross")
        w("")

    # Funnel
    if len(candidates):
        total = len(candidates)
        acc = int(candidates["accepted"].sum()) if "accepted" in candidates.columns else 0
        filled = len(trades)
        w("SIGNAL FUNNEL")
        w(f"  candidates   : {total:,}")
        w(f"  accepted     : {acc:,} ({_pct(acc, total)})")
        w(f"  filled       : {filled:,} ({_pct(filled, total)})")
        w("")

    # Family × direction
    fb = _family_breakdown(trades)
    if len(fb):
        w("PER FAMILY × DIRECTION  (sorted by net)")
        w(fb.round(2).to_string())
        w("")
        # Totals per family
        fam_only = fb.groupby(level="family").agg(
            n=("n", "sum"), net=("net", "sum"), gross=("gross", "sum"),
            commission=("commission", "sum"),
        )
        fam_only["expectancy_usd"] = fam_only["net"] / fam_only["n"]
        fam_only = fam_only.sort_values("net", ascending=False)
        w("PER FAMILY (collapsed)")
        w(fam_only.round(2).to_string())
        w("")

    # Session
    sb = _session_breakdown(trades)
    if len(sb):
        w("PER SESSION")
        w(sb.round(2).to_string())
        w("")

    # Hour
    hb = _hour_breakdown(trades)
    if len(hb):
        w("PER HOUR (UTC)")
        w(hb.round(2).to_string())
        w("")

    # 5-min slot — top 30 and bottom 10 to keep it readable
    b5 = _bucket_5m_breakdown(trades)
    if len(b5):
        b5 = b5.sort_values("net", ascending=False)
        w("TOP 30 5-MIN BUCKETS BY NET USD")
        w(b5.head(30).round(2).to_string())
        w("")
        w("BOTTOM 10 5-MIN BUCKETS BY NET USD")
        w(b5.tail(10).round(2).to_string())
        w("")

    # Variant leaderboard
    if len(vf):
        lb = _variant_leaderboard(vf)
        w(f"TMLab LEADERBOARD — {len(lb)} variants total")
        w(lb.round(2).to_string())
        w("")

        # Variant by family — show winners
        vbf = _variant_by_family(vf)
        if len(vbf):
            w("BEST-3 VARIANTS PER FAMILY")
            for fam in vbf.index:
                row = vbf.loc[fam].sort_values(ascending=False).head(3)
                w(f"  {fam:20s}")
                for name, val in row.items():
                    w(f"    {name:22s}  {_fmt_usd(val)}")
            w("")

        # Variant by session — winners
        vbs = _variant_by_session(vf)
        if len(vbs):
            w("BEST-3 VARIANTS PER SESSION")
            for sess in vbs.index:
                row = vbs.loc[sess].sort_values(ascending=False).head(3)
                w(f"  {sess}")
                for name, val in row.items():
                    w(f"    {name:22s}  {_fmt_usd(val)}")
            w("")

    # Haircut-adjusted variant leaderboard
    if len(leaderboard_haircut):
        w("")
        w("#" * 78)
        w("# HAIRCUT-ADJUSTED TMLab LEADERBOARD")
        w(f"# (haircut = {haircut.get('haircut_pips','?')} pips/trade derived from live)")
        w("#" * 78)
        w(leaderboard_haircut.round(2).to_string())
        w("")
        only_positive = leaderboard_haircut[leaderboard_haircut["net_haircut"] > 0]
        if len(only_positive):
            w(f"Variants still profitable after live-calibrated haircut: {len(only_positive)}")
            w(only_positive.round(2).to_string())
        else:
            w("NO variant is profitable after the haircut.")
        w("")

    # Per-family recommended config block — THE MAIN DELIVERABLE
    best_per_fam = (recommended or {}).get("best_variant_per_family", {})
    if best_per_fam:
        w("=" * 78)
        w("PER-FAMILY RECOMMENDED CONFIG BLOCK")
        w("=" * 78)
        w(_render_per_family_config_block(best_per_fam))
        w("")

    # Recommended configs
    if recommended:
        w("=" * 78)
        w("FULL RECOMMENDED CONFIGS (JSON)")
        w("=" * 78)
        w(json.dumps(recommended, indent=2, default=str))
        w("")

    text = "\n".join(lines)
    (out_dir / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print(f"[summary] wrote {out_dir / 'SUMMARY.txt'}")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CONFIG_PATH)
    print(f"[run] config loaded from {CONFIG_PATH}")
    print(f"[run] writing bundle to {OUT_DIR}")

    res = _run_base(cfg)
    trades = res.trades
    candidates = res.candidates
    equity = res.equity_curve

    if len(trades) == 0:
        print("[run] no trades — aborting PNG bundle")
        return

    trades.to_parquet(OUT_DIR / "trades_XAUUSD.parquet", index=False)
    candidates.to_parquet(OUT_DIR / "candidates_XAUUSD.parquet", index=False)
    equity.to_parquet(OUT_DIR / "equity_XAUUSD.parquet", index=False)
    (OUT_DIR / "metrics_XAUUSD.json").write_text(
        json.dumps(res.metrics, indent=2, default=str)
    )

    vf = _run_variants(cfg, trades)
    if len(vf):
        vf.to_parquet(OUT_DIR / "variants_XAUUSD.parquet", index=False)

    recommended = _recommend_configs(vf, trades)
    (OUT_DIR / "recommended_config.json").write_text(
        json.dumps(recommended, indent=2, default=str)
    )

    # ---- Live telemetry reality check -------------------------------
    print("[live] loading live telemetry for reality check...")
    live = _load_live_trades()
    if len(live):
        live.to_parquet(OUT_DIR / "live_trades_recent.parquet", index=False)
        print(f"[live] loaded {len(live)} live trades across "
              f"{live['day'].nunique()} days")
    haircut = _compute_haircut(live, trades)
    if haircut:
        print(f"[live] backtest expectancy: "
              f"{haircut['bt_expectancy_pips']:+.2f} pips/trade, "
              f"live: {haircut['live_expectancy_pips']:+.2f} pips/trade "
              f"-> haircut {haircut['haircut_pips']:+.2f}")
        (OUT_DIR / "reality_check.json").write_text(
            json.dumps(
                {
                    "live_per_day": _live_summary(live),
                    "backtest_vs_live": haircut,
                },
                indent=2,
                default=str,
            )
        )

    leaderboard_haircut = (
        _apply_haircut_to_leaderboard(vf, haircut.get("haircut_pips", 0.0))
        if haircut else pd.DataFrame()
    )
    if len(leaderboard_haircut):
        leaderboard_haircut.to_parquet(
            OUT_DIR / "variants_haircut_leaderboard.parquet"
        )

    print("[plot] generating PNG bundle...")
    _plot_equity(equity, OUT_DIR / "01_equity_curve.png")
    _plot_drawdown(equity, OUT_DIR / "02_drawdown.png")
    _plot_family_pnl(trades, OUT_DIR / "03_family_direction_pnl.png")
    _plot_session(trades, OUT_DIR / "04_session_breakdown.png")
    _plot_hour(trades, OUT_DIR / "05_hour_breakdown.png")
    _plot_bucket_5m(trades, OUT_DIR / "06_bucket_5m_breakdown.png")
    _plot_hour_family_heatmap(trades, OUT_DIR / "07_family_hour_heatmap.png")
    _plot_family_session_stacked(trades, OUT_DIR / "08_family_session_stacked.png")
    _plot_block_reasons(candidates, OUT_DIR / "09_block_reasons.png")
    _plot_signal_funnel(candidates, trades, OUT_DIR / "10_signal_funnel.png")
    _plot_commission_impact(trades, OUT_DIR / "11_commission_impact.png")
    _plot_variant_leaderboard(vf, OUT_DIR / "12_variant_leaderboard.png")
    _plot_variant_by_family(vf, OUT_DIR / "13_variant_by_family.png")
    _plot_variant_by_session(vf, OUT_DIR / "14_variant_by_session.png")
    _plot_close_reasons(trades, vf, OUT_DIR / "15_close_reasons.png")
    best_variant = (
        recommended.get("best_variant_overall", {}).get("name", "base")
        if recommended else "base"
    )
    _plot_top_variant_5m(vf, best_variant, OUT_DIR / "16_best_variant_5m.png")
    _plot_reality_check(live, trades, haircut, OUT_DIR / "17_reality_check.png")
    _plot_haircut_leaderboard(
        leaderboard_haircut, OUT_DIR / "18_haircut_leaderboard.png"
    )

    _write_summary(
        OUT_DIR, trades, candidates, vf, recommended,
        live, haircut, leaderboard_haircut,
    )

    print(f"[run] done — bundle at {OUT_DIR}")


if __name__ == "__main__":
    main()
