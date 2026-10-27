"""Reliable end-of-day recap using MT5 broker-authoritative history.

PnL comes from the MetaTrader5 Python package (DEAL_PROFIT + commission +
swap + fee — exact match with terminal). Family/session/variant attribution
comes from telemetry, joined via position_id = tm_lab.parent_ticket.

Usage:
    python day_recap_v2.py 20260415
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import MetaTrader5 as mt5  # type: ignore

LIVE_ROOT = Path("C:/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/telemetry")
OUT_DIR = Path("path/to/output/filohft_backtest_20260415")


def _session(h: int) -> str:
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "overlap"
    if 16 <= h < 21:
        return "ny"
    return "offhours"


def _read(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, on_bad_lines="skip", engine="python")


def fetch_positions(day: str) -> pd.DataFrame:
    """Return one row per closed position on `day` with broker PnL + attribution.

    A "position" groups deals by `position_id`; we aggregate entry (in) and
    exit (out) legs to get net position size, entry/exit vwap, and total PnL
    (profit + commission + swap + fee).
    """
    start = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init: {mt5.last_error()}")
    try:
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            raise RuntimeError(f"history_deals_get: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    if not deals:
        return pd.DataFrame()

    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df = df[df["symbol"] == "XAUUSD"].copy()
    df["ts"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    df["total"] = df["profit"] + df["commission"] + df["swap"] + df["fee"]

    # Pull position close timestamps: one per position_id from earliest "out" deal
    out_deals = df[df["entry"] == mt5.DEAL_ENTRY_OUT].copy()
    in_deals = df[df["entry"] == mt5.DEAL_ENTRY_IN].copy()

    # Aggregate per position
    agg_in = in_deals.groupby("position_id").agg(
        open_ts=("ts", "min"),
        open_vol=("volume", "sum"),
        open_vwap=("price", lambda s: np.average(s, weights=in_deals.loc[s.index, "volume"])),
        open_type=("type", "first"),
        open_comm=("commission", "sum"),
    )
    agg_out = out_deals.groupby("position_id").agg(
        close_ts=("ts", "max"),
        close_vol=("volume", "sum"),
        close_vwap=("price", lambda s: np.average(s, weights=out_deals.loc[s.index, "volume"])),
        close_comm=("commission", "sum"),
        profit=("profit", "sum"),
        swap=("swap", "sum"),
        fee=("fee", "sum"),
    )
    pos = agg_in.join(agg_out, how="outer")
    pos["direction"] = np.where(pos["open_type"] == mt5.DEAL_TYPE_BUY, "long", "short")
    pos["commission"] = pos["open_comm"].fillna(0) + pos["close_comm"].fillna(0)
    pos["total"] = (
        pos["profit"].fillna(0)
        + pos["commission"].fillna(0)
        + pos["swap"].fillna(0)
        + pos["fee"].fillna(0)
    )
    pos["hour"] = pos["open_ts"].dt.hour
    pos["session"] = pos["hour"].apply(_session)
    pos["pips"] = np.where(
        pos["direction"] == "long",
        (pos["close_vwap"] - pos["open_vwap"]) / 0.10,
        (pos["open_vwap"] - pos["close_vwap"]) / 0.10,
    )
    pos = pos.reset_index().rename(columns={"position_id": "ticket"})
    return pos


def attribute_families(pos: pd.DataFrame, day: str) -> pd.DataFrame:
    """Join position_id → tm_lab.parent_ticket and intents fallback for family."""
    root = LIVE_ROOT / day
    tm = _read(root / "tm_lab.csv")
    if not tm.empty and "variant" in tm.columns:
        base = (
            tm[tm["variant"] == "base"]
            .drop_duplicates("parent_ticket", keep="first")[
                ["parent_ticket", "family", "entry_score", "entry_kl_weight",
                 "entry_atr_pips", "entry_velocity", "echo_count"]
            ]
            .rename(columns={"parent_ticket": "ticket"})
        )
        pos = pos.merge(base, on="ticket", how="left")
    if "family" not in pos.columns:
        pos["family"] = np.nan
        pos["echo_count"] = np.nan
        pos["entry_score"] = np.nan

    # Intents fuzzy fallback by (ts, direction, trigger_price)
    intents = _read(root / "intents.csv")
    if not intents.empty:
        it = intents.dropna(subset=["ts_ms", "family", "direction"])
        if "state" in it.columns:
            it = it[it["state"].isin(["created", "filled"])]
        it = it.sort_values("ts_ms").reset_index(drop=True)
        it_ts = it["ts_ms"].to_numpy()
        need = pos.index[pos["family"].isna() & pos["open_ts"].notna()].tolist()
        WIN_MS = 5000
        GAP_PIPS = 3.0
        for ridx in need:
            tr_ts = int(pos.at[ridx, "open_ts"].timestamp() * 1000)
            tr_dir = str(pos.at[ridx, "direction"])
            tr_px = float(pos.at[ridx, "open_vwap"])
            lo = np.searchsorted(it_ts, tr_ts - WIN_MS, side="left")
            hi = np.searchsorted(it_ts, tr_ts, side="right")
            if lo >= hi:
                continue
            w = it.iloc[lo:hi]
            w = w[w["direction"].astype(str) == tr_dir]
            if w.empty:
                continue
            gaps = (w["trigger_price"].astype(float) - tr_px).abs() / 0.10
            w = w[gaps <= GAP_PIPS]
            if w.empty:
                continue
            choice = w.iloc[-1]
            pos.at[ridx, "family"] = choice["family"]
            if pd.isna(pos.at[ridx, "entry_score"]) and "setup_score" in choice.index:
                pos.at[ridx, "entry_score"] = choice.get("setup_score")
    return pos


def _grp_pnl(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "net_usd": g["total"].sum().round(2),
        "profit_usd": g["profit"].sum().round(2),
        "comm_usd": g["commission"].sum().round(2),
        "swap_usd": g["swap"].sum().round(2),
        "exp_pips": g["pips"].mean().round(2),
        "win_rate": (g["pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
    })
    return out


def _hdr(s: str, ch: str = "=") -> str:
    return "\n" + ch * 72 + "\n" + s + "\n" + ch * 72


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).strftime("%Y%m%d")
    pos = fetch_positions(day)
    if pos.empty:
        print(f"no positions for {day}")
        return
    pos = attribute_families(pos, day)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pos.to_parquet(OUT_DIR / f"day_recap_{day}_positions.parquet", index=False)

    lines: list[str] = []
    lines.append(_hdr(f"FiloHFT DAY RECAP — {day}  (broker-authoritative)"))
    lines.append(f"closed positions       : {len(pos)}")
    lines.append(f"volume (lots)          : {pos['open_vol'].sum():.2f}")
    lines.append(f"gross profit (DEAL_PROFIT): {pos['profit'].sum():+,.2f}")
    lines.append(f"commission             : {pos['commission'].sum():+,.2f}")
    lines.append(f"swap                   : {pos['swap'].sum():+,.2f}")
    lines.append(f"fee                    : {pos['fee'].sum():+,.2f}")
    lines.append(f"TOTAL (terminal)       : {pos['total'].sum():+,.2f}")
    lines.append(f"expectancy pips        : {pos['pips'].mean():+.2f}")
    lines.append(f"win rate               : {(pos['pips']>0).mean()*100:.1f}%")
    lines.append(f"first open ts          : {pos['open_ts'].min()}")
    lines.append(f"last open ts           : {pos['open_ts'].max()}")
    lab = pos["family"].notna().sum()
    lines.append(f"family-labeled         : {lab}/{len(pos)} ({100*lab/len(pos):.1f}%)")
    lines.append(f"labeled net USD        : {pos[pos['family'].notna()]['total'].sum():+,.2f}")
    lines.append(f"unlabeled net USD      : {pos[pos['family'].isna()]['total'].sum():+,.2f}")

    # ---- per family
    labd = pos[pos["family"].notna()]
    if not labd.empty:
        lines.append(_hdr("PER FAMILY", "-"))
        fam = _grp_pnl(labd, "family").sort_values("net_usd")
        lines.append(fam.to_string())

        lines.append(_hdr("PER FAMILY × DIRECTION", "-"))
        fd = _grp_pnl(labd, ["family", "direction"]).sort_values("net_usd")
        lines.append(fd.to_string())

        lines.append(_hdr("PER FAMILY × SESSION", "-"))
        fs = _grp_pnl(labd, ["family", "session"]).sort_values("net_usd")
        lines.append(fs.to_string())

    # ---- per session
    lines.append(_hdr("PER SESSION", "-"))
    ses = _grp_pnl(pos, "session").sort_values("net_usd")
    lines.append(ses.to_string())

    # ---- per hour
    lines.append(_hdr("PER HOUR (UTC)", "-"))
    hr = _grp_pnl(pos, "hour")
    lines.append(hr.to_string())

    # ---- 5-minute bucket top/bottom
    pos["bucket_5m"] = (pos["open_ts"].dt.hour * 60 + pos["open_ts"].dt.minute) // 5 * 5
    lines.append(_hdr("TOP/BOTTOM 10 5-MIN BUCKETS", "-"))
    b5 = _grp_pnl(pos, "bucket_5m")
    lines.append("TOP 10:\n" + b5.sort_values("net_usd", ascending=False).head(10).to_string())
    lines.append("\nBOTTOM 10:\n" + b5.sort_values("net_usd").head(10).to_string())

    # ---- echo tag split
    lines.append(_hdr("ECHO-TAG SPLIT (echo_count==0 vs >=1)", "-"))
    if "echo_count" in pos.columns and pos["echo_count"].notna().any():
        pos["echo_bucket"] = np.where(pos["echo_count"].fillna(-1) >= 1, "echo>=1",
                                       np.where(pos["echo_count"].fillna(-1) == 0, "echo==0", "unknown"))
        eb = _grp_pnl(pos[pos["echo_bucket"] != "unknown"], "echo_bucket")
        lines.append(eb.to_string())
    else:
        lines.append("(echo_count not available)")

    # ---- TMLab variant leaderboard + best per family
    lines.append(_hdr("TMLAB VARIANT LEADERBOARD (top & bottom 10)", "-"))
    tm = _read(LIVE_ROOT / day / "tm_lab.csv")
    if not tm.empty and "variant" in tm.columns and "realized_pips" in tm.columns:
        # TMLab rows are simulated PnL at 1 lot; scale back to real lot=open_vol avg
        # Note: tm_lab's realized_usd column uses its own lot sim. We recompute at 1.0 lot.
        tm["gross"] = tm["realized_pips"] * 1.0 * 10.0  # 1 lot × $10/pip
        tm["net"] = tm["gross"] - 7.0  # $7/lot round turn at 1 lot
        g = tm.groupby("variant")
        lb = pd.DataFrame({
            "trades": g.size(),
            "net_usd": g["net"].sum().round(2),
            "exp_pips": g["realized_pips"].mean().round(2),
            "win_rate": (g["realized_pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        }).sort_values("net_usd", ascending=False)
        lines.append("TOP 10:\n" + lb.head(10).to_string())
        lines.append("\nBOTTOM 10:\n" + lb.tail(10).to_string())

        if "family" in tm.columns:
            lines.append(_hdr("BEST TMLAB VARIANT PER FAMILY (min 5 samples)", "-"))
            g2 = tm.groupby(["family", "variant"])
            fv = pd.DataFrame({
                "trades": g2.size(),
                "net_usd": g2["net"].sum().round(2),
                "exp_pips": g2["realized_pips"].mean().round(2),
            }).reset_index()
            fv = fv[fv["trades"] >= 5]
            best = fv.sort_values("net_usd", ascending=False).drop_duplicates("family", keep="first")
            lines.append(best.to_string(index=False))

    # ---- signal funnel
    lines.append(_hdr("SIGNAL FUNNEL (setups.csv block reasons)", "-"))
    setups = _read(LIVE_ROOT / day / "setups.csv")
    if not setups.empty and "decision" in setups.columns:
        n_total = len(setups)
        n_block = (setups["decision"] == "blocked").sum()
        lines.append(f"total setup evaluations  : {n_total:,}")
        lines.append(f"  blocked                : {n_block:,} ({100*n_block/n_total:.2f}%)")
        if "block_reason" in setups.columns:
            br = (
                setups[setups["decision"] == "blocked"]["block_reason"]
                .fillna("(none)")
                .str.split(" ", n=1).str[0]
                .value_counts()
                .head(15)
            )
            lines.append("\ntop block categories:")
            lines.append(br.to_string())

    # ---- shadow stream
    lines.append(_hdr("SHADOW TRADES (signal-layer virtual stream)", "-"))
    sh = _read(LIVE_ROOT / day / "shadow_trades.csv")
    if not sh.empty and "realized_pips" in sh.columns:
        sh_exp = sh["realized_pips"].mean()
        sh_wr = (sh["realized_pips"] > 0).mean() * 100
        # Shadow is at 1 lot sim
        sh_net = sh["realized_pips"].sum() * 10.0 - len(sh) * 7.0
        lines.append(f"shadow rows       : {len(sh):,}")
        lines.append(f"shadow exp pips   : {sh_exp:+.2f}")
        lines.append(f"shadow WR         : {sh_wr:.1f}%")
        lines.append(f"shadow net @ 1lot : {sh_net:+,.2f}")
        if "family" in sh.columns:
            g = sh.groupby("family")
            sfam = pd.DataFrame({
                "trades": g.size(),
                "exp_pips": g["realized_pips"].mean().round(2),
                "win_rate": (g["realized_pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
            }).sort_values("exp_pips")
            lines.append("\nshadow per family:")
            lines.append(sfam.to_string())

    # ---- errors & safety
    lines.append(_hdr("ERRORS & SAFETY", "-"))
    errs = _read(LIVE_ROOT / day / "errors.csv")
    saf = _read(LIVE_ROOT / day / "safety.csv")
    lines.append(f"errors.csv rows : {len(errs):,}")
    lines.append(f"safety.csv rows : {len(saf):,}")

    txt = "\n".join(lines)
    print(txt)
    (OUT_DIR / f"day_recap_{day}.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
