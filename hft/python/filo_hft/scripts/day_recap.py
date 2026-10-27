"""End-of-day recap for a single trading day.

Full breakdown across the required slices:
    - per family
    - per family x direction
    - per session
    - per hour (UTC)
    - TMLab variant leaderboard
    - TMLab best variant per family
    - echo-tag split (echo_count==0 vs >=1)
    - signal funnel (setups.csv block reasons)
    - errors / safety summary

Usage:
    python day_recap.py 20260415
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIVE_ROOT = Path("C:/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/telemetry")
OUT_DIR = Path("path/to/output/filohft_backtest_20260415")

PIP_PRICE = 0.10
LOT = 0.10
USD_PER_PIP_PER_LOT = 10.0
COMMISSION = LOT * 7.0  # $0.70 per round-turn


def _read(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, on_bad_lines="skip", engine="python")


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


def _load_executed(day: str) -> pd.DataFrame:
    root = LIVE_ROOT / day
    trades = _read(root / "trades.csv")
    if trades.empty:
        return trades
    opens = (
        trades[trades["event"] == "open"]
        .sort_values("ts_ms")
        .drop_duplicates("ticket", keep="first")[["ticket", "ts_ms", "direction", "price"]]
        .rename(columns={"ts_ms": "ts_open_ms", "price": "entry"})
    )
    closes = (
        trades[trades["event"] == "close"]
        .sort_values("ts_ms")
        .drop_duplicates("ticket", keep="last")[["ticket", "ts_ms", "price"]]
        .rename(columns={"ts_ms": "ts_close_ms", "price": "exit"})
    )
    df = opens.merge(closes, on="ticket", how="inner")

    # Family attribution via tm_lab (base row) then intents fuzzy
    tm = _read(root / "tm_lab.csv")
    if not tm.empty:
        base = (
            tm[tm["variant"] == "base"]
            .drop_duplicates("parent_ticket", keep="first")[
                ["parent_ticket", "family", "entry_score", "entry_kl_weight",
                 "entry_atr_pips", "entry_velocity", "echo_count"]
            ]
            .rename(columns={"parent_ticket": "ticket"})
        )
        df = df.merge(base, on="ticket", how="left")

    intents = _read(root / "intents.csv")
    if not intents.empty:
        it = intents[intents["state"].isin(["created", "filled"])].copy() if "state" in intents.columns else intents.copy()
        it = it.dropna(subset=["ts_ms", "family", "direction"]).sort_values("ts_ms").reset_index(drop=True)
        it_ts = it["ts_ms"].to_numpy()

        needs = df.index[df.get("family", pd.Series(dtype=object)).isna() | df["family"].isna()].tolist() if "family" in df.columns else df.index.tolist()
        if "family" not in df.columns:
            df["family"] = np.nan
            df["entry_score"] = np.nan
            df["echo_count"] = np.nan

        MATCH_WIN_MS = 5000
        MAX_PX_GAP_PIPS = 3.0
        for ridx in needs:
            tr_ts = int(df.at[ridx, "ts_open_ms"])
            tr_dir = str(df.at[ridx, "direction"])
            tr_px = float(df.at[ridx, "entry"])
            lo = np.searchsorted(it_ts, tr_ts - MATCH_WIN_MS, side="left")
            hi = np.searchsorted(it_ts, tr_ts, side="right")
            if lo >= hi:
                continue
            w = it.iloc[lo:hi]
            w = w[w["direction"].astype(str) == tr_dir]
            if w.empty:
                continue
            gaps = (w["trigger_price"].astype(float) - tr_px).abs() / PIP_PRICE
            w = w[gaps <= MAX_PX_GAP_PIPS]
            if w.empty:
                continue
            choice = w.iloc[-1]
            df.at[ridx, "family"] = choice["family"]
            if pd.isna(df.at[ridx, "entry_score"]):
                df.at[ridx, "entry_score"] = choice.get("setup_score")

    # PnL
    df["pips"] = (df["exit"] - df["entry"]) / PIP_PRICE
    df.loc[df["direction"].astype(str) == "short", "pips"] *= -1
    df["gross_usd"] = df["pips"] * LOT * USD_PER_PIP_PER_LOT
    df["commission_usd"] = COMMISSION
    df["net_usd"] = df["gross_usd"] - df["commission_usd"]
    df["ts_open"] = pd.to_datetime(df["ts_open_ms"], unit="ms", utc=True)
    df["hour"] = df["ts_open"].dt.hour
    df["session"] = df["hour"].apply(_session)
    return df


def _fmt(n: float) -> str:
    return f"{n:+,.2f}" if abs(n) > 0.005 else "0.00"


def _hdr(s: str, ch: str = "=") -> str:
    return "\n" + ch * 72 + "\n" + s + "\n" + ch * 72


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else "20260415"
    root = LIVE_ROOT / day
    if not root.exists():
        print(f"no telemetry dir for {day}")
        return

    df = _load_executed(day)
    out_lines: list[str] = []

    # ---- Topline --------------------------------------------------
    out_lines.append(_hdr(f"FiloHFT DAY RECAP — {day}"))
    out_lines.append(f"executed trades (deduped)  : {len(df)}")
    out_lines.append(f"gross USD                  : {_fmt(df['gross_usd'].sum())}")
    out_lines.append(f"commissions                : {_fmt(df['commission_usd'].sum())}")
    out_lines.append(f"NET USD                    : {_fmt(df['net_usd'].sum())}")
    if len(df):
        exp = df["pips"].mean()
        wr = (df["pips"] > 0).mean() * 100
        out_lines.append(f"expectancy (pips/trade)    : {exp:+.2f}")
        out_lines.append(f"win rate                   : {wr:.1f}%")
        out_lines.append(f"first trade ts (UTC)       : {df['ts_open'].min()}")
        out_lines.append(f"last trade ts (UTC)        : {df['ts_open'].max()}")
        labeled = df["family"].notna().sum()
        out_lines.append(f"family-labeled             : {labeled}/{len(df)} ({100*labeled/len(df):.1f}%)")

    # ---- Per family ----------------------------------------------
    out_lines.append(_hdr("PER FAMILY", "-"))
    lab = df[df["family"].notna()].copy()
    if len(lab):
        g = lab.groupby("family")
        fam = pd.DataFrame({
            "trades": g.size(),
            "net_usd": g["net_usd"].sum().round(2),
            "exp_pips": g["pips"].mean().round(2),
            "win_rate": (g["pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
            "avg_score": g["entry_score"].mean().round(3),
        }).sort_values("net_usd")
        out_lines.append(fam.to_string())

    # ---- Family x direction --------------------------------------
    out_lines.append(_hdr("PER FAMILY × DIRECTION", "-"))
    if len(lab):
        g = lab.groupby(["family", "direction"])
        fd = pd.DataFrame({
            "trades": g.size(),
            "net_usd": g["net_usd"].sum().round(2),
            "exp_pips": g["pips"].mean().round(2),
            "win_rate": (g["pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        }).sort_values("net_usd")
        out_lines.append(fd.to_string())

    # ---- Per session ---------------------------------------------
    out_lines.append(_hdr("PER SESSION", "-"))
    if len(df):
        g = df.groupby("session")
        ses = pd.DataFrame({
            "trades": g.size(),
            "net_usd": g["net_usd"].sum().round(2),
            "exp_pips": g["pips"].mean().round(2),
            "win_rate": (g["pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        }).sort_values("net_usd")
        out_lines.append(ses.to_string())

    # ---- Per hour ------------------------------------------------
    out_lines.append(_hdr("PER HOUR (UTC)", "-"))
    if len(df):
        g = df.groupby("hour")
        hr = pd.DataFrame({
            "trades": g.size(),
            "net_usd": g["net_usd"].sum().round(2),
            "exp_pips": g["pips"].mean().round(2),
        })
        out_lines.append(hr.to_string())

    # ---- TMLab variant leaderboard ------------------------------
    out_lines.append(_hdr("TMLAB VARIANT LEADERBOARD (top + bottom)", "-"))
    tm = _read(root / "tm_lab.csv")
    if not tm.empty and "variant" in tm.columns and "realized_pips" in tm.columns:
        tm["gross_usd_var"] = tm["realized_pips"] * LOT * USD_PER_PIP_PER_LOT
        tm["net_usd_var"] = tm["gross_usd_var"] - COMMISSION
        g = tm.groupby("variant")
        lb = pd.DataFrame({
            "trades": g.size(),
            "net_usd": g["net_usd_var"].sum().round(2),
            "exp_pips": g["realized_pips"].mean().round(2),
            "win_rate": (g["realized_pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        }).sort_values("net_usd", ascending=False)
        out_lines.append("TOP 10")
        out_lines.append(lb.head(10).to_string())
        out_lines.append("\nBOTTOM 10")
        out_lines.append(lb.tail(10).to_string())

        # ---- Best variant per family ----------------------------
        out_lines.append(_hdr("BEST TMLAB VARIANT PER FAMILY", "-"))
        if "family" in tm.columns:
            g2 = tm.groupby(["family", "variant"])
            fv = pd.DataFrame({
                "trades": g2.size(),
                "net_usd": g2["net_usd_var"].sum().round(2),
                "exp_pips": g2["realized_pips"].mean().round(2),
            }).reset_index()
            # require at least 5 samples per cell for meaning
            fv = fv[fv["trades"] >= 5]
            if len(fv):
                best = fv.sort_values("net_usd", ascending=False).drop_duplicates("family", keep="first")
                out_lines.append(best.to_string(index=False))

    # ---- Echo-tag split -----------------------------------------
    out_lines.append(_hdr("ECHO-TAG SPLIT (echo_count==0 vs >=1)", "-"))
    if not tm.empty and "echo_count" in tm.columns and "realized_pips" in tm.columns:
        tm_base = tm[tm["variant"] == "base"].copy() if "variant" in tm.columns else tm.copy()
        tm_base["echo_bucket"] = np.where(tm_base["echo_count"].fillna(0) >= 1, "echo>=1", "echo==0")
        g = tm_base.groupby("echo_bucket")
        eb = pd.DataFrame({
            "trades": g.size(),
            "net_usd": (g["realized_pips"].sum() * LOT * USD_PER_PIP_PER_LOT - g.size() * COMMISSION).round(2),
            "exp_pips": g["realized_pips"].mean().round(2),
            "win_rate": (g["realized_pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        })
        out_lines.append(eb.to_string())
    else:
        out_lines.append("(echo_count not available)")

    # ---- Signal funnel (setups.csv block reasons) ---------------
    out_lines.append(_hdr("SIGNAL FUNNEL — setups.csv block reasons", "-"))
    setups = _read(root / "setups.csv")
    if not setups.empty and "decision" in setups.columns:
        total = len(setups)
        allowed = (setups["decision"] == "allowed").sum()
        blocked = total - allowed
        out_lines.append(f"total setup evaluations  : {total:,}")
        out_lines.append(f"  allowed                : {allowed:,}  ({100*allowed/total:.2f}%)")
        out_lines.append(f"  blocked                : {blocked:,}  ({100*blocked/total:.2f}%)")
        # Top block reasons
        if "block_reason" in setups.columns:
            br = (
                setups[setups["decision"] == "blocked"]["block_reason"]
                .fillna("(none)")
                .str.split(" ", n=1).str[0]  # coarse bucket by first token
                .value_counts()
                .head(15)
            )
            out_lines.append("\ntop block categories (by leading token):")
            out_lines.append(br.to_string())

        # Family split of allowed candidates
        if "family" in setups.columns:
            fam_allowed = setups[setups["decision"] == "allowed"]["family"].value_counts()
            out_lines.append("\nallowed candidates per family:")
            out_lines.append(fam_allowed.to_string())

    # ---- Shadow trades -----------------------------------------
    out_lines.append(_hdr("SHADOW TRADES (signal-layer virtual stream)", "-"))
    shadow = _read(root / "shadow_trades.csv")
    if not shadow.empty:
        out_lines.append(f"shadow trade rows        : {len(shadow):,}")
        if "realized_pips" in shadow.columns:
            sh_exp = shadow["realized_pips"].mean()
            sh_wr = (shadow["realized_pips"] > 0).mean() * 100
            sh_net = (shadow["realized_pips"].sum() * LOT * USD_PER_PIP_PER_LOT
                      - len(shadow) * COMMISSION)
            out_lines.append(f"shadow expectancy (pips) : {sh_exp:+.2f}")
            out_lines.append(f"shadow win rate          : {sh_wr:.1f}%")
            out_lines.append(f"shadow implied net USD   : {sh_net:+,.2f}")
            if "family" in shadow.columns:
                g = shadow.groupby("family")
                sfam = pd.DataFrame({
                    "trades": g.size(),
                    "exp_pips": g["realized_pips"].mean().round(2),
                    "win_rate": (g["realized_pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
                }).sort_values("exp_pips")
                out_lines.append("\nshadow per family:")
                out_lines.append(sfam.to_string())

    # ---- Errors / safety ---------------------------------------
    out_lines.append(_hdr("ERRORS & SAFETY", "-"))
    errs = _read(root / "errors.csv")
    saf = _read(root / "safety.csv")
    out_lines.append(f"errors.csv rows   : {len(errs):,}")
    out_lines.append(f"safety.csv rows   : {len(saf):,}")
    if not errs.empty and "code" in errs.columns:
        ec = errs["code"].value_counts().head(5)
        out_lines.append("\ntop error codes:")
        out_lines.append(ec.to_string())

    txt = "\n".join(out_lines)
    print(txt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"day_recap_{day}.txt").write_text(txt, encoding="utf-8")
    df.to_parquet(OUT_DIR / f"day_recap_{day}_executed.parquet", index=False)


if __name__ == "__main__":
    main()
