"""Attribute live trades by family by joining trades.csv -> tm_lab.csv.

Live trades.csv has no family column. tm_lab.csv has one row per (ticket,
variant) with the family label. We join trades -> tm_lab on ticket/parent_ticket,
dedupe to one family per ticket, then compute per-family live PnL.

Outputs:
    desktop/filohft_backtest_20260415/live_family_attribution.txt
    desktop/filohft_backtest_20260415/live_trades_with_family.parquet
    desktop/filohft_backtest_20260415/19_live_family_pnl.png
    desktop/filohft_backtest_20260415/20_live_vs_bt_family.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIVE_ROOT = Path("C:/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/telemetry")
LIVE_DAYS = ("20260413", "20260414", "20260415")
OUT_DIR = Path("path/to/output/filohft_backtest_20260415")

PIP_PRICE = 0.10
LOT = 0.10
USD_PER_PIP_PER_LOT = 10.0
COMMISSION = LOT * 7.0  # $7/lot round-turn @ 0.10 lots = $0.70


def _load_trades_one_day(day: str) -> pd.DataFrame:
    p = LIVE_ROOT / day / "trades.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, on_bad_lines="skip", engine="python")
    if df.empty:
        return df
    df["day"] = day
    return df


def _load_tmlab_one_day(day: str) -> pd.DataFrame:
    p = LIVE_ROOT / day / "tm_lab.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, on_bad_lines="skip", engine="python")
    if df.empty:
        return df
    df["day"] = day
    return df


def _load_fills_one_day(day: str) -> pd.DataFrame:
    p = LIVE_ROOT / day / "fills.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, on_bad_lines="skip", engine="python")


def _load_intents_one_day(day: str) -> pd.DataFrame:
    p = LIVE_ROOT / day / "intents.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, on_bad_lines="skip", engine="python")


def _load_live() -> pd.DataFrame:
    """Returns deduped live trades with family, direction, pips, pnl."""
    all_trades = pd.concat(
        [_load_trades_one_day(d) for d in LIVE_DAYS], ignore_index=True,
    )
    all_tmlab = pd.concat(
        [_load_tmlab_one_day(d) for d in LIVE_DAYS], ignore_index=True,
    )
    all_fills = pd.concat(
        [_load_fills_one_day(d) for d in LIVE_DAYS], ignore_index=True,
    )
    all_intents = pd.concat(
        [_load_intents_one_day(d) for d in LIVE_DAYS], ignore_index=True,
    )
    print(f"raw live trades rows={len(all_trades)}  raw tmlab rows={len(all_tmlab)}  "
          f"fills={len(all_fills)}  intents={len(all_intents)}")

    # Dedupe trades: earliest open, latest close per ticket
    opens = (
        all_trades[all_trades["event"] == "open"]
        .sort_values("ts_ms")
        .drop_duplicates("ticket", keep="first")[
            ["day", "ticket", "ts_ms", "direction", "price"]
        ]
        .rename(columns={"ts_ms": "ts_open_ms", "price": "entry"})
    )
    closes = (
        all_trades[all_trades["event"] == "close"]
        .sort_values("ts_ms")
        .drop_duplicates("ticket", keep="last")[
            ["ticket", "ts_ms", "price"]
        ]
        .rename(columns={"ts_ms": "ts_close_ms", "price": "exit"})
    )
    trades = opens.merge(closes, on="ticket", how="inner")
    print(f"deduped trades (open+close matched)={len(trades)}")

    # Family from tmlab: take base variant row per parent_ticket for authoritative labels
    tmlab_base = (
        all_tmlab[all_tmlab["variant"] == "base"]
        .drop_duplicates("parent_ticket", keep="first")[
            ["parent_ticket", "family", "entry_score", "entry_kl_weight",
             "entry_atr_pips", "entry_velocity", "echo_count"]
        ]
        .rename(columns={"parent_ticket": "ticket"})
    )
    print(f"tmlab base rows (unique tickets)={len(tmlab_base)}")

    trades = trades.merge(tmlab_base, on="ticket", how="left")
    matched_tmlab = trades["family"].notna().sum()
    print(f"matched via tm_lab  : {matched_tmlab}/{len(trades)}")

    # Fallback #1: ticket -> fills -> intent_id -> intents.family (for trades
    # not in tm_lab)
    if not all_fills.empty and not all_intents.empty:
        fills_unique = (
            all_fills.sort_values("ts_ms")
            .drop_duplicates("ticket", keep="first")[["ticket", "intent_id"]]
        )
        intents_fam = (
            all_intents[["intent_id", "family", "setup_score"]]
            .drop_duplicates("intent_id", keep="first")
            .rename(columns={"family": "family_int", "setup_score": "score_int"})
        )
        ticket_to_fam = fills_unique.merge(intents_fam, on="intent_id", how="left")
        trades = trades.merge(ticket_to_fam, on="ticket", how="left")
        # Fill missing family from intents path
        needs_fill = trades["family"].isna() & trades["family_int"].notna()
        trades.loc[needs_fill, "family"] = trades.loc[needs_fill, "family_int"]
        if "entry_score" in trades.columns:
            score_fill = trades["entry_score"].isna() & trades["score_int"].notna()
            trades.loc[score_fill, "entry_score"] = trades.loc[score_fill, "score_int"]
        trades = trades.drop(columns=["family_int", "score_int", "intent_id"], errors="ignore")

    matched_after_intents = trades["family"].notna().sum()
    print(f"matched via intents : +{matched_after_intents - matched_tmlab}  "
          f"(total {matched_after_intents}/{len(trades)})")

    # Fallback #2: time+direction fuzzy match to intents for still-unlabeled trades.
    # On 04-13 and early 04-14 the fills.csv ticket IDs don't match broker
    # tickets, so we find the intent that was created <= MATCH_WIN_MS before
    # each trade open with the same direction and the closest trigger price.
    MATCH_WIN_MS = 5000
    MAX_PX_GAP_PIPS = 3.0

    if not all_intents.empty:
        it = all_intents.copy()
        it = it[it["state"].isin(["created", "filled"])] if "state" in it.columns else it
        it = it.dropna(subset=["ts_ms", "family", "direction"])
        it = it.sort_values("ts_ms").reset_index(drop=True)
        it_ts = it["ts_ms"].to_numpy()

        unmatched_idx = trades.index[trades["family"].isna()].tolist()
        fuzzy_hits = 0
        for ridx in unmatched_idx:
            tr_ts = int(trades.at[ridx, "ts_open_ms"])
            tr_dir = str(trades.at[ridx, "direction"])
            tr_px = float(trades.at[ridx, "entry"])
            lo = np.searchsorted(it_ts, tr_ts - MATCH_WIN_MS, side="left")
            hi = np.searchsorted(it_ts, tr_ts, side="right")
            if lo >= hi:
                continue
            window = it.iloc[lo:hi]
            window = window[window["direction"].astype(str) == tr_dir]
            if window.empty:
                continue
            px_gaps = (window["trigger_price"].astype(float) - tr_px).abs() / PIP_PRICE
            within = window[px_gaps <= MAX_PX_GAP_PIPS]
            if within.empty:
                continue
            # take the one closest in time
            choice = within.iloc[-1]
            trades.at[ridx, "family"] = choice["family"]
            if pd.isna(trades.at[ridx, "entry_score"]) and "setup_score" in choice:
                trades.at[ridx, "entry_score"] = choice.get("setup_score")
            fuzzy_hits += 1
        print(f"matched via fuzzy   : +{fuzzy_hits}")

    matched_final = trades["family"].notna().sum()
    print(f"TOTAL matched       : {matched_final}/{len(trades)} "
          f"({100*matched_final/max(len(trades),1):.1f}%)")

    # PnL reconstruction (realized_usd in trades.csv is zero on closes)
    trades["pip_diff"] = (trades["exit"] - trades["entry"]) / PIP_PRICE
    is_short = trades["direction"].astype(str) == "short"
    trades.loc[is_short, "pip_diff"] *= -1
    trades["gross_usd"] = trades["pip_diff"] * LOT * USD_PER_PIP_PER_LOT
    trades["commission_usd"] = COMMISSION
    trades["net_usd"] = trades["gross_usd"] - trades["commission_usd"]
    trades["ts_open"] = pd.to_datetime(trades["ts_open_ms"], unit="ms", utc=True)
    trades["hour_utc"] = trades["ts_open"].dt.hour
    return trades


def _family_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    labeled = trades[trades["family"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame()
    g = labeled.groupby("family", dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "net_usd": g["net_usd"].sum().round(2),
        "gross_usd": g["gross_usd"].sum().round(2),
        "commission_usd": g["commission_usd"].sum().round(2),
        "pips_sum": g["pip_diff"].sum().round(1),
        "exp_pips": g["pip_diff"].mean().round(2),
        "win_rate": (g["pip_diff"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        "avg_score": g["entry_score"].mean().round(3),
        "avg_kl": g["entry_kl_weight"].mean().round(3),
        "echo_hit_rate": (g["echo_count"].apply(lambda s: (s >= 1).mean()) * 100).round(1),
    })
    out = out.sort_values("net_usd")
    return out


def _family_direction_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    labeled = trades[trades["family"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame()
    g = labeled.groupby(["family", "direction"], dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "net_usd": g["net_usd"].sum().round(2),
        "exp_pips": g["pip_diff"].mean().round(2),
        "win_rate": (g["pip_diff"].apply(lambda s: (s > 0).mean()) * 100).round(1),
    }).sort_values("net_usd")
    return out


def _family_day_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    labeled = trades[trades["family"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame()
    g = labeled.groupby(["day", "family"], dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "net_usd": g["net_usd"].sum().round(2),
        "exp_pips": g["pip_diff"].mean().round(2),
    }).reset_index()
    return out


def _family_session_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    labeled = trades[trades["family"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame()

    def session(h):
        if 0 <= h < 7:
            return "asia"
        if 7 <= h < 12:
            return "london"
        if 12 <= h < 16:
            return "ln_ny_overlap"
        if 16 <= h < 21:
            return "ny"
        return "offhours"

    labeled["session"] = labeled["hour_utc"].apply(session)
    g = labeled.groupby(["session", "family"], dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "net_usd": g["net_usd"].sum().round(2),
        "exp_pips": g["pip_diff"].mean().round(2),
        "win_rate": (g["pip_diff"].apply(lambda s: (s > 0).mean()) * 100).round(1),
    }).reset_index()
    return out


def _load_backtest_family() -> pd.DataFrame | None:
    """Backtest trades are in the variants file at base variant — load it."""
    bt_path = OUT_DIR / "variants.parquet"
    if not bt_path.exists():
        return None
    bt = pd.read_parquet(bt_path)
    base = bt[bt["variant"] == "base"].copy()
    if base.empty:
        return None
    g = base.groupby("family", dropna=False)
    return pd.DataFrame({
        "bt_trades": g.size(),
        "bt_net_usd": g["usd"].sum().round(2),
        "bt_exp_pips": g["pips"].mean().round(2),
        "bt_win_rate": (g["pips"].apply(lambda s: (s > 0).mean()) * 100).round(1),
    }).sort_values("bt_exp_pips", ascending=False)


def _render_txt(trades: pd.DataFrame, fam: pd.DataFrame, fam_dir: pd.DataFrame,
                fam_day: pd.DataFrame, fam_ses: pd.DataFrame,
                bt_fam: pd.DataFrame | None) -> str:
    labeled = trades["family"].notna().sum()
    total = len(trades)
    lines = []
    lines.append("=" * 72)
    lines.append("LIVE TRADES — FAMILY ATTRIBUTION (2026-04-13 .. 2026-04-15)")
    lines.append("=" * 72)
    lines.append(f"deduped trades      : {total}")
    lines.append(f"matched to family   : {labeled} ({100*labeled/max(total,1):.1f}%)")
    lines.append(f"unmatched           : {total - labeled}")
    lines.append(f"total net           : ${trades['net_usd'].sum():,.2f}")
    lines.append(f"labeled net         : ${trades[trades['family'].notna()]['net_usd'].sum():,.2f}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("PER FAMILY (sorted by net_usd ascending — worst first)")
    lines.append("-" * 72)
    lines.append(fam.to_string())
    lines.append("")

    lines.append("-" * 72)
    lines.append("PER FAMILY × DIRECTION")
    lines.append("-" * 72)
    lines.append(fam_dir.to_string())
    lines.append("")

    lines.append("-" * 72)
    lines.append("PER FAMILY × DAY")
    lines.append("-" * 72)
    lines.append(fam_day.to_string(index=False))
    lines.append("")

    lines.append("-" * 72)
    lines.append("PER FAMILY × SESSION")
    lines.append("-" * 72)
    lines.append(fam_ses.to_string(index=False))
    lines.append("")

    if bt_fam is not None:
        lines.append("-" * 72)
        lines.append("LIVE vs BACKTEST — per family")
        lines.append("-" * 72)
        merged = fam[["trades", "net_usd", "exp_pips", "win_rate"]].rename(
            columns={
                "trades": "live_trades",
                "net_usd": "live_net_usd",
                "exp_pips": "live_exp_pips",
                "win_rate": "live_wr",
            }
        ).join(bt_fam, how="outer")
        merged["exp_gap_pips"] = (merged["live_exp_pips"] - merged["bt_exp_pips"]).round(2)
        merged = merged.sort_values("exp_gap_pips")
        lines.append(merged.to_string())
        lines.append("")

    return "\n".join(lines)


def _plot_family_pnl(fam: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in fam["net_usd"]]
    ax.barh(fam.index, fam["net_usd"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Live per-family net PnL (04-13 → 04-15)")
    ax.set_xlabel("net USD")
    for i, (idx, row) in enumerate(fam.iterrows()):
        ax.text(
            row["net_usd"],
            i,
            f" {row['trades']:.0f}t  {row['exp_pips']:+.2f}p  wr={row['win_rate']:.0f}%",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_live_vs_bt(fam: pd.DataFrame, bt_fam: pd.DataFrame, out: Path) -> None:
    if bt_fam is None:
        return
    common = fam.index.intersection(bt_fam.index)
    if len(common) == 0:
        return
    df = pd.DataFrame({
        "live_exp": fam.loc[common, "exp_pips"],
        "bt_exp": bt_fam.loc[common, "bt_exp_pips"],
    }).sort_values("bt_exp", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["bt_exp"], width=w, label="backtest", color="#2980b9")
    ax.bar(x + w / 2, df["live_exp"], width=w, label="live", color="#e67e22")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("expectancy (pips/trade)")
    ax.set_title("Per-family expectancy — backtest vs live")
    ax.legend()
    for i, (fam_name, row) in enumerate(df.iterrows()):
        gap = row["live_exp"] - row["bt_exp"]
        ax.text(i, min(row["live_exp"], row["bt_exp"]) - 0.7,
                f"Δ{gap:+.1f}", ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trades = _load_live()
    trades.to_parquet(OUT_DIR / "live_trades_with_family.parquet", index=False)

    fam = _family_breakdown(trades)
    fam_dir = _family_direction_breakdown(trades)
    fam_day = _family_day_breakdown(trades)
    fam_ses = _family_session_breakdown(trades)
    bt_fam = _load_backtest_family()

    txt = _render_txt(trades, fam, fam_dir, fam_day, fam_ses, bt_fam)
    (OUT_DIR / "live_family_attribution.txt").write_text(txt, encoding="utf-8")
    print(txt)

    _plot_family_pnl(fam, OUT_DIR / "19_live_family_pnl.png")
    if bt_fam is not None:
        _plot_live_vs_bt(fam, bt_fam, OUT_DIR / "20_live_vs_bt_family.png")

    payload = {
        "total_trades": int(len(trades)),
        "labeled_trades": int(trades["family"].notna().sum()),
        "total_net_usd": float(trades["net_usd"].sum()),
        "per_family": fam.reset_index().to_dict(orient="records"),
        "per_family_direction": fam_dir.reset_index().to_dict(orient="records"),
    }
    (OUT_DIR / "live_family_attribution.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
