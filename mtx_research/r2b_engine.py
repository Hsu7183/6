from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .backtest_engine import materialize_r2a_trades
from .checkpoint import append_log
from .config import ResearchConfig
from .metrics import costed_points, points_summary
from .report_writer import write_csv, write_next_research_suggestion


def exit_param_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bars in cfg.R2B_EXIT_BARS_LIST:
        rows.append({"ExitMode": "E01_HOLD_N_OPEN", "ExitBars": bars, "TP": np.nan, "SL": np.nan, "MaxBars": np.nan})
    for tp in cfg.R2B_TP_LIST:
        for sl in cfg.R2B_SL_LIST:
            for max_bars in cfg.R2B_MAX_BARS_LIST:
                rows.append({"ExitMode": "E02_TP_SL_MAXBARS", "ExitBars": np.nan, "TP": tp, "SL": sl, "MaxBars": max_bars})
    return pd.DataFrame(rows)


def _bar_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "datetime": df["datetime"].to_numpy(),
        "date": df["DateInt"].to_numpy(),
        "year": df["Year"].to_numpy(),
        "open": df["Open"].to_numpy(dtype=float),
        "high": df["High"].to_numpy(dtype=float),
        "low": df["Low"].to_numpy(dtype=float),
    }


def _e01_exit(bars: dict[str, np.ndarray], trade: object, exit_bars: int) -> tuple[int, float, str] | None:
    entry_index = int(trade.EntryIndex)
    exit_index = entry_index + int(exit_bars)
    if exit_index >= len(bars["open"]):
        return None
    if int(bars["date"][exit_index]) != int(bars["date"][entry_index]):
        return None
    return exit_index, float(bars["open"][exit_index]), f"HOLD_{exit_bars}_OPEN"


def _e02_exit(bars: dict[str, np.ndarray], trade: object, tp: float, sl: float, max_bars: int) -> tuple[int, float, str] | None:
    entry_index = int(trade.EntryIndex)
    entry_px = float(trade.EntryPx)
    side = 1 if str(trade.Side) == "LONG" else -1
    last_index = entry_index + int(max_bars)
    if last_index >= len(bars["open"]):
        return None
    if int(bars["date"][last_index]) != int(bars["date"][entry_index]):
        return None

    for idx in range(entry_index + 1, last_index + 1):
        if int(bars["date"][idx]) != int(bars["date"][entry_index]):
            return None
        high = float(bars["high"][idx])
        low = float(bars["low"][idx])
        if side == 1:
            hit_tp = high >= entry_px + tp
            hit_sl = low <= entry_px - sl
            if hit_tp and hit_sl:
                return idx, entry_px - sl, "SL_WORST_CASE"
            if hit_sl:
                return idx, entry_px - sl, "SL"
            if hit_tp:
                return idx, entry_px + tp, "TP"
        else:
            hit_tp = low <= entry_px - tp
            hit_sl = high >= entry_px + sl
            if hit_tp and hit_sl:
                return idx, entry_px + sl, "SL_WORST_CASE"
            if hit_sl:
                return idx, entry_px + sl, "SL"
            if hit_tp:
                return idx, entry_px - tp, "TP"
    return last_index, float(bars["open"][last_index]), f"MAXBARS_{max_bars}_OPEN"


def evaluate_exit_mode(
    *,
    bars: dict[str, np.ndarray],
    base_trades: pd.DataFrame,
    rule: pd.Series,
    exit_rule: pd.Series,
    config: ResearchConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    last_exit_index = -10_000_000
    for trade in base_trades.sort_values("EntryIndex").itertuples(index=False):
        if int(trade.EntryIndex) <= last_exit_index:
            continue
        if exit_rule.ExitMode == "E01_HOLD_N_OPEN":
            exit_info = _e01_exit(bars, trade, int(exit_rule.ExitBars))
        else:
            exit_info = _e02_exit(bars, trade, float(exit_rule.TP), float(exit_rule.SL), int(exit_rule.MaxBars))
        if exit_info is None:
            continue
        exit_index, exit_px, exit_reason = exit_info
        side = 1 if str(trade.Side) == "LONG" else -1
        costed = costed_points(side, float(trade.EntryPx), exit_px, config.cost)
        rows.append(
            {
                "StrategyID": rule.StrategyID,
                "RuleID": int(rule.RuleID),
                "Family": rule.Family,
                "ExecMode": rule.ExecMode,
                "ExitMode": exit_rule.ExitMode,
                "ExitBars": exit_rule.ExitBars,
                "TP": exit_rule.TP,
                "SL": exit_rule.SL,
                "MaxBars": exit_rule.MaxBars,
                "EntryIndex": int(trade.EntryIndex),
                "ExitIndex": int(exit_index),
                "EntryDate": trade.EntryDate,
                "EntryTime": trade.EntryTime,
                "ExitDate": str(pd.Timestamp(bars["datetime"][exit_index]).date()),
                "ExitTime": pd.Timestamp(bars["datetime"][exit_index]).strftime("%H:%M:%S"),
                "Year": int(bars["year"][exit_index]),
                "ExitReason": exit_reason,
                "Side": trade.Side,
                "EntryPx": float(trade.EntryPx),
                "ExitPx": exit_px,
                "RawPoints": costed.raw_points,
                "NetPoints": costed.net_points,
                "NetProfitTWD": costed.net_profit_twd,
                "FeeTWD": costed.fee_twd,
                "TaxTWD": costed.tax_twd,
                "SlippageTWD": costed.slippage_twd,
                "CostPoints": costed.cost_points,
            }
        )
        last_exit_index = int(exit_index)

    trades = pd.DataFrame(rows)
    points = trades["NetPoints"].to_numpy(dtype=float) if not trades.empty else np.asarray([], dtype=float)
    metrics = points_summary(points)
    summary = {
        "Stage": "R2B",
        "StrategyID": rule.StrategyID,
        "RuleID": int(rule.RuleID),
        "Family": rule.Family,
        "ExecMode": rule.ExecMode,
        "ExitMode": exit_rule.ExitMode,
        "ExitBars": exit_rule.ExitBars,
        "TP": exit_rule.TP,
        "SL": exit_rule.SL,
        "MaxBars": exit_rule.MaxBars,
        **metrics,
    }
    summary["NetProfitTWD"] = summary["NetPoints"] * config.cost.point_value_twd
    summary["TotalReturnRate"] = summary["NetProfitTWD"] / config.cost.capital_twd
    summary["TotalFeeTWD"] = float(trades["FeeTWD"].sum()) if not trades.empty else 0.0
    summary["TotalTaxTWD"] = float(trades["TaxTWD"].sum()) if not trades.empty else 0.0
    summary["TotalSlippageTWD"] = float(trades["SlippageTWD"].sum()) if not trades.empty else 0.0
    if not trades.empty:
        by_year = trades.groupby("Year")["NetPoints"].sum()
        summary["YearCount"] = int(len(by_year))
        summary["PositiveYears"] = int((by_year > 0).sum())
        summary["WorstYearNetPoints"] = float(by_year.min())
        summary["BestYearNetPoints"] = float(by_year.max())
    else:
        summary["YearCount"] = 0
        summary["PositiveYears"] = 0
        summary["WorstYearNetPoints"] = np.nan
        summary["BestYearNetPoints"] = np.nan
    return summary, trades


def run_r2b(
    *,
    df: pd.DataFrame,
    samples: pd.DataFrame,
    outdir: Path,
    config: ResearchConfig,
    progress_every: int = 25,
) -> None:
    r2a_dir = outdir / "r2a_1k_trend_pullback_all_families"
    r2b_dir = outdir / "r2b_exit_universe"
    r2b_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = r2a_dir / "top_r2a_candidates_for_r2b.csv"
    if not candidates_path.exists():
        append_log(outdir, "R2B skipped: top_r2a_candidates_for_r2b.csv not found")
        return
    candidates = pd.read_csv(candidates_path)
    if len(candidates) < 100:
        append_log(outdir, f"R2B skipped because candidate count < 100: {len(candidates):,}")
        write_next_research_suggestion(r2b_dir / "next_research_suggestion.txt", len(candidates))
        return
    candidates = candidates.head(config.r2b_max_candidates).copy()
    exits = exit_param_grid()
    bars = _bar_arrays(df)

    summaries: list[dict[str, object]] = []
    trade_dir = r2b_dir / "top_50_r2b_trades"
    trade_dir.mkdir(parents=True, exist_ok=True)
    trade_logs: list[tuple[float, pd.DataFrame, dict[str, object]]] = []

    append_log(outdir, f"R2B start candidates={len(candidates):,} exit_combos={len(exits):,}")
    for cand_i, rule_tuple in enumerate(candidates.itertuples(index=False), start=1):
        rule = pd.Series(rule_tuple._asdict())
        base_trades = materialize_r2a_trades(samples, rule, config)
        for exit_tuple in exits.itertuples(index=False):
            exit_rule = pd.Series(exit_tuple._asdict())
            summary, trades = evaluate_exit_mode(
                bars=bars,
                base_trades=base_trades,
                rule=rule,
                exit_rule=exit_rule,
                config=config,
            )
            summaries.append(summary)
            if len(trade_logs) < config.top_trade_logs or summary["NetProfitTWD"] > min(v for v, _, _ in trade_logs):
                trade_logs.append((float(summary["NetProfitTWD"]), trades, summary))
                trade_logs = sorted(trade_logs, key=lambda item: item[0], reverse=True)[: config.top_trade_logs]
        if progress_every and cand_i % progress_every == 0:
            append_log(outdir, f"R2B progress candidates={cand_i:,}/{len(candidates):,}")

    summary_df = pd.DataFrame(summaries)
    write_csv(summary_df, r2b_dir / "summary_r2b_all.csv")
    tradable = summary_df[summary_df["TotalTrades"] >= 300].copy()
    write_csv(summary_df.sort_values("NetProfitTWD", ascending=False).head(500), r2b_dir / "top_r2b_net.csv")
    write_csv(tradable.sort_values("PFNet", ascending=False, na_position="last").head(500), r2b_dir / "top_r2b_pf.csv")
    write_csv(tradable.sort_values("AvgNetPoints", ascending=False, na_position="last").head(500), r2b_dir / "top_r2b_avg.csv")
    robust = tradable[
        (tradable["NetProfitTWD"] > 0)
        & (tradable["PFNet"] > 1.05)
        & (tradable["AvgNetPoints"] > 0)
        & (tradable["PositiveYears"] >= 4)
    ].copy()
    write_csv(
        robust.sort_values(["PFNet", "AvgNetPoints", "NetProfitTWD", "MaxDrawdownNetPoints"], ascending=[False, False, False, True]).head(500),
        r2b_dir / "top_r2b_robust.csv",
    )

    for group_col, file_name in [
        ("Family", "by_family_r2b.csv"),
        ("ExecMode", "by_exec_mode_r2b.csv"),
        ("ExitMode", "by_exit_mode_r2b.csv"),
    ]:
        grouped = summary_df.groupby(group_col, dropna=False).agg(
            ComboCount=("StrategyID", "count"),
            BestNetProfitTWD=("NetProfitTWD", "max"),
            BestPFNet=("PFNet", "max"),
            BestAvgNetPoints=("AvgNetPoints", "max"),
            MedianAvgNetPoints=("AvgNetPoints", "median"),
        )
        write_csv(grouped.reset_index(), r2b_dir / file_name)

    for rank, (_score, trades, summary) in enumerate(trade_logs, start=1):
        if trades.empty:
            continue
        name = f"rank_{rank:02d}_rule_{int(summary['RuleID']):07d}_{summary['ExitMode']}_trades.csv"
        write_csv(trades, trade_dir / name)

    append_log(outdir, f"R2B done summary_rows={len(summary_df):,} robust={len(robust):,}")
