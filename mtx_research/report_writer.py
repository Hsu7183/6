from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest_engine import materialize_r2a_trades
from .config import ResearchConfig


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def combine_csv_parts(parts: list[Path], output_path: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in parts if path.exists() and path.stat().st_size > 0]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_csv(combined, output_path)
    return combined


def _top(df: pd.DataFrame, sort_cols: list[str], ascending: list[bool], n: int = 500) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values(sort_cols, ascending=ascending, na_position="last").head(n).copy()


def write_r2a_final_outputs(
    *,
    outdir: Path,
    checkpoint_dir: Path,
    samples: pd.DataFrame,
    config: ResearchConfig,
) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_parts = sorted(checkpoint_dir.glob("summary_part_*.csv"))
    by_year_parts = sorted(checkpoint_dir.glob("by_year_part_*.csv"))
    by_side_parts = sorted(checkpoint_dir.glob("by_side_part_*.csv"))
    by_class_parts = sorted(checkpoint_dir.glob("by_open_class_part_*.csv"))
    by_segment_parts = sorted(checkpoint_dir.glob("by_time_segment_part_*.csv"))

    summary = combine_csv_parts(summary_parts, outdir / "summary_r2a_all.csv")
    by_year = combine_csv_parts(by_year_parts, outdir / "by_year_r2a.csv")
    by_side = combine_csv_parts(by_side_parts, outdir / "by_side_r2a.csv")
    by_class = combine_csv_parts(by_class_parts, outdir / "by_open_class_r2a.csv")
    by_segment = combine_csv_parts(by_segment_parts, outdir / "by_time_segment_r2a.csv")

    top_net = _top(summary, ["NetProfitTWD"], [False])
    write_csv(top_net, outdir / "top_r2a_net.csv")

    tradable = summary[summary["TotalTrades"] >= 300].copy() if "TotalTrades" in summary else pd.DataFrame()
    top_pf = _top(tradable, ["PFNet", "NetProfitTWD"], [False, False])
    top_avg = _top(tradable, ["AvgNetPoints", "NetProfitTWD"], [False, False])
    top_adv = _top(tradable, ["PullbackAdvantageNet", "NetProfitTWD"], [False, False])
    write_csv(top_pf, outdir / "top_r2a_pf.csv")
    write_csv(top_avg, outdir / "top_r2a_avg.csv")
    write_csv(top_adv, outdir / "top_r2a_pullback_advantage.csv")

    robust = tradable[
        (tradable["NetProfitTWD"] > 0)
        & (tradable["PFNet"] > 1.05)
        & (tradable["AvgNetPoints"] > 0)
        & (tradable["PositiveYears"] >= 4)
    ].copy()
    robust = _top(robust, ["PFNet", "AvgNetPoints", "NetProfitTWD"], [False, False, False], n=500)
    write_csv(robust, outdir / "top_r2a_robust.csv")

    candidates = build_r2a_candidates(robust, top_pf, top_avg, max_rows=config.r2b_max_candidates)
    write_csv(candidates, outdir / "top_r2a_candidates_for_r2b.csv")

    if not summary.empty:
        write_csv(_group_summary(summary, "Family"), outdir / "by_family_r2a.csv")
        write_csv(_group_summary(summary, "ExecMode"), outdir / "by_exec_mode_r2a.csv")
    if not by_side.empty:
        write_csv(by_side, outdir / "by_side_r2a.csv")
    if not by_class.empty:
        write_csv(by_class, outdir / "by_open_class_r2a.csv")
    if not by_segment.empty:
        write_csv(by_segment, outdir / "by_time_segment_r2a.csv")

    trade_dir = outdir / "top_50_r2a_trades"
    if trade_dir.exists():
        shutil.rmtree(trade_dir)
    trade_dir.mkdir(parents=True, exist_ok=True)
    trade_source = candidates.head(config.top_trade_logs)
    for rank, row in enumerate(trade_source.itertuples(index=False), start=1):
        rule = pd.Series(row._asdict())
        trades = materialize_r2a_trades(samples, rule, config)
        name = f"rank_{rank:02d}_rule_{int(rule.RuleID):07d}_trades.csv"
        write_csv(trades, trade_dir / name)

    return {
        "summary_rows": int(len(summary)),
        "tradable_count": int(len(tradable)),
        "net_positive_count": int((summary["NetProfitTWD"] > 0).sum()) if not summary.empty else 0,
        "tradable_net_positive_count": int((tradable["NetProfitTWD"] > 0).sum()) if not tradable.empty else 0,
        "tradable_pf_105_count": int((tradable["PFNet"] > 1.05).sum()) if not tradable.empty else 0,
        "robust_count": int(len(robust)),
        "candidate_count": int(len(candidates)),
    }


def build_r2a_candidates(robust: pd.DataFrame, top_pf: pd.DataFrame, top_avg: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    frames = []
    if not robust.empty:
        frames.append(robust)
    if len(pd.concat(frames, ignore_index=True)) < max_rows if frames else True:
        if not top_pf.empty:
            fallback_pf = top_pf[
                (top_pf["TotalTrades"] >= 300)
                & (top_pf["RawAvgPoints"] >= 5.0)
                & (top_pf["PFNet"] > 0.95)
                & (top_pf["PositiveYears"] >= 3)
            ]
            frames.append(fallback_pf)
    if frames:
        current = pd.concat(frames, ignore_index=True).drop_duplicates("StrategyID")
    else:
        current = pd.DataFrame()
    if len(current) < max_rows and not top_avg.empty:
        current = pd.concat([current, top_avg], ignore_index=True).drop_duplicates("StrategyID")
    if current.empty:
        return current
    return current.head(max_rows).copy()


def _group_summary(summary: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, group in summary.groupby(group_col, dropna=False):
        tradable = group[group["TotalTrades"] >= 300]
        rows.append(
            {
                group_col: key,
                "ComboCount": int(len(group)),
                "ComboCount_TotalTrades300": int(len(tradable)),
                "ComboCount_NetProfitPositive": int((group["NetProfitTWD"] > 0).sum()),
                "ComboCount_PFNet105": int((group["PFNet"] > 1.05).sum()),
                "BestNetProfitTWD": float(group["NetProfitTWD"].max()) if len(group) else np.nan,
                "BestPFNet": float(group["PFNet"].replace(np.inf, np.nan).max()) if len(group) else np.nan,
                "BestAvgNetPoints": float(group["AvgNetPoints"].max()) if len(group) else np.nan,
                "MedianPFNet": float(group["PFNet"].replace(np.inf, np.nan).median()) if len(group) else np.nan,
                "MedianAvgNetPoints": float(group["AvgNetPoints"].median()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_next_research_suggestion(path: Path, candidate_count: int) -> None:
    text = f"""R2B skipped because candidate count < 100.

candidate_count={candidate_count}

建議：
1. 如果大部分 Family 的 RawAvgPoints < 4，先精簡前一根 K 的條件，不要急著測更多出場。
2. 如果 RawAvgPoints >= 5 但 NetProfitTWD 轉負，優先測 R2B 出場組合。
3. 如果某些 ExecMode 明顯拖累，下一輪先排除該 ExecMode。
4. 如果 Long / Short 差異很大，下一輪拆 LongOnly / ShortOnly。
5. 如果 TimeSegment 差異很大，下一輪加入時間段濾網。
"""
    path.write_text(text, encoding="utf-8")
