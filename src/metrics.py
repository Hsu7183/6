from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")


def wilson_lower_bound(win_count: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return float("nan")
    phat = win_count / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def max_drawdown(points: np.ndarray) -> float:
    if len(points) == 0:
        return float("nan")
    equity = np.cumsum(points)
    peak = np.maximum.accumulate(np.maximum(equity, 0))
    return float(np.max(peak - equity))


def max_streak(points: np.ndarray, positive: bool) -> int:
    best = 0
    current = 0
    for point in points:
        ok = point > 0 if positive else point < 0
        if ok:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def metric_dict(points: np.ndarray, datetimes: np.ndarray | None = None) -> dict[str, object]:
    points = np.asarray(points, dtype=float)
    n = int(len(points))
    if n == 0:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "tie_count": 0,
            "win_rate": np.nan,
            "loss_rate": np.nan,
            "tie_rate": np.nan,
            "avg_points": np.nan,
            "median_points": np.nan,
            "gross_points": 0.0,
            "avg_win_points": np.nan,
            "avg_loss_points": np.nan,
            "profit_factor": np.nan,
            "profit_factor_capped": np.nan,
            "max_drawdown_points": np.nan,
            "max_losing_streak": 0,
            "max_winning_streak": 0,
            "first_trade_datetime": "",
            "last_trade_datetime": "",
            "wilson_win_rate_lower_95": np.nan,
        }

    wins = points > 0
    losses = points < 0
    ties = points == 0
    win_count = int(wins.sum())
    loss_count = int(losses.sum())
    tie_count = int(ties.sum())
    gross_win = float(points[wins].sum())
    gross_loss = float(points[losses].sum())
    if loss_count == 0:
        profit_factor = np.inf if gross_win > 0 else np.nan
    else:
        profit_factor = gross_win / abs(gross_loss)
    first_dt = ""
    last_dt = ""
    if datetimes is not None and len(datetimes):
        first_dt = str(datetimes[0])
        last_dt = str(datetimes[-1])

    return {
        "trade_count": n,
        "win_count": win_count,
        "loss_count": loss_count,
        "tie_count": tie_count,
        "win_rate": win_count / n,
        "loss_rate": loss_count / n,
        "tie_rate": tie_count / n,
        "avg_points": float(points.mean()),
        "median_points": float(np.median(points)),
        "gross_points": float(points.sum()),
        "avg_win_points": float(points[wins].mean()) if win_count else np.nan,
        "avg_loss_points": float(points[losses].mean()) if loss_count else np.nan,
        "profit_factor": profit_factor,
        "profit_factor_capped": min(profit_factor, 10) if np.isfinite(profit_factor) else 10.0,
        "max_drawdown_points": max_drawdown(points),
        "max_losing_streak": max_streak(points, positive=False),
        "max_winning_streak": max_streak(points, positive=True),
        "first_trade_datetime": first_dt,
        "last_trade_datetime": last_dt,
        "wilson_win_rate_lower_95": wilson_lower_bound(win_count, n),
    }


def split_metric_dict(points: np.ndarray) -> dict[str, object]:
    base = metric_dict(points)
    return {
        "trade_count": base["trade_count"],
        "win_rate": base["win_rate"],
        "avg_points": base["avg_points"],
        "profit_factor": base["profit_factor"],
    }


def robust_score(row: pd.Series, prefix: str, min_trades: int) -> float:
    pf = row.get(f"{prefix}_profit_factor", np.nan)
    pf_capped = min(float(pf), 3.0) if pd.notna(pf) and np.isfinite(pf) else 3.0
    wilson = row.get(f"{prefix}_wilson_win_rate_lower_95", 0)
    avg_points = row.get(f"{prefix}_avg_points", 0)
    valid_win_rate = row.get(f"{prefix}_valid_win_rate", 0)
    test_win_rate = row.get(f"{prefix}_test_win_rate", 0)
    wilson = 0 if pd.isna(wilson) else float(wilson)
    avg_points = 0 if pd.isna(avg_points) else float(avg_points)
    valid_win_rate = 0 if pd.isna(valid_win_rate) else float(valid_win_rate)
    test_win_rate = 0 if pd.isna(test_win_rate) else float(test_win_rate)
    score = 0.0
    score += wilson * 100
    score += avg_points * 5
    score += pf_capped * 5
    score += valid_win_rate * 10
    score += test_win_rate * 20
    if pd.isna(row.get(f"{prefix}_valid_avg_points")) or float(row.get(f"{prefix}_valid_avg_points")) <= 0:
        score -= 20
    if pd.isna(row.get(f"{prefix}_test_avg_points")) or float(row.get(f"{prefix}_test_avg_points")) <= 0:
        score -= 30
    if int(row.get(f"{prefix}_trade_count", 0) or 0) < min_trades:
        score -= 50
    return score


def aggregate_trades(
    trades: pd.DataFrame,
    *,
    num_rules: int,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame({"rule_id": np.arange(1, num_rules + 1)})

    ordered = trades.sort_values(["rule_id", "order"], kind="mergesort")
    for rule_id, group in ordered.groupby("rule_id", sort=False):
        values = metric_dict(group["points"].to_numpy(), group["datetime"].to_numpy())
        values = {f"{prefix}_{key}": value for key, value in values.items()}
        values["rule_id"] = int(rule_id)
        rows.append(values)
    result = pd.DataFrame(rows)
    all_rules = pd.DataFrame({"rule_id": np.arange(1, num_rules + 1)})
    result = all_rules.merge(result, on="rule_id", how="left")
    count_cols = [
        f"{prefix}_trade_count",
        f"{prefix}_win_count",
        f"{prefix}_loss_count",
        f"{prefix}_tie_count",
        f"{prefix}_max_losing_streak",
        f"{prefix}_max_winning_streak",
    ]
    for col in count_cols:
        if col in result:
            result[col] = result[col].fillna(0).astype(int)

    split_rows: list[dict[str, object]] = []
    for (rule_id, split), group in ordered.groupby(["rule_id", "split"], sort=False):
        vals = split_metric_dict(group["points"].to_numpy())
        split_rows.append(
            {
                "rule_id": int(rule_id),
                "split": split,
                **{f"{prefix}_{split}_{key}": value for key, value in vals.items()},
            }
        )
    if split_rows:
        split_df = pd.DataFrame(split_rows)
        wide = pd.DataFrame({"rule_id": np.arange(1, num_rules + 1)})
        for split in SPLITS:
            part = split_df[split_df["split"] == split].drop(columns=["split"])
            wide = wide.merge(part, on="rule_id", how="left")
        result = result.merge(wide, on="rule_id", how="left")

    for split in SPLITS:
        count_col = f"{prefix}_{split}_trade_count"
        if count_col in result:
            result[count_col] = result[count_col].fillna(0).astype(int)
    return result
