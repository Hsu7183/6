from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import CostConfig


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def profit_factor(gross_profit: np.ndarray, gross_loss_abs: np.ndarray) -> np.ndarray:
    gross_profit = np.asarray(gross_profit, dtype=float)
    gross_loss_abs = np.asarray(gross_loss_abs, dtype=float)
    out = np.full(gross_profit.shape, np.nan, dtype=float)
    has_loss = gross_loss_abs > 0
    np.divide(gross_profit, gross_loss_abs, out=out, where=has_loss)
    out[(gross_loss_abs == 0) & (gross_profit > 0)] = np.inf
    return out


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def tax_twd(entry_price: float, exit_price: float, cost: CostConfig) -> int:
    entry_tax = round_half_up(entry_price * cost.point_value_twd * cost.tax_rate)
    exit_tax = round_half_up(exit_price * cost.point_value_twd * cost.tax_rate)
    return entry_tax + exit_tax


@dataclass(frozen=True)
class CostedTrade:
    raw_points: float
    net_points: float
    net_profit_twd: float
    fee_twd: int
    tax_twd: int
    slippage_twd: float
    cost_points: float
    effective_entry: float
    effective_exit: float


def costed_points(side: int, entry_price: float, exit_price: float, cost: CostConfig) -> CostedTrade:
    if side == 1:
        raw_points = exit_price - entry_price
        effective_entry = entry_price + cost.entry_slippage_points
        effective_exit = exit_price - cost.exit_slippage_points
        slipped_points = effective_exit - effective_entry
    else:
        raw_points = entry_price - exit_price
        effective_entry = entry_price - cost.entry_slippage_points
        effective_exit = exit_price + cost.exit_slippage_points
        slipped_points = effective_entry - effective_exit

    trade_tax = tax_twd(effective_entry, effective_exit, cost)
    net_profit_twd = slipped_points * cost.point_value_twd - cost.round_trip_fee_twd - trade_tax
    net_points = net_profit_twd / cost.point_value_twd
    return CostedTrade(
        raw_points=float(raw_points),
        net_points=float(net_points),
        net_profit_twd=float(net_profit_twd),
        fee_twd=cost.round_trip_fee_twd,
        tax_twd=trade_tax,
        slippage_twd=cost.slippage_points * cost.point_value_twd,
        cost_points=float(raw_points - net_points),
        effective_entry=float(effective_entry),
        effective_exit=float(effective_exit),
    )


def max_drawdown(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 0.0
    equity = np.cumsum(points)
    peak = np.maximum.accumulate(np.maximum(equity, 0.0))
    return float(np.max(peak - equity))


def max_losing_streak(points: np.ndarray) -> int:
    best = 0
    current = 0
    for point in points:
        if point < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def points_summary(points: np.ndarray) -> dict[str, float | int]:
    points = np.asarray(points, dtype=float)
    n = int(len(points))
    if n == 0:
        return {
            "TotalTrades": 0,
            "WinTrades": 0,
            "LossTrades": 0,
            "FlatTrades": 0,
            "WinRate": math.nan,
            "NetPoints": 0.0,
            "GrossProfitNet": 0.0,
            "GrossLossNet": 0.0,
            "PFNet": math.nan,
            "AvgNetPoints": math.nan,
            "MedianNetPoints": math.nan,
            "MaxTradeNetPoints": math.nan,
            "MinTradeNetPoints": math.nan,
            "StdNetPoints": math.nan,
            "MaxDrawdownNetPoints": 0.0,
            "MaxLosingStreak": 0,
        }
    wins = points > 0
    losses = points < 0
    gross_profit = float(points[wins].sum())
    gross_loss_abs = float(-points[losses].sum())
    if gross_loss_abs > 0:
        pf = gross_profit / gross_loss_abs
    else:
        pf = math.inf if gross_profit > 0 else math.nan
    return {
        "TotalTrades": n,
        "WinTrades": int(wins.sum()),
        "LossTrades": int(losses.sum()),
        "FlatTrades": int((points == 0).sum()),
        "WinRate": float(wins.sum() / n),
        "NetPoints": float(points.sum()),
        "GrossProfitNet": gross_profit,
        "GrossLossNet": -gross_loss_abs,
        "PFNet": pf,
        "AvgNetPoints": float(points.mean()),
        "MedianNetPoints": float(np.median(points)),
        "MaxTradeNetPoints": float(points.max()),
        "MinTradeNetPoints": float(points.min()),
        "StdNetPoints": float(points.std(ddof=0)),
        "MaxDrawdownNetPoints": max_drawdown(points),
        "MaxLosingStreak": max_losing_streak(points),
    }
