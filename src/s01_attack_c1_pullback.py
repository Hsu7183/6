from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .build_samples import build_samples
from .s01_html_report import write_s01_html_report
from .load_data import load_price_data


STRATEGY_ID = "S01_ATTACK_C1_PULLBACK"
CAPITAL_TWD = 250_000
POINT_VALUE_TWD = 50
ENTRY_SLIPPAGE_POINTS = 0
EXIT_SLIPPAGE_POINTS = 2
FEE_PER_SIDE_TWD = 18
ROUND_TURN_FEE_TWD = FEE_PER_SIDE_TWD * 2
TRANSACTION_TAX_RATE = 0.00002

BODY_MIN_LIST = [2, 5, 8, 12, 18, 25, 35]
RANGE_MAX_LIST = [30, 50, 80, 120, 180]
BODY_PCT_MIN_LIST = [35, 45, 55, 65, 75]
CLOSE_POS_MIN_LIST = [60, 70, 80, 90]
UPPER_TAIL_MAX_LIST = [10, 20, 30, 40]
OPEN_GAP_MIN_LIST = [1, 2, 3, 5, 8, 12]
OPEN_GAP_MAX_LIST = [5, 8, 12, 18, 25, 35, 50]
PENETRATE_LIST = [1, 2, 3]

EXPECTED_PARAM_COUNT = 318_240
OPEN_CLASS_NAMES = ("INSIDE", "BREAK_SMALL", "BREAK_LARGE")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = numerator.astype(float, copy=False)
    denominator = denominator.astype(float, copy=False)
    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _profit_factor(gross_profit: np.ndarray, gross_loss_abs: np.ndarray) -> np.ndarray:
    out = np.full(gross_profit.shape, np.nan, dtype=float)
    has_loss = gross_loss_abs > 0
    np.divide(gross_profit, gross_loss_abs, out=out, where=has_loss)
    out[(gross_loss_abs == 0) & (gross_profit > 0)] = np.inf
    return out


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _tax_twd(entry_price: float, exit_price: float) -> int:
    entry_tax = _round_half_up(entry_price * POINT_VALUE_TWD * TRANSACTION_TAX_RATE)
    exit_tax = _round_half_up(exit_price * POINT_VALUE_TWD * TRANSACTION_TAX_RATE)
    return entry_tax + exit_tax


def _costed_pullback_points(side: int, entry_price: float, next_open: float) -> tuple[float, float, int, int, int]:
    if side == 1:
        raw_points = next_open - entry_price
        effective_entry = entry_price + ENTRY_SLIPPAGE_POINTS
        effective_exit = next_open - EXIT_SLIPPAGE_POINTS
        slipped_points = effective_exit - effective_entry
    else:
        raw_points = entry_price - next_open
        effective_entry = entry_price - ENTRY_SLIPPAGE_POINTS
        effective_exit = next_open + EXIT_SLIPPAGE_POINTS
        slipped_points = effective_entry - effective_exit
    tax = _tax_twd(effective_entry, effective_exit)
    net_profit = slipped_points * POINT_VALUE_TWD - ROUND_TURN_FEE_TWD - tax
    return raw_points, net_profit / POINT_VALUE_TWD, int(net_profit), ROUND_TURN_FEE_TWD, tax


def _costed_direct_open_points(side: int, open_price: float, next_open: float) -> tuple[float, float]:
    if side == 1:
        raw_points = next_open - open_price
        effective_entry = open_price + ENTRY_SLIPPAGE_POINTS
        effective_exit = next_open - EXIT_SLIPPAGE_POINTS
        slipped_points = effective_exit - effective_entry
    else:
        raw_points = open_price - next_open
        effective_entry = open_price - ENTRY_SLIPPAGE_POINTS
        effective_exit = next_open + EXIT_SLIPPAGE_POINTS
        slipped_points = effective_entry - effective_exit
    tax = _tax_twd(effective_entry, effective_exit)
    net_profit = slipped_points * POINT_VALUE_TWD - ROUND_TURN_FEE_TWD - tax
    return raw_points, net_profit / POINT_VALUE_TWD


def _threshold_high_index(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(thresholds, dtype=float), values, side="right") - 1


def _threshold_low_index(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(thresholds, dtype=float), values, side="left")


def _gap_bounds(gap: float) -> tuple[int, int] | None:
    mins = np.asarray(OPEN_GAP_MIN_LIST, dtype=float)
    maxs = np.asarray(OPEN_GAP_MAX_LIST, dtype=float)
    min_hi = int(np.searchsorted(mins, gap, side="right") - 1)
    max_lo = int(np.searchsorted(maxs, gap, side="left"))
    if min_hi < 0 or max_lo >= len(maxs):
        return None
    return min_hi, max_lo


def _legal_mask(shape: tuple[int, ...]) -> np.ndarray:
    body_min = np.asarray(BODY_MIN_LIST)[:, None, None, None, None, None, None, None]
    range_max = np.asarray(RANGE_MAX_LIST)[None, :, None, None, None, None, None, None]
    gap_min = np.asarray(OPEN_GAP_MIN_LIST)[None, None, None, None, None, :, None, None]
    gap_max = np.asarray(OPEN_GAP_MAX_LIST)[None, None, None, None, None, None, :, None]
    legal = (body_min <= range_max) & (gap_max >= gap_min)
    return np.broadcast_to(legal, shape).copy()


def build_param_grid() -> tuple[pd.DataFrame, tuple[np.ndarray, ...], np.ndarray]:
    shape = (
        len(BODY_MIN_LIST),
        len(RANGE_MAX_LIST),
        len(BODY_PCT_MIN_LIST),
        len(CLOSE_POS_MIN_LIST),
        len(UPPER_TAIL_MAX_LIST),
        len(OPEN_GAP_MIN_LIST),
        len(OPEN_GAP_MAX_LIST),
        len(PENETRATE_LIST),
    )
    legal = _legal_mask(shape)
    coords = np.where(legal)

    close_pos = np.asarray(CLOSE_POS_MIN_LIST, dtype=int)[coords[3]]
    upper_tail = np.asarray(UPPER_TAIL_MAX_LIST, dtype=int)[coords[4]]
    params = pd.DataFrame(
        {
            "RuleID": np.arange(1, len(coords[0]) + 1, dtype=np.int32),
            "StrategyID": STRATEGY_ID,
            "BodyMin": np.asarray(BODY_MIN_LIST, dtype=int)[coords[0]],
            "RangeMax": np.asarray(RANGE_MAX_LIST, dtype=int)[coords[1]],
            "BodyPctMin": np.asarray(BODY_PCT_MIN_LIST, dtype=int)[coords[2]],
            "ClosePosMin": close_pos,
            "UpperTailMax": upper_tail,
            "EffOppTailMax": np.minimum(upper_tail, 100 - close_pos),
            "OpenGapMin": np.asarray(OPEN_GAP_MIN_LIST, dtype=int)[coords[5]],
            "OpenGapMax": np.asarray(OPEN_GAP_MAX_LIST, dtype=int)[coords[6]],
            "Penetrate": np.asarray(PENETRATE_LIST, dtype=int)[coords[7]],
            "CostPoints": 0.0,
        }
    )
    if len(params) != EXPECTED_PARAM_COUNT:
        raise RuntimeError(f"legal parameter count {len(params):,} != {EXPECTED_PARAM_COUNT:,}")
    return params, coords, legal


def _add_count(arr: np.ndarray, slc: tuple[slice, ...], mask: np.ndarray) -> None:
    if mask.any():
        arr[slc][mask] += 1


def _add_sum(arr: np.ndarray, slc: tuple[slice, ...], mask: np.ndarray, value: float) -> None:
    if mask.any():
        arr[slc][mask] += value


def _open_class_long(o0: float, h1: float) -> int:
    if o0 <= h1:
        return 0
    if o0 <= h1 + 8:
        return 1
    return 2


def _open_class_short(o0: float, l1: float) -> int:
    if o0 >= l1:
        return 0
    if o0 >= l1 - 8:
        return 1
    return 2


@dataclass
class TradeStats:
    shape: tuple[int, ...]
    years: tuple[int, ...]

    def __post_init__(self) -> None:
        self.trade_count = np.zeros(self.shape, dtype=np.int32)
        self.win_count = np.zeros(self.shape, dtype=np.int32)
        self.loss_count = np.zeros(self.shape, dtype=np.int32)
        self.flat_count = np.zeros(self.shape, dtype=np.int32)
        self.sum_points = np.zeros(self.shape, dtype=np.float64)
        self.sum_sq_points = np.zeros(self.shape, dtype=np.float64)
        self.raw_sum_points = np.zeros(self.shape, dtype=np.float64)
        self.total_fee_twd = np.zeros(self.shape, dtype=np.float64)
        self.total_tax_twd = np.zeros(self.shape, dtype=np.float64)
        self.total_slippage_twd = np.zeros(self.shape, dtype=np.float64)
        self.gross_profit = np.zeros(self.shape, dtype=np.float64)
        self.gross_loss_abs = np.zeros(self.shape, dtype=np.float64)
        self.max_trade = np.full(self.shape, -np.inf, dtype=np.float64)
        self.min_trade = np.full(self.shape, np.inf, dtype=np.float64)
        self.equity = np.zeros(self.shape, dtype=np.float64)
        self.peak = np.zeros(self.shape, dtype=np.float64)
        self.max_drawdown = np.zeros(self.shape, dtype=np.float64)
        self.current_losing_streak = np.zeros(self.shape, dtype=np.int32)
        self.max_losing_streak = np.zeros(self.shape, dtype=np.int32)
        self.long_count = np.zeros(self.shape, dtype=np.int32)
        self.short_count = np.zeros(self.shape, dtype=np.int32)

        year_shape = (len(self.years), *self.shape)
        self.year_trade_count = np.zeros(year_shape, dtype=np.int32)
        self.year_long_count = np.zeros(year_shape, dtype=np.int32)
        self.year_short_count = np.zeros(year_shape, dtype=np.int32)
        self.year_win_count = np.zeros(year_shape, dtype=np.int32)
        self.year_sum_points = np.zeros(year_shape, dtype=np.float64)
        self.year_raw_sum_points = np.zeros(year_shape, dtype=np.float64)
        self.year_gross_profit = np.zeros(year_shape, dtype=np.float64)
        self.year_gross_loss_abs = np.zeros(year_shape, dtype=np.float64)
        self.year_equity = np.zeros(year_shape, dtype=np.float64)
        self.year_peak = np.zeros(year_shape, dtype=np.float64)
        self.year_max_drawdown = np.zeros(year_shape, dtype=np.float64)

        class_shape = (len(OPEN_CLASS_NAMES), *self.shape)
        self.class_trade_count = np.zeros(class_shape, dtype=np.int32)
        self.class_win_count = np.zeros(class_shape, dtype=np.int32)
        self.class_sum_points = np.zeros(class_shape, dtype=np.float64)
        self.class_raw_sum_points = np.zeros(class_shape, dtype=np.float64)
        self.class_gross_profit = np.zeros(class_shape, dtype=np.float64)
        self.class_gross_loss_abs = np.zeros(class_shape, dtype=np.float64)

    def update(
        self,
        slc: tuple[slice, ...],
        mask: np.ndarray,
        *,
        point: float,
        raw_point: float,
        fee_twd: int,
        tax_twd: int,
        slippage_twd: int,
        year_index: int,
        open_class_index: int,
        side: int,
    ) -> None:
        if not mask.any():
            return

        self.trade_count[slc][mask] += 1
        self.sum_points[slc][mask] += point
        self.sum_sq_points[slc][mask] += point * point
        self.raw_sum_points[slc][mask] += raw_point
        self.total_fee_twd[slc][mask] += fee_twd
        self.total_tax_twd[slc][mask] += tax_twd
        self.total_slippage_twd[slc][mask] += slippage_twd
        self.max_trade[slc][mask] = np.maximum(self.max_trade[slc][mask], point)
        self.min_trade[slc][mask] = np.minimum(self.min_trade[slc][mask], point)

        if side == 1:
            self.long_count[slc][mask] += 1
            self.year_long_count[year_index][slc][mask] += 1
        else:
            self.short_count[slc][mask] += 1
            self.year_short_count[year_index][slc][mask] += 1

        self.equity[slc][mask] += point
        equity_view = self.equity[slc]
        peak_view = self.peak[slc]
        peak_view[mask] = np.maximum(peak_view[mask], equity_view[mask])
        drawdown = peak_view[mask] - equity_view[mask]
        mdd_view = self.max_drawdown[slc]
        mdd_view[mask] = np.maximum(mdd_view[mask], drawdown)

        self.year_trade_count[year_index][slc][mask] += 1
        self.year_sum_points[year_index][slc][mask] += point
        self.year_raw_sum_points[year_index][slc][mask] += raw_point
        self.year_equity[year_index][slc][mask] += point
        year_equity = self.year_equity[year_index][slc]
        year_peak = self.year_peak[year_index][slc]
        year_peak[mask] = np.maximum(year_peak[mask], year_equity[mask])
        year_mdd = self.year_max_drawdown[year_index][slc]
        year_mdd[mask] = np.maximum(year_mdd[mask], year_peak[mask] - year_equity[mask])

        self.class_trade_count[open_class_index][slc][mask] += 1
        self.class_sum_points[open_class_index][slc][mask] += point
        self.class_raw_sum_points[open_class_index][slc][mask] += raw_point

        if point > 0:
            self.win_count[slc][mask] += 1
            self.gross_profit[slc][mask] += point
            self.year_win_count[year_index][slc][mask] += 1
            self.year_gross_profit[year_index][slc][mask] += point
            self.class_win_count[open_class_index][slc][mask] += 1
            self.class_gross_profit[open_class_index][slc][mask] += point
            self.current_losing_streak[slc][mask] = 0
        elif point < 0:
            loss = -point
            self.loss_count[slc][mask] += 1
            self.gross_loss_abs[slc][mask] += loss
            self.year_gross_loss_abs[year_index][slc][mask] += loss
            self.class_gross_loss_abs[open_class_index][slc][mask] += loss
            streak = self.current_losing_streak[slc]
            streak[mask] += 1
            max_streak = self.max_losing_streak[slc]
            max_streak[mask] = np.maximum(max_streak[mask], streak[mask])
        else:
            self.flat_count[slc][mask] += 1
            self.current_losing_streak[slc][mask] = 0


@dataclass
class S01Result:
    summary: pd.DataFrame
    by_year: pd.DataFrame
    by_open_class: pd.DataFrame
    params: pd.DataFrame


def scan(samples: pd.DataFrame, *, progress_every: int = 10000) -> S01Result:
    params, coords, legal = build_param_grid()
    shape = legal.shape
    years = tuple(int(year) for year in sorted(samples["year"].unique()))
    year_to_index = {year: idx for idx, year in enumerate(years)}
    stats = TradeStats(shape, years)

    long_points_all = samples["O_NEXT"].to_numpy(dtype=float) - samples["C1"].to_numpy(dtype=float)
    short_points_all = -long_points_all
    point_min = int(np.floor(min(np.nanmin(long_points_all), np.nanmin(short_points_all))))
    point_max = int(np.ceil(max(np.nanmax(long_points_all), np.nanmax(short_points_all))))
    point_values = np.arange(point_min, point_max + 1, dtype=float)
    point_hist = np.zeros((len(point_values), *shape), dtype=np.uint32)

    raw_trigger_count = np.zeros(shape, dtype=np.int32)
    eligible_trigger_count = np.zeros(shape, dtype=np.int32)
    long_trigger_count = np.zeros(shape, dtype=np.int32)
    short_trigger_count = np.zeros(shape, dtype=np.int32)
    fill_count = np.zeros(shape, dtype=np.int32)

    direct_all_sum = np.zeros(shape, dtype=np.float64)
    direct_filled_sum = np.zeros(shape, dtype=np.float64)
    direct_unfilled_sum = np.zeros(shape, dtype=np.float64)
    direct_all_count = np.zeros(shape, dtype=np.int32)
    direct_filled_count = np.zeros(shape, dtype=np.int32)
    direct_unfilled_count = np.zeros(shape, dtype=np.int32)

    class_eligible_count = np.zeros((len(OPEN_CLASS_NAMES), *shape), dtype=np.int32)
    class_fill_count = np.zeros((len(OPEN_CLASS_NAMES), *shape), dtype=np.int32)
    class_direct_all_sum = np.zeros((len(OPEN_CLASS_NAMES), *shape), dtype=np.float64)
    class_direct_all_count = np.zeros((len(OPEN_CLASS_NAMES), *shape), dtype=np.int32)
    class_direct_unfilled_sum = np.zeros((len(OPEN_CLASS_NAMES), *shape), dtype=np.float64)
    class_direct_unfilled_count = np.zeros((len(OPEN_CLASS_NAMES), *shape), dtype=np.int32)

    last_entry_order = np.full(shape, -10_000_000, dtype=np.int32)

    o1 = samples["O1"].to_numpy(dtype=float)
    h1 = samples["H1"].to_numpy(dtype=float)
    l1 = samples["L1"].to_numpy(dtype=float)
    c1 = samples["C1"].to_numpy(dtype=float)
    o0 = samples["O0"].to_numpy(dtype=float)
    h0 = samples["H0"].to_numpy(dtype=float)
    l0 = samples["L0"].to_numpy(dtype=float)
    o_next = samples["O_NEXT"].to_numpy(dtype=float)
    years_arr = samples["year"].map(year_to_index).to_numpy(dtype=np.int16)

    range1 = h1 - l1
    long_body = c1 - o1
    short_body = o1 - c1
    long_upper_tail = h1 - c1
    short_lower_tail = c1 - l1
    body_pct_long = np.divide(long_body * 100, range1, out=np.full_like(range1, np.nan), where=range1 > 0)
    body_pct_short = np.divide(short_body * 100, range1, out=np.full_like(range1, np.nan), where=range1 > 0)
    upper_tail_pct = np.divide(long_upper_tail * 100, range1, out=np.full_like(range1, np.nan), where=range1 > 0)
    lower_tail_pct = np.divide(short_lower_tail * 100, range1, out=np.full_like(range1, np.nan), where=range1 > 0)
    close_pos_pct = np.divide((c1 - l1) * 100, range1, out=np.full_like(range1, np.nan), where=range1 > 0)

    body_hi_long = _threshold_high_index(long_body, BODY_MIN_LIST)
    body_hi_short = _threshold_high_index(short_body, BODY_MIN_LIST)
    range_lo = _threshold_low_index(range1, RANGE_MAX_LIST)
    body_pct_hi_long = _threshold_high_index(body_pct_long, BODY_PCT_MIN_LIST)
    body_pct_hi_short = _threshold_high_index(body_pct_short, BODY_PCT_MIN_LIST)
    close_pos_hi_long = _threshold_high_index(close_pos_pct, CLOSE_POS_MIN_LIST)
    close_pos_hi_short = _threshold_high_index(100 - close_pos_pct, CLOSE_POS_MIN_LIST)
    tail_lo_long = _threshold_low_index(upper_tail_pct, UPPER_TAIL_MAX_LIST)
    tail_lo_short = _threshold_low_index(lower_tail_pct, UPPER_TAIL_MAX_LIST)
    pen_hi_long = _threshold_high_index(c1 - l0, PENETRATE_LIST)
    pen_hi_short = _threshold_high_index(h0 - c1, PENETRATE_LIST)

    total = len(samples)
    for order in range(total):
        if range1[order] > 0 and c1[order] > o1[order]:
            bounds = _gap_bounds(float(o0[order] - c1[order]))
            if bounds is not None:
                min_hi, max_lo = bounds
                if (
                    body_hi_long[order] >= 0
                    and range_lo[order] < len(RANGE_MAX_LIST)
                    and body_pct_hi_long[order] >= 0
                    and close_pos_hi_long[order] >= 0
                    and tail_lo_long[order] < len(UPPER_TAIL_MAX_LIST)
                ):
                    trigger_slc = (
                        slice(0, int(body_hi_long[order]) + 1),
                        slice(int(range_lo[order]), len(RANGE_MAX_LIST)),
                        slice(0, int(body_pct_hi_long[order]) + 1),
                        slice(0, int(close_pos_hi_long[order]) + 1),
                        slice(int(tail_lo_long[order]), len(UPPER_TAIL_MAX_LIST)),
                        slice(0, min_hi + 1),
                        slice(max_lo, len(OPEN_GAP_MAX_LIST)),
                        slice(None),
                    )
                    raw_mask = legal[trigger_slc]
                    active = raw_mask & (last_entry_order[trigger_slc] < order - 1)
                    open_class = _open_class_long(float(o0[order]), float(h1[order]))
                    raw_direct_point, direct_point = _costed_direct_open_points(
                        1, float(o0[order]), float(o_next[order])
                    )

                    _add_count(raw_trigger_count, trigger_slc, raw_mask)
                    _add_count(eligible_trigger_count, trigger_slc, active)
                    _add_count(long_trigger_count, trigger_slc, active)
                    _add_count(direct_all_count, trigger_slc, active)
                    _add_sum(direct_all_sum, trigger_slc, active, direct_point)
                    _add_count(class_eligible_count[open_class], trigger_slc, active)
                    _add_count(class_direct_all_count[open_class], trigger_slc, active)
                    _add_sum(class_direct_all_sum[open_class], trigger_slc, active, direct_point)

                    pen_hi = int(pen_hi_long[order])
                    if pen_hi >= 0:
                        fill_slc = (*trigger_slc[:-1], slice(0, pen_hi + 1))
                        fill_active = legal[fill_slc] & (last_entry_order[fill_slc] < order - 1)
                        raw_point, point, _net_profit, fee_twd, tax_twd = _costed_pullback_points(
                            1, float(c1[order]), float(o_next[order])
                        )
                        point_bin = int(round(raw_point - point_min))
                        _add_count(fill_count, fill_slc, fill_active)
                        _add_count(direct_filled_count, fill_slc, fill_active)
                        _add_sum(direct_filled_sum, fill_slc, fill_active, direct_point)
                        _add_count(class_fill_count[open_class], fill_slc, fill_active)
                        point_hist[point_bin][fill_slc][fill_active] += 1
                        stats.update(
                            fill_slc,
                            fill_active,
                            point=point,
                            raw_point=raw_point,
                            fee_twd=fee_twd,
                            tax_twd=tax_twd,
                            slippage_twd=EXIT_SLIPPAGE_POINTS * POINT_VALUE_TWD,
                            year_index=int(years_arr[order]),
                            open_class_index=open_class,
                            side=1,
                        )
                        last_entry_order[fill_slc][fill_active] = order
                    if pen_hi < len(PENETRATE_LIST) - 1:
                        unfilled_slc = (*trigger_slc[:-1], slice(max(pen_hi + 1, 0), len(PENETRATE_LIST)))
                        unfilled_active = legal[unfilled_slc] & (last_entry_order[unfilled_slc] < order - 1)
                        _add_count(direct_unfilled_count, unfilled_slc, unfilled_active)
                        _add_sum(direct_unfilled_sum, unfilled_slc, unfilled_active, direct_point)
                        _add_count(class_direct_unfilled_count[open_class], unfilled_slc, unfilled_active)
                        _add_sum(class_direct_unfilled_sum[open_class], unfilled_slc, unfilled_active, direct_point)

        if range1[order] > 0 and c1[order] < o1[order]:
            bounds = _gap_bounds(float(c1[order] - o0[order]))
            if bounds is not None:
                min_hi, max_lo = bounds
                if (
                    body_hi_short[order] >= 0
                    and range_lo[order] < len(RANGE_MAX_LIST)
                    and body_pct_hi_short[order] >= 0
                    and close_pos_hi_short[order] >= 0
                    and tail_lo_short[order] < len(UPPER_TAIL_MAX_LIST)
                ):
                    trigger_slc = (
                        slice(0, int(body_hi_short[order]) + 1),
                        slice(int(range_lo[order]), len(RANGE_MAX_LIST)),
                        slice(0, int(body_pct_hi_short[order]) + 1),
                        slice(0, int(close_pos_hi_short[order]) + 1),
                        slice(int(tail_lo_short[order]), len(UPPER_TAIL_MAX_LIST)),
                        slice(0, min_hi + 1),
                        slice(max_lo, len(OPEN_GAP_MAX_LIST)),
                        slice(None),
                    )
                    raw_mask = legal[trigger_slc]
                    active = raw_mask & (last_entry_order[trigger_slc] < order - 1)
                    open_class = _open_class_short(float(o0[order]), float(l1[order]))
                    raw_direct_point, direct_point = _costed_direct_open_points(
                        -1, float(o0[order]), float(o_next[order])
                    )

                    _add_count(raw_trigger_count, trigger_slc, raw_mask)
                    _add_count(eligible_trigger_count, trigger_slc, active)
                    _add_count(short_trigger_count, trigger_slc, active)
                    _add_count(direct_all_count, trigger_slc, active)
                    _add_sum(direct_all_sum, trigger_slc, active, direct_point)
                    _add_count(class_eligible_count[open_class], trigger_slc, active)
                    _add_count(class_direct_all_count[open_class], trigger_slc, active)
                    _add_sum(class_direct_all_sum[open_class], trigger_slc, active, direct_point)

                    pen_hi = int(pen_hi_short[order])
                    if pen_hi >= 0:
                        fill_slc = (*trigger_slc[:-1], slice(0, pen_hi + 1))
                        fill_active = legal[fill_slc] & (last_entry_order[fill_slc] < order - 1)
                        raw_point, point, _net_profit, fee_twd, tax_twd = _costed_pullback_points(
                            -1, float(c1[order]), float(o_next[order])
                        )
                        point_bin = int(round(raw_point - point_min))
                        _add_count(fill_count, fill_slc, fill_active)
                        _add_count(direct_filled_count, fill_slc, fill_active)
                        _add_sum(direct_filled_sum, fill_slc, fill_active, direct_point)
                        _add_count(class_fill_count[open_class], fill_slc, fill_active)
                        point_hist[point_bin][fill_slc][fill_active] += 1
                        stats.update(
                            fill_slc,
                            fill_active,
                            point=point,
                            raw_point=raw_point,
                            fee_twd=fee_twd,
                            tax_twd=tax_twd,
                            slippage_twd=EXIT_SLIPPAGE_POINTS * POINT_VALUE_TWD,
                            year_index=int(years_arr[order]),
                            open_class_index=open_class,
                            side=-1,
                        )
                        last_entry_order[fill_slc][fill_active] = order
                    if pen_hi < len(PENETRATE_LIST) - 1:
                        unfilled_slc = (*trigger_slc[:-1], slice(max(pen_hi + 1, 0), len(PENETRATE_LIST)))
                        unfilled_active = legal[unfilled_slc] & (last_entry_order[unfilled_slc] < order - 1)
                        _add_count(direct_unfilled_count, unfilled_slc, unfilled_active)
                        _add_sum(direct_unfilled_sum, unfilled_slc, unfilled_active, direct_point)
                        _add_count(class_direct_unfilled_count[open_class], unfilled_slc, unfilled_active)
                        _add_sum(class_direct_unfilled_sum[open_class], unfilled_slc, unfilled_active, direct_point)

        if progress_every > 0 and (order + 1) % progress_every == 0:
            print(f"S01 scan progress: {order + 1:,}/{total:,} samples")
    print(f"S01 scan progress: {total:,}/{total:,} samples")

    idx = coords
    total_trades = stats.trade_count[idx]
    gross_profit = stats.gross_profit[idx]
    gross_loss_abs = stats.gross_loss_abs[idx]
    net_points = stats.sum_points[idx]
    eligible_triggers = eligible_trigger_count[idx]

    hist_legal = point_hist[(slice(None), *idx)]
    cum_hist = np.cumsum(hist_legal, axis=0, dtype=np.uint32)
    lower_rank = (total_trades + 1) // 2
    upper_rank = (total_trades + 2) // 2
    has_trades = total_trades > 0
    lower_bins = np.zeros(len(total_trades), dtype=np.int32)
    upper_bins = np.zeros(len(total_trades), dtype=np.int32)
    if has_trades.any():
        lower_bins[has_trades] = (cum_hist[:, has_trades] >= lower_rank[has_trades][None, :]).argmax(axis=0)
        upper_bins[has_trades] = (cum_hist[:, has_trades] >= upper_rank[has_trades][None, :]).argmax(axis=0)
    raw_median_points = np.full(len(total_trades), np.nan, dtype=float)
    raw_median_points[has_trades] = (
        point_values[lower_bins[has_trades]] + point_values[upper_bins[has_trades]]
    ) / 2
    del hist_legal, cum_hist, point_hist

    raw_net_points = stats.raw_sum_points[idx]
    total_fee_twd = stats.total_fee_twd[idx]
    total_tax_twd = stats.total_tax_twd[idx]
    total_slippage_twd = stats.total_slippage_twd[idx]
    total_cost_points = raw_net_points - net_points
    avg_cost_points = _safe_divide(total_cost_points, total_trades)
    median_points = raw_median_points - avg_cost_points
    net_profit_twd = net_points * POINT_VALUE_TWD
    avg_points = _safe_divide(net_points, total_trades)
    raw_avg_points = _safe_divide(raw_net_points, total_trades)
    variance = _safe_divide(stats.sum_sq_points[idx], total_trades) - avg_points * avg_points
    variance = np.where((total_trades > 0) & (variance < 0) & (variance > -1e-12), 0, variance)
    std_points = np.where(total_trades > 0, np.sqrt(variance), np.nan)

    year_points = np.vstack([stats.year_sum_points[year_i][idx] for year_i in range(len(years))])
    year_counts = np.vstack([stats.year_trade_count[year_i][idx] for year_i in range(len(years))])
    active_years = year_counts > 0
    positive_years = ((year_points > 0) & active_years).sum(axis=0)
    negative_years = ((year_points < 0) & active_years).sum(axis=0)
    year_count = active_years.sum(axis=0)
    has_active_year = active_years.any(axis=0)
    worst_year_points = np.where(has_active_year, np.where(active_years, year_points, np.inf).min(axis=0), np.nan)
    best_year_points = np.where(has_active_year, np.where(active_years, year_points, -np.inf).max(axis=0), np.nan)
    worst_year_index = np.where(has_active_year, np.where(active_years, year_points, np.inf).argmin(axis=0), -1)
    best_year_index = np.where(has_active_year, np.where(active_years, year_points, -np.inf).argmax(axis=0), -1)
    year_labels = np.asarray(years, dtype=object)
    worst_year = np.where(worst_year_index >= 0, year_labels[np.maximum(worst_year_index, 0)], "")
    best_year = np.where(best_year_index >= 0, year_labels[np.maximum(best_year_index, 0)], "")

    direct_ev = _safe_divide(direct_all_sum[idx], eligible_triggers)
    pullback_ev = _safe_divide(net_points, eligible_triggers)

    summary = params.copy()
    summary["CostPoints"] = avg_cost_points
    summary["RawTriggerCount"] = raw_trigger_count[idx]
    summary["EligibleTriggerCount"] = eligible_triggers
    summary["FillCount"] = fill_count[idx]
    summary["FillRate"] = _safe_divide(fill_count[idx], eligible_triggers)
    summary["LongTriggers"] = long_trigger_count[idx]
    summary["ShortTriggers"] = short_trigger_count[idx]
    summary["LongTrades"] = stats.long_count[idx]
    summary["ShortTrades"] = stats.short_count[idx]
    summary["TotalTrades"] = total_trades
    summary["WinTrades"] = stats.win_count[idx]
    summary["LossTrades"] = stats.loss_count[idx]
    summary["FlatTrades"] = stats.flat_count[idx]
    summary["WinRate"] = _safe_divide(stats.win_count[idx], total_trades)
    summary["RawNetPoints"] = raw_net_points
    summary["RawAvgPoints"] = raw_avg_points
    summary["NetPoints"] = net_points
    summary["NetProfitTWD"] = net_profit_twd
    summary["TotalReturnRate"] = net_profit_twd / CAPITAL_TWD
    summary["TotalCostPoints"] = total_cost_points
    summary["TotalFeeTWD"] = total_fee_twd
    summary["TotalTaxTWD"] = total_tax_twd
    summary["TotalSlippageTWD"] = total_slippage_twd
    summary["GrossProfit"] = gross_profit
    summary["GrossLoss"] = -gross_loss_abs
    summary["PF"] = _profit_factor(gross_profit, gross_loss_abs)
    summary["AvgPoints"] = avg_points
    summary["RawMedianPoints"] = raw_median_points
    summary["MedianPoints"] = median_points
    summary["MaxTradePoints"] = np.where(total_trades > 0, stats.max_trade[idx], np.nan)
    summary["MinTradePoints"] = np.where(total_trades > 0, stats.min_trade[idx], np.nan)
    summary["StdPoints"] = std_points
    summary["MaxDrawdownPoints"] = np.where(total_trades > 0, stats.max_drawdown[idx], np.nan)
    summary["MaxLosingStreak"] = stats.max_losing_streak[idx]
    summary["PullbackEVPerTrigger"] = pullback_ev
    summary["DirectOpenNet_AllTriggers"] = direct_all_sum[idx]
    summary["DirectOpenAvg_AllTriggers"] = _safe_divide(direct_all_sum[idx], direct_all_count[idx])
    summary["DirectEVPerTrigger"] = direct_ev
    summary["DirectOpenNet_FilledSubset"] = direct_filled_sum[idx]
    summary["DirectOpenAvg_FilledSubset"] = _safe_divide(direct_filled_sum[idx], direct_filled_count[idx])
    summary["DirectOpenNet_UnfilledSubset"] = direct_unfilled_sum[idx]
    summary["DirectOpenAvg_UnfilledSubset"] = _safe_divide(direct_unfilled_sum[idx], direct_unfilled_count[idx])
    summary["PullbackAdvantage"] = pullback_ev - direct_ev
    summary["YearCount"] = year_count
    summary["PositiveYears"] = positive_years
    summary["NegativeYears"] = negative_years
    summary["WorstYear"] = worst_year
    summary["WorstYearPoints"] = worst_year_points
    summary["BestYear"] = best_year
    summary["BestYearPoints"] = best_year_points

    summary_columns = [
        "RuleID",
        "StrategyID",
        "BodyMin",
        "RangeMax",
        "BodyPctMin",
        "ClosePosMin",
        "UpperTailMax",
        "EffOppTailMax",
        "OpenGapMin",
        "OpenGapMax",
        "Penetrate",
        "CostPoints",
        "RawTriggerCount",
        "EligibleTriggerCount",
        "FillCount",
        "FillRate",
        "LongTriggers",
        "ShortTriggers",
        "LongTrades",
        "ShortTrades",
        "TotalTrades",
        "WinTrades",
        "LossTrades",
        "FlatTrades",
        "WinRate",
        "RawNetPoints",
        "RawAvgPoints",
        "NetPoints",
        "NetProfitTWD",
        "TotalReturnRate",
        "TotalCostPoints",
        "TotalFeeTWD",
        "TotalTaxTWD",
        "TotalSlippageTWD",
        "GrossProfit",
        "GrossLoss",
        "PF",
        "AvgPoints",
        "RawMedianPoints",
        "MedianPoints",
        "MaxTradePoints",
        "MinTradePoints",
        "StdPoints",
        "MaxDrawdownPoints",
        "MaxLosingStreak",
        "PullbackEVPerTrigger",
        "DirectOpenNet_AllTriggers",
        "DirectOpenAvg_AllTriggers",
        "DirectEVPerTrigger",
        "DirectOpenNet_FilledSubset",
        "DirectOpenAvg_FilledSubset",
        "DirectOpenNet_UnfilledSubset",
        "DirectOpenAvg_UnfilledSubset",
        "PullbackAdvantage",
        "YearCount",
        "PositiveYears",
        "NegativeYears",
        "WorstYear",
        "WorstYearPoints",
        "BestYear",
        "BestYearPoints",
    ]
    summary = summary[summary_columns]

    by_year_parts = []
    for year_i, year in enumerate(years):
        count = stats.year_trade_count[year_i][idx]
        gp = stats.year_gross_profit[year_i][idx]
        gl = stats.year_gross_loss_abs[year_i][idx]
        part = params.copy()
        part["Year"] = year
        part["Trades"] = count
        part["LongTrades"] = stats.year_long_count[year_i][idx]
        part["ShortTrades"] = stats.year_short_count[year_i][idx]
        part["WinRate"] = _safe_divide(stats.year_win_count[year_i][idx], count)
        part["RawNetPoints"] = stats.year_raw_sum_points[year_i][idx]
        part["NetPoints"] = stats.year_sum_points[year_i][idx]
        part["NetProfitTWD"] = part["NetPoints"] * POINT_VALUE_TWD
        part["TotalReturnRate"] = part["NetProfitTWD"] / CAPITAL_TWD
        part["GrossProfit"] = gp
        part["GrossLoss"] = -gl
        part["PF"] = _profit_factor(gp, gl)
        part["AvgPoints"] = _safe_divide(stats.year_sum_points[year_i][idx], count)
        part["MaxDrawdownPoints"] = np.where(count > 0, stats.year_max_drawdown[year_i][idx], np.nan)
        by_year_parts.append(part)
    by_year = pd.concat(by_year_parts, ignore_index=True)
    by_year = by_year[
        [
            "RuleID",
            "StrategyID",
            "Year",
            "BodyMin",
            "RangeMax",
            "BodyPctMin",
            "ClosePosMin",
            "UpperTailMax",
            "EffOppTailMax",
            "OpenGapMin",
            "OpenGapMax",
            "Penetrate",
            "Trades",
            "LongTrades",
            "ShortTrades",
            "WinRate",
            "RawNetPoints",
            "NetPoints",
            "NetProfitTWD",
            "TotalReturnRate",
            "GrossProfit",
            "GrossLoss",
            "PF",
            "AvgPoints",
            "MaxDrawdownPoints",
        ]
    ]

    by_class_parts = []
    for class_i, class_name in enumerate(OPEN_CLASS_NAMES):
        count = stats.class_trade_count[class_i][idx]
        gp = stats.class_gross_profit[class_i][idx]
        gl = stats.class_gross_loss_abs[class_i][idx]
        eligible = class_eligible_count[class_i][idx]
        part = params.copy()
        part["OpenClass"] = class_name
        part["EligibleTriggerCount"] = eligible
        part["FillCount"] = class_fill_count[class_i][idx]
        part["FillRate"] = _safe_divide(part["FillCount"].to_numpy(), eligible)
        part["Trades"] = count
        part["WinRate"] = _safe_divide(stats.class_win_count[class_i][idx], count)
        part["RawNetPoints"] = stats.class_raw_sum_points[class_i][idx]
        part["NetPoints"] = stats.class_sum_points[class_i][idx]
        part["NetProfitTWD"] = part["NetPoints"] * POINT_VALUE_TWD
        part["TotalReturnRate"] = part["NetProfitTWD"] / CAPITAL_TWD
        part["AvgPoints"] = _safe_divide(stats.class_sum_points[class_i][idx], count)
        part["PF"] = _profit_factor(gp, gl)
        part["DirectOpenNet_AllTriggers"] = class_direct_all_sum[class_i][idx]
        part["DirectOpenAvg_AllTriggers"] = _safe_divide(
            class_direct_all_sum[class_i][idx], class_direct_all_count[class_i][idx]
        )
        part["DirectOpenNet_UnfilledSubset"] = class_direct_unfilled_sum[class_i][idx]
        part["DirectOpenAvg_UnfilledSubset"] = _safe_divide(
            class_direct_unfilled_sum[class_i][idx], class_direct_unfilled_count[class_i][idx]
        )
        by_class_parts.append(part)
    by_open_class = pd.concat(by_class_parts, ignore_index=True)
    by_open_class = by_open_class[
        [
            "RuleID",
            "StrategyID",
            "OpenClass",
            "BodyMin",
            "RangeMax",
            "BodyPctMin",
            "ClosePosMin",
            "UpperTailMax",
            "EffOppTailMax",
            "OpenGapMin",
            "OpenGapMax",
            "Penetrate",
            "EligibleTriggerCount",
            "FillCount",
            "FillRate",
            "Trades",
            "WinRate",
            "RawNetPoints",
            "NetPoints",
            "NetProfitTWD",
            "TotalReturnRate",
            "AvgPoints",
            "PF",
            "DirectOpenNet_AllTriggers",
            "DirectOpenAvg_AllTriggers",
            "DirectOpenNet_UnfilledSubset",
            "DirectOpenAvg_UnfilledSubset",
        ]
    ]

    return S01Result(summary=summary, by_year=by_year, by_open_class=by_open_class, params=params)


def _row_matches(row: pd.Series, sample: object) -> tuple[bool, bool, str | None]:
    range1 = sample.H1 - sample.L1
    if range1 <= 0 or range1 > row.RangeMax:
        return False, False, None
    if sample.C1 > sample.O1:
        body = sample.C1 - sample.O1
        body_pct = body / range1 * 100
        tail_pct = (sample.H1 - sample.C1) / range1 * 100
        close_pos = (sample.C1 - sample.L1) / range1 * 100
        trigger = (
            body >= row.BodyMin
            and body_pct >= row.BodyPctMin
            and close_pos >= row.ClosePosMin
            and tail_pct <= row.UpperTailMax
            and sample.O0 >= sample.C1 + row.OpenGapMin
            and sample.O0 <= sample.C1 + row.OpenGapMax
        )
        fill = trigger and sample.L0 <= sample.C1 - row.Penetrate
        return trigger, fill, "LONG"
    if sample.C1 < sample.O1:
        body = sample.O1 - sample.C1
        body_pct = body / range1 * 100
        tail_pct = (sample.C1 - sample.L1) / range1 * 100
        close_pos = (sample.C1 - sample.L1) / range1 * 100
        trigger = (
            body >= row.BodyMin
            and body_pct >= row.BodyPctMin
            and close_pos <= 100 - row.ClosePosMin
            and tail_pct <= row.UpperTailMax
            and sample.O0 <= sample.C1 - row.OpenGapMin
            and sample.O0 >= sample.C1 - row.OpenGapMax
        )
        fill = trigger and sample.H0 >= sample.C1 + row.Penetrate
        return trigger, fill, "SHORT"
    return False, False, None


def materialize_trades(samples: pd.DataFrame, rule: pd.Series) -> pd.DataFrame:
    rows = []
    last_entry_order = -10_000_000
    for order, sample in enumerate(samples.itertuples(index=False)):
        if last_entry_order >= order - 1:
            continue
        trigger, fill, side = _row_matches(rule, sample)
        if not trigger or not fill or side is None:
            continue

        entry_dt = pd.Timestamp(sample.datetime)
        exit_dt = pd.Timestamp(sample.NEXT_DATETIME)
        range1 = sample.H1 - sample.L1
        if side == "LONG":
            body = sample.C1 - sample.O1
            opp_tail_pct = (sample.H1 - sample.C1) / range1 * 100
            raw_points, points, net_profit_twd, fee_twd, tax_twd = _costed_pullback_points(
                1, float(sample.C1), float(sample.O_NEXT)
            )
            raw_direct_open_points, direct_open_points = _costed_direct_open_points(
                1, float(sample.O0), float(sample.O_NEXT)
            )
            effective_exit = sample.O_NEXT - EXIT_SLIPPAGE_POINTS
            open_gap = sample.O0 - sample.C1
            open_class = OPEN_CLASS_NAMES[_open_class_long(sample.O0, sample.H1)]
        else:
            body = sample.O1 - sample.C1
            opp_tail_pct = (sample.C1 - sample.L1) / range1 * 100
            raw_points, points, net_profit_twd, fee_twd, tax_twd = _costed_pullback_points(
                -1, float(sample.C1), float(sample.O_NEXT)
            )
            raw_direct_open_points, direct_open_points = _costed_direct_open_points(
                -1, float(sample.O0), float(sample.O_NEXT)
            )
            effective_exit = sample.O_NEXT + EXIT_SLIPPAGE_POINTS
            open_gap = sample.C1 - sample.O0
            open_class = OPEN_CLASS_NAMES[_open_class_short(sample.O0, sample.L1)]
        slippage_twd = EXIT_SLIPPAGE_POINTS * POINT_VALUE_TWD

        rows.append(
            {
                "StrategyID": STRATEGY_ID,
                "EntryIndex": order,
                "EntryDate": entry_dt.strftime("%Y-%m-%d"),
                "EntryTime": entry_dt.strftime("%H:%M:%S"),
                "ExitIndex": order + 1,
                "ExitDate": exit_dt.strftime("%Y-%m-%d"),
                "ExitTime": exit_dt.strftime("%H:%M:%S"),
                "ExitReason": "NEXT_OPEN",
                "Side": side,
                "EntryPx": sample.C1,
                "RawExitPx": sample.O_NEXT,
                "ExitPx": effective_exit,
                "RawPoints": raw_points,
                "Points": points,
                "NetProfitTWD": net_profit_twd,
                "FeeTWD": fee_twd,
                "TaxTWD": tax_twd,
                "SlippageTWD": slippage_twd,
                "CostPoints": raw_points - points,
                "O1": sample.O1,
                "H1": sample.H1,
                "L1": sample.L1,
                "C1": sample.C1,
                "O0": sample.O0,
                "H0": sample.H0,
                "L0": sample.L0,
                "NextOpen": sample.O_NEXT,
                "Range1": range1,
                "Body1": body,
                "BodyPct1": body / range1 * 100 if range1 > 0 else np.nan,
                "ClosePosPct1": (sample.C1 - sample.L1) / range1 * 100 if range1 > 0 else np.nan,
                "OppTailPct1": opp_tail_pct,
                "OpenGap": open_gap,
                "Penetrate": rule.Penetrate,
                "OpenClass": open_class,
                "BodyMin": rule.BodyMin,
                "RangeMax": rule.RangeMax,
                "BodyPctMin": rule.BodyPctMin,
                "ClosePosMin": rule.ClosePosMin,
                "UpperTailMax": rule.UpperTailMax,
                "EffOppTailMax": rule.EffOppTailMax,
                "OpenGapMin": rule.OpenGapMin,
                "OpenGapMax": rule.OpenGapMax,
                "DirectOpenRawPoints": raw_direct_open_points,
                "DirectOpenPoints": direct_open_points,
            }
        )
        last_entry_order = order
    return pd.DataFrame(rows)


def _validate_result(summary: pd.DataFrame) -> list[str]:
    checks: list[str] = []
    if len(summary) != EXPECTED_PARAM_COUNT:
        raise RuntimeError(f"summary rows {len(summary):,} != {EXPECTED_PARAM_COUNT:,}")
    checks.append(f"parameter_count_ok={len(summary):,}")

    if (summary["TotalTrades"] > summary["EligibleTriggerCount"]).any():
        raise RuntimeError("TotalTrades > EligibleTriggerCount found")
    checks.append("total_trades_le_eligible_trigger_ok=1")

    if not (summary["FillCount"].to_numpy() == summary["TotalTrades"].to_numpy()).all():
        raise RuntimeError("FillCount != TotalTrades found")
    checks.append("fill_count_eq_total_trades_ok=1")

    expected_fill_rate = _safe_divide(
        summary["FillCount"].to_numpy(dtype=float),
        summary["EligibleTriggerCount"].to_numpy(dtype=float),
    )
    actual_fill_rate = summary["FillRate"].to_numpy(dtype=float)
    if not np.allclose(np.nan_to_num(expected_fill_rate), np.nan_to_num(actual_fill_rate)):
        raise RuntimeError("FillRate mismatch found")
    checks.append("fill_rate_ok=1")
    return checks


def _write_outputs(result: S01Result, samples: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    summary = result.summary
    summary.to_csv(output_dir / "summary_all.csv", index=False, encoding="utf-8-sig")

    top_net = summary.sort_values("NetPoints", ascending=False, na_position="last").head(200)
    top_net.to_csv(output_dir / "top_net.csv", index=False, encoding="utf-8-sig")

    tradable = summary[summary["TotalTrades"] >= 300]
    top_pf = tradable.sort_values("PF", ascending=False, na_position="last").head(200)
    top_pf.to_csv(output_dir / "top_pf.csv", index=False, encoding="utf-8-sig")

    top_avg = tradable.sort_values("AvgPoints", ascending=False, na_position="last").head(200)
    top_avg.to_csv(output_dir / "top_avg.csv", index=False, encoding="utf-8-sig")

    top_adv = tradable.sort_values("PullbackAdvantage", ascending=False, na_position="last").head(200)
    top_adv.to_csv(output_dir / "top_pullback_advantage.csv", index=False, encoding="utf-8-sig")

    robust = tradable[
        (tradable["PF"] > 1.05)
        & (tradable["AvgPoints"] > 0)
        & (tradable["PositiveYears"] >= 4)
    ].copy()
    robust = robust.sort_values(
        ["PF", "AvgPoints", "NetPoints"],
        ascending=[False, False, False],
        na_position="last",
    )
    top_robust = robust.head(200)
    top_robust.to_csv(output_dir / "top_robust.csv", index=False, encoding="utf-8-sig")

    result.by_year.to_csv(output_dir / "by_year.csv", index=False, encoding="utf-8-sig")
    result.by_open_class.to_csv(output_dir / "by_open_class.csv", index=False, encoding="utf-8-sig")

    trade_dir = output_dir / "top_20_trades"
    trade_dir.mkdir(parents=True, exist_ok=True)
    top_source = robust if len(robust) >= 20 else top_pf
    if len(top_source) < 20:
        top_source = top_net
    for old_file in trade_dir.glob("*.csv"):
        old_file.unlink()
    for rank, row in enumerate(top_source.head(20).itertuples(index=False), start=1):
        rule = pd.Series(row._asdict())
        trades = materialize_trades(samples, rule)
        file_name = f"S01_ATTACK_C1_PULLBACK_rank_{rank:02d}_rule_{int(rule.RuleID):06d}_trades.csv"
        trades.to_csv(trade_dir / file_name, index=False, encoding="utf-8-sig")

    return {
        "tradable_count": int(len(tradable)),
        "pf_gt_105_tradable_count": int(((tradable["PF"] > 1.05)).sum()),
        "top_robust_count": int(len(robust)),
        "pullback_advantage_gt_0_tradable_count": int((tradable["PullbackAdvantage"] > 0).sum()),
        "direct_unfilled_nonzero_count": int((summary["DirectOpenNet_UnfilledSubset"] != 0).sum()),
    }


def run(data_path: Path, output_dir: Path, *, progress_every: int = 10000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df, data_report = load_price_data(data_path)
    samples, sample_report = build_samples(
        df,
        begin_time="09:05",
        end_time="13:10",
        exit_time_limit="13:12",
    )
    result = scan(samples, progress_every=progress_every)
    validation_lines = _validate_result(result.summary)
    output_counts = _write_outputs(result, samples, output_dir)

    robust_preview = (
        result.summary[
            (result.summary["TotalTrades"] >= 300)
            & (result.summary["PF"] > 1.05)
            & (result.summary["AvgPoints"] > 0)
            & (result.summary["PositiveYears"] >= 4)
        ]
        .sort_values(["PF", "AvgPoints", "NetPoints"], ascending=[False, False, False], na_position="last")
        .head(10)
    )
    preview_lines = []
    for row in robust_preview.itertuples(index=False):
        preview_lines.append(
            "top_robust "
            f"RuleID={int(row.RuleID)} "
            f"BM={row.BodyMin} RM={row.RangeMax} BP={row.BodyPctMin} "
            f"CP={row.ClosePosMin} UT={row.UpperTailMax} "
            f"OG={row.OpenGapMin}-{row.OpenGapMax} P={row.Penetrate} "
            f"Trades={row.TotalTrades} PF={row.PF:.4f} Avg={row.AvgPoints:.4f} Net={row.NetPoints:.0f}"
        )

    log_lines = [
        "S01_ATTACK_C1_PULLBACK run log",
        f"data_file={data_path}",
        f"output_dir={output_dir}",
        f"raw_rows={data_report.raw_rows:,}",
        f"cleaned_rows={data_report.cleaned_rows:,}",
        f"duplicate_datetime_rows={data_report.duplicate_datetime_rows:,}",
        f"invalid_ohlc_rows={data_report.invalid_ohlc_rows:,}",
        f"missing_ohlc_rows={data_report.missing_ohlc_rows:,}",
        f"research_samples={sample_report.sample_count:,}",
        f"sample_date_min={sample_report.date_min}",
        f"sample_date_max={sample_report.date_max}",
        f"expected_param_count={EXPECTED_PARAM_COUNT:,}",
        f"actual_param_count={len(result.summary):,}",
        f"summary_all_rows={len(result.summary):,}",
        f"by_year_rows={len(result.by_year):,}",
        f"by_open_class_rows={len(result.by_open_class):,}",
        f"TotalTrades>=300_count={output_counts['tradable_count']:,}",
        f"PF>1.05_and_TotalTrades>=300_count={output_counts['pf_gt_105_tradable_count']:,}",
        f"top_robust_count={output_counts['top_robust_count']:,}",
        f"PullbackAdvantage>0_and_TotalTrades>=300_count={output_counts['pullback_advantage_gt_0_tradable_count']:,}",
        f"DirectOpen_UnfilledSubset_nonzero_count={output_counts['direct_unfilled_nonzero_count']:,}",
        "signal_fields=O1,H1,L1,C1,O0 only",
        "fill_fields=L0 for long, H0 for short only",
        "k0_close_used_for_signal=0",
        "entry_price=C1",
        f"entry_slippage_points={ENTRY_SLIPPAGE_POINTS}",
        f"exit_slippage_points={EXIT_SLIPPAGE_POINTS}",
        "exit=next open after exit slippage",
        f"point_value_twd={POINT_VALUE_TWD}",
        f"capital_twd={CAPITAL_TWD}",
        f"fee_per_side_twd={FEE_PER_SIDE_TWD}",
        f"round_turn_fee_twd={ROUND_TURN_FEE_TWD}",
        f"transaction_tax_rate={TRANSACTION_TAX_RATE}",
        "same_bar_reversal=0",
        "exit_bar_reentry=0",
        "cost=deducted: exit slippage, round-turn fee, transaction tax",
        f"total_return_rate=NetProfitTWD/{CAPITAL_TWD}",
        "median_points=raw median minus average cost points",
        *validation_lines,
        *preview_lines,
    ]
    log_text = "\n".join(log_lines) + "\n"
    (output_dir / "run_log.txt").write_text(log_text, encoding="utf-8")
    (output_dir / "run_report.txt").write_text(log_text, encoding="utf-8")
    write_s01_html_report(
        outdir=output_dir,
        output_html=output_dir / "s01_attack_c1_pullback_report.html",
        root_copy=Path("rod_yearly_report.html"),
    )
