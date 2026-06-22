from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as cfg
from .config import ResearchConfig
from .metrics import costed_points, profit_factor, safe_divide
from .param_grid import (
    ALL_FAMILY_PARAM_COLUMNS,
    FamilySpec,
    assert_expected_r2a_count,
    exec_param_columns,
    family_legal_mask,
    family_param_columns,
    family_specs,
    open_gap_legal_mask,
)


SIDE_LONG = 1
SIDE_SHORT = -1


def _hi(value: float, values: list[int]) -> int:
    return int(np.searchsorted(np.asarray(values, dtype=float), value, side="right") - 1)


def _lo(value: float, values: list[int]) -> int:
    return int(np.searchsorted(np.asarray(values, dtype=float), value, side="left"))


def _gap_bounds(gap: float) -> tuple[slice, slice] | None:
    min_hi = _hi(gap, cfg.OPEN_GAP_MIN_LIST)
    max_lo = _lo(gap, cfg.OPEN_GAP_MAX_LIST)
    if min_hi < 0 or max_lo >= len(cfg.OPEN_GAP_MAX_LIST):
        return None
    return slice(0, min_hi + 1), slice(max_lo, len(cfg.OPEN_GAP_MAX_LIST))


def _side_name(side: int) -> str:
    return "LONG" if side == SIDE_LONG else "SHORT"


def _time_segment_index(segment: str) -> int:
    return cfg.TIME_SEGMENTS.index(segment)


def _open_class_index(name: str) -> int:
    return cfg.OPEN_CLASSES.index(name)


def _add_count(arr: np.ndarray, slc: tuple[slice, ...], mask: np.ndarray) -> None:
    if mask.any():
        arr[slc][mask] += 1


def _add_sum(arr: np.ndarray, slc: tuple[slice, ...], mask: np.ndarray, value: float) -> None:
    if mask.any():
        arr[slc][mask] += value


@dataclass
class DenseStats:
    shape: tuple[int, ...]
    years: tuple[int, ...]

    def __post_init__(self) -> None:
        self.raw_trigger = np.zeros(self.shape, dtype=np.int32)
        self.eligible_trigger = np.zeros(self.shape, dtype=np.int32)
        self.fill_count = np.zeros(self.shape, dtype=np.int32)
        self.long_trigger = np.zeros(self.shape, dtype=np.int32)
        self.short_trigger = np.zeros(self.shape, dtype=np.int32)
        self.long_trades = np.zeros(self.shape, dtype=np.int32)
        self.short_trades = np.zeros(self.shape, dtype=np.int32)
        self.trade_count = np.zeros(self.shape, dtype=np.int32)
        self.win_count = np.zeros(self.shape, dtype=np.int32)
        self.loss_count = np.zeros(self.shape, dtype=np.int32)
        self.flat_count = np.zeros(self.shape, dtype=np.int32)

        self.raw_sum = np.zeros(self.shape, dtype=np.float64)
        self.raw_gross_profit = np.zeros(self.shape, dtype=np.float64)
        self.raw_gross_loss_abs = np.zeros(self.shape, dtype=np.float64)
        self.net_sum = np.zeros(self.shape, dtype=np.float64)
        self.net_sq_sum = np.zeros(self.shape, dtype=np.float64)
        self.gross_profit = np.zeros(self.shape, dtype=np.float64)
        self.gross_loss_abs = np.zeros(self.shape, dtype=np.float64)
        self.max_trade = np.full(self.shape, -np.inf, dtype=np.float64)
        self.min_trade = np.full(self.shape, np.inf, dtype=np.float64)

        self.total_fee_twd = np.zeros(self.shape, dtype=np.float64)
        self.total_tax_twd = np.zeros(self.shape, dtype=np.float64)
        self.total_slippage_twd = np.zeros(self.shape, dtype=np.float64)

        self.equity = np.zeros(self.shape, dtype=np.float64)
        self.peak = np.zeros(self.shape, dtype=np.float64)
        self.max_drawdown = np.zeros(self.shape, dtype=np.float64)
        self.current_losing_streak = np.zeros(self.shape, dtype=np.int32)
        self.max_losing_streak = np.zeros(self.shape, dtype=np.int32)

        self.direct_raw_all_sum = np.zeros(self.shape, dtype=np.float64)
        self.direct_net_all_sum = np.zeros(self.shape, dtype=np.float64)
        self.direct_all_count = np.zeros(self.shape, dtype=np.int32)
        self.direct_raw_filled_sum = np.zeros(self.shape, dtype=np.float64)
        self.direct_net_filled_sum = np.zeros(self.shape, dtype=np.float64)
        self.direct_filled_count = np.zeros(self.shape, dtype=np.int32)
        self.direct_raw_unfilled_sum = np.zeros(self.shape, dtype=np.float64)
        self.direct_net_unfilled_sum = np.zeros(self.shape, dtype=np.float64)
        self.direct_unfilled_count = np.zeros(self.shape, dtype=np.int32)

        year_shape = (len(self.years), *self.shape)
        self.year_count = np.zeros(year_shape, dtype=np.int32)
        self.year_win = np.zeros(year_shape, dtype=np.int32)
        self.year_net_sum = np.zeros(year_shape, dtype=np.float64)
        self.year_gp = np.zeros(year_shape, dtype=np.float64)
        self.year_gl_abs = np.zeros(year_shape, dtype=np.float64)
        self.year_equity = np.zeros(year_shape, dtype=np.float64)
        self.year_peak = np.zeros(year_shape, dtype=np.float64)
        self.year_mdd = np.zeros(year_shape, dtype=np.float64)

        side_shape = (2, *self.shape)
        self.side_count = np.zeros(side_shape, dtype=np.int32)
        self.side_win = np.zeros(side_shape, dtype=np.int32)
        self.side_net_sum = np.zeros(side_shape, dtype=np.float64)
        self.side_gp = np.zeros(side_shape, dtype=np.float64)
        self.side_gl_abs = np.zeros(side_shape, dtype=np.float64)

        class_shape = (len(cfg.OPEN_CLASSES), *self.shape)
        self.class_eligible = np.zeros(class_shape, dtype=np.int32)
        self.class_fill = np.zeros(class_shape, dtype=np.int32)
        self.class_count = np.zeros(class_shape, dtype=np.int32)
        self.class_win = np.zeros(class_shape, dtype=np.int32)
        self.class_net_sum = np.zeros(class_shape, dtype=np.float64)
        self.class_gp = np.zeros(class_shape, dtype=np.float64)
        self.class_gl_abs = np.zeros(class_shape, dtype=np.float64)
        self.class_direct_net_all_sum = np.zeros(class_shape, dtype=np.float64)
        self.class_direct_all_count = np.zeros(class_shape, dtype=np.int32)
        self.class_direct_net_unfilled_sum = np.zeros(class_shape, dtype=np.float64)
        self.class_direct_unfilled_count = np.zeros(class_shape, dtype=np.int32)

        segment_shape = (len(cfg.TIME_SEGMENTS), *self.shape)
        self.segment_count = np.zeros(segment_shape, dtype=np.int32)
        self.segment_win = np.zeros(segment_shape, dtype=np.int32)
        self.segment_net_sum = np.zeros(segment_shape, dtype=np.float64)
        self.segment_gp = np.zeros(segment_shape, dtype=np.float64)
        self.segment_gl_abs = np.zeros(segment_shape, dtype=np.float64)

    def add_trigger(
        self,
        slc: tuple[slice, ...],
        raw_mask: np.ndarray,
        active: np.ndarray,
        *,
        side: int,
        direct_raw: float,
        direct_net: float,
        open_class: int,
    ) -> None:
        _add_count(self.raw_trigger, slc, raw_mask)
        _add_count(self.eligible_trigger, slc, active)
        if side == SIDE_LONG:
            _add_count(self.long_trigger, slc, active)
        else:
            _add_count(self.short_trigger, slc, active)
        _add_count(self.direct_all_count, slc, active)
        _add_sum(self.direct_raw_all_sum, slc, active, direct_raw)
        _add_sum(self.direct_net_all_sum, slc, active, direct_net)
        _add_count(self.class_eligible[open_class], slc, active)
        _add_count(self.class_direct_all_count[open_class], slc, active)
        _add_sum(self.class_direct_net_all_sum[open_class], slc, active, direct_net)

    def add_unfilled(
        self,
        slc: tuple[slice, ...],
        active: np.ndarray,
        *,
        direct_raw: float,
        direct_net: float,
        open_class: int,
    ) -> None:
        _add_count(self.direct_unfilled_count, slc, active)
        _add_sum(self.direct_raw_unfilled_sum, slc, active, direct_raw)
        _add_sum(self.direct_net_unfilled_sum, slc, active, direct_net)
        _add_count(self.class_direct_unfilled_count[open_class], slc, active)
        _add_sum(self.class_direct_net_unfilled_sum[open_class], slc, active, direct_net)

    def add_trade(
        self,
        slc: tuple[slice, ...],
        active: np.ndarray,
        *,
        side: int,
        raw_points: float,
        net_points: float,
        fee_twd: int,
        tax_twd: int,
        slippage_twd: float,
        year_index: int,
        open_class: int,
        time_segment: int,
        direct_raw: float,
        direct_net: float,
    ) -> None:
        if not active.any():
            return
        _add_count(self.fill_count, slc, active)
        _add_count(self.direct_filled_count, slc, active)
        _add_sum(self.direct_raw_filled_sum, slc, active, direct_raw)
        _add_sum(self.direct_net_filled_sum, slc, active, direct_net)
        _add_count(self.class_fill[open_class], slc, active)

        self.trade_count[slc][active] += 1
        self.raw_sum[slc][active] += raw_points
        self.net_sum[slc][active] += net_points
        self.net_sq_sum[slc][active] += net_points * net_points
        self.total_fee_twd[slc][active] += fee_twd
        self.total_tax_twd[slc][active] += tax_twd
        self.total_slippage_twd[slc][active] += slippage_twd
        self.max_trade[slc][active] = np.maximum(self.max_trade[slc][active], net_points)
        self.min_trade[slc][active] = np.minimum(self.min_trade[slc][active], net_points)

        side_index = 0 if side == SIDE_LONG else 1
        self.side_count[side_index][slc][active] += 1
        self.side_net_sum[side_index][slc][active] += net_points
        if side == SIDE_LONG:
            self.long_trades[slc][active] += 1
        else:
            self.short_trades[slc][active] += 1

        self.equity[slc][active] += net_points
        equity_view = self.equity[slc]
        peak_view = self.peak[slc]
        peak_view[active] = np.maximum(peak_view[active], equity_view[active])
        mdd_view = self.max_drawdown[slc]
        mdd_view[active] = np.maximum(mdd_view[active], peak_view[active] - equity_view[active])

        self.year_count[year_index][slc][active] += 1
        self.year_net_sum[year_index][slc][active] += net_points
        self.year_equity[year_index][slc][active] += net_points
        y_eq = self.year_equity[year_index][slc]
        y_peak = self.year_peak[year_index][slc]
        y_peak[active] = np.maximum(y_peak[active], y_eq[active])
        y_mdd = self.year_mdd[year_index][slc]
        y_mdd[active] = np.maximum(y_mdd[active], y_peak[active] - y_eq[active])

        self.class_count[open_class][slc][active] += 1
        self.class_net_sum[open_class][slc][active] += net_points
        self.segment_count[time_segment][slc][active] += 1
        self.segment_net_sum[time_segment][slc][active] += net_points

        if raw_points > 0:
            self.raw_gross_profit[slc][active] += raw_points
        elif raw_points < 0:
            self.raw_gross_loss_abs[slc][active] += -raw_points

        if net_points > 0:
            self.win_count[slc][active] += 1
            self.gross_profit[slc][active] += net_points
            self.year_win[year_index][slc][active] += 1
            self.year_gp[year_index][slc][active] += net_points
            self.side_win[side_index][slc][active] += 1
            self.side_gp[side_index][slc][active] += net_points
            self.class_win[open_class][slc][active] += 1
            self.class_gp[open_class][slc][active] += net_points
            self.segment_win[time_segment][slc][active] += 1
            self.segment_gp[time_segment][slc][active] += net_points
            self.current_losing_streak[slc][active] = 0
        elif net_points < 0:
            loss = -net_points
            self.loss_count[slc][active] += 1
            self.gross_loss_abs[slc][active] += loss
            self.year_gl_abs[year_index][slc][active] += loss
            self.side_gl_abs[side_index][slc][active] += loss
            self.class_gl_abs[open_class][slc][active] += loss
            self.segment_gl_abs[time_segment][slc][active] += loss
            streak = self.current_losing_streak[slc]
            streak[active] += 1
            best = self.max_losing_streak[slc]
            best[active] = np.maximum(best[active], streak[active])
        else:
            self.flat_count[slc][active] += 1
            self.current_losing_streak[slc][active] = 0


def _family_match(spec: FamilySpec, row: object, side: int) -> tuple[tuple[slice, ...], np.ndarray | None] | None:
    range1 = float(row.Range1)
    if not np.isfinite(range1) or range1 <= 0:
        return None

    if side == SIDE_LONG and not bool(row.LongDirection):
        return None
    if side == SIDE_SHORT and not bool(row.ShortDirection):
        return None

    body = float(row.BodyLong if side == SIDE_LONG else row.BodyShort)
    body_pct = float(row.BodyPct)
    close_pos = float(row.ClosePosPct)
    upper_tail_pct = float(row.UpperTailPct)
    lower_tail_pct = float(row.LowerTailPct)
    opp_tail_pct = upper_tail_pct if side == SIDE_LONG else lower_tail_pct
    main_tail_pct = lower_tail_pct if side == SIDE_LONG else upper_tail_pct

    if spec.name == "F01_ATTACK":
        body_hi = _hi(body, cfg.BODY_MIN_LIST)
        range_lo = _lo(range1, cfg.RANGE_MAX_LIST)
        body_pct_hi = _hi(body_pct, cfg.BODY_PCT_MIN_LIST)
        opp_lo = _lo(opp_tail_pct, cfg.EFF_OPP_TAIL_MAX_LIST)
        if body_hi < 0 or range_lo >= len(cfg.RANGE_MAX_LIST) or body_pct_hi < 0 or opp_lo >= len(cfg.EFF_OPP_TAIL_MAX_LIST):
            return None
        return (slice(0, body_hi + 1), slice(range_lo, len(cfg.RANGE_MAX_LIST)), slice(0, body_pct_hi + 1), slice(opp_lo, len(cfg.EFF_OPP_TAIL_MAX_LIST))), None

    if spec.name == "F02_STRONG_CLOSE":
        body_hi = _hi(body, cfg.BODY_MIN_LIST)
        range_lo = _lo(range1, cfg.RANGE_MAX_LIST)
        pct_hi = _hi(body_pct, cfg.BODY_PCT_FLOOR_LIST)
        cp_value = close_pos if side == SIDE_LONG else 100.0 - close_pos
        cp_hi = _hi(cp_value, cfg.CLOSE_POS_STRONG_LIST)
        if body_hi < 0 or range_lo >= len(cfg.RANGE_MAX_LIST) or pct_hi < 0 or cp_hi < 0:
            return None
        return (slice(0, body_hi + 1), slice(range_lo, len(cfg.RANGE_MAX_LIST)), slice(0, pct_hi + 1), slice(0, cp_hi + 1)), None

    if spec.name == "F03_MARUBOZU":
        body_hi = _hi(body, cfg.BODY_MIN_LIST)
        range_lo = _lo(range1, cfg.RANGE_MAX_LIST)
        pct_hi = _hi(body_pct, cfg.MARU_BODY_PCT_LIST)
        tail_lo = max(_lo(upper_tail_pct, cfg.TAIL_MAX_LIST), _lo(lower_tail_pct, cfg.TAIL_MAX_LIST))
        if body_hi < 0 or range_lo >= len(cfg.RANGE_MAX_LIST) or pct_hi < 0 or tail_lo >= len(cfg.TAIL_MAX_LIST):
            return None
        return (slice(0, body_hi + 1), slice(range_lo, len(cfg.RANGE_MAX_LIST)), slice(0, pct_hi + 1), slice(tail_lo, len(cfg.TAIL_MAX_LIST))), None

    if spec.name == "F04_BODY_EFFICIENCY":
        body_hi = _hi(body, cfg.BODY_MIN_LIST)
        range_lo = _lo(range1, cfg.RANGE_MAX_LIST)
        pct_hi = _hi(body_pct, cfg.BODY_PCT_MIN_LIST)
        cp_value = close_pos if side == SIDE_LONG else 100.0 - close_pos
        cp_hi = _hi(cp_value, cfg.CLOSE_POS_EFF_LIST)
        if body_hi < 0 or range_lo >= len(cfg.RANGE_MAX_LIST) or pct_hi < 0 or cp_hi < 0:
            return None
        return (slice(0, body_hi + 1), slice(range_lo, len(cfg.RANGE_MAX_LIST)), slice(0, pct_hi + 1), slice(0, cp_hi + 1)), None

    if spec.name == "F05_TAIL_SUPPORT_CONTINUATION":
        fixed_close_ok = close_pos >= 60 if side == SIDE_LONG else close_pos <= 40
        if not fixed_close_ok:
            return None
        body_hi = _hi(body, cfg.BODY_MIN_LIST)
        range_lo = _lo(range1, cfg.RANGE_MAX_LIST)
        main_hi = _hi(main_tail_pct, cfg.MAIN_TAIL_MIN_LIST)
        opp_lo = _lo(opp_tail_pct, cfg.EFF_OPP_TAIL_MAX_LIST)
        if body_hi < 0 or range_lo >= len(cfg.RANGE_MAX_LIST) or main_hi < 0 or opp_lo >= len(cfg.EFF_OPP_TAIL_MAX_LIST):
            return None
        return (slice(0, body_hi + 1), slice(range_lo, len(cfg.RANGE_MAX_LIST)), slice(0, main_hi + 1), slice(opp_lo, len(cfg.EFF_OPP_TAIL_MAX_LIST))), None

    if spec.name == "F06_BODY_CENTER":
        body_hi = _hi(body, cfg.BODY_MIN_LIST)
        range_lo = _lo(range1, cfg.RANGE_MAX_LIST)
        pct_hi = _hi(body_pct, cfg.BODY_PCT_CENTER_LIST)
        cp_value = close_pos if side == SIDE_LONG else 100.0 - close_pos
        cp_hi = _hi(cp_value, cfg.CLOSE_POS_CENTER_LIST)
        center_value = float(row.BM1 - row.M1) if side == SIDE_LONG else float(row.M1 - row.BM1)
        center_hi = _hi(center_value, cfg.CENTER_OFFSET_LIST)
        if body_hi < 0 or range_lo >= len(cfg.RANGE_MAX_LIST) or pct_hi < 0 or cp_hi < 0 or center_hi < 0:
            return None
        return (slice(0, body_hi + 1), slice(range_lo, len(cfg.RANGE_MAX_LIST)), slice(0, pct_hi + 1), slice(0, cp_hi + 1), slice(0, center_hi + 1)), None

    if spec.name == "F07_LARGE_RANGE_ATTACK":
        pairs = np.asarray(cfg.LARGE_RANGE_PAIR_LIST, dtype=float)
        range_mask = (range1 >= pairs[:, 0]) & (range1 <= pairs[:, 1])
        if not range_mask.any():
            return None
        body_hi = _hi(body, cfg.BODY_MIN_LARGE_LIST)
        pct_hi = _hi(body_pct, cfg.BODY_PCT_MIN_LIST)
        opp_lo = _lo(opp_tail_pct, cfg.EFF_OPP_TAIL_MAX_LIST)
        if body_hi < 0 or pct_hi < 0 or opp_lo >= len(cfg.EFF_OPP_TAIL_MAX_LIST):
            return None
        fam_slc = (slice(None), slice(0, body_hi + 1), slice(0, pct_hi + 1), slice(opp_lo, len(cfg.EFF_OPP_TAIL_MAX_LIST)))
        mask = np.broadcast_to(
            range_mask[:, None, None, None],
            (
                len(cfg.LARGE_RANGE_PAIR_LIST),
                body_hi + 1,
                pct_hi + 1,
                len(cfg.EFF_OPP_TAIL_MAX_LIST) - opp_lo,
            ),
        ).copy()
        return fam_slc, mask

    raise ValueError(f"unknown family: {spec.name}")


def _exec_signal(row: object, mode: str, side: int, break_small_max: int) -> tuple[bool, float, str]:
    c1 = float(row.C1)
    h1 = float(row.H1)
    l1 = float(row.L1)
    o0 = float(row.O0)
    bm1 = float(row.BM1)

    if side == SIDE_LONG:
        if mode == "X01_INSIDE_C1":
            return (c1 < o0 <= h1), c1, "INSIDE"
        if mode == "X02_BREAK_SMALL_C1":
            return (o0 > h1 and o0 <= h1 + break_small_max), c1, "BREAK_SMALL"
        if mode == "X03_BREAK_SMALL_HL":
            return (o0 > h1 and o0 <= h1 + break_small_max), h1, "BREAK_SMALL"
        if mode == "X04_AUTO_C1_HL":
            if c1 < o0 <= h1:
                return True, c1, "INSIDE"
            if h1 < o0 <= h1 + break_small_max:
                return True, h1, "BREAK_SMALL"
            return False, c1, "BREAK_LARGE"
        if mode == "X05_INSIDE_BM1":
            return (c1 < o0 <= h1), bm1, "INSIDE"
    else:
        if mode == "X01_INSIDE_C1":
            return (l1 <= o0 < c1), c1, "INSIDE"
        if mode == "X02_BREAK_SMALL_C1":
            return (o0 < l1 and o0 >= l1 - break_small_max), c1, "BREAK_SMALL"
        if mode == "X03_BREAK_SMALL_HL":
            return (o0 < l1 and o0 >= l1 - break_small_max), l1, "BREAK_SMALL"
        if mode == "X04_AUTO_C1_HL":
            if l1 <= o0 < c1:
                return True, c1, "INSIDE"
            if l1 - break_small_max <= o0 < l1:
                return True, l1, "BREAK_SMALL"
            return False, c1, "BREAK_LARGE"
        if mode == "X05_INSIDE_BM1":
            return (l1 <= o0 < c1), bm1, "INSIDE"
    raise ValueError(f"unknown exec mode: {mode}")


def _make_legal(spec: FamilySpec) -> np.ndarray:
    family_legal = family_legal_mask(spec)
    gap_legal = open_gap_legal_mask()
    return (
        family_legal.reshape(spec.dense_shape + (1, 1, 1, 1))
        & gap_legal.reshape((1,) * len(spec.dense_shape) + gap_legal.shape + (1, 1))
        & np.ones(spec.dense_shape + gap_legal.shape + (len(cfg.PENETRATE_LIST), len(cfg.EXEC_MODE_LIST)), dtype=bool)
    )


def scan_family_r2a(
    samples: pd.DataFrame,
    spec: FamilySpec,
    *,
    config: ResearchConfig,
    rule_id_start: int,
    progress_every: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = tuple(int(v) for v in sorted(samples["Year"].unique()))
    year_to_index = {year: i for i, year in enumerate(years)}
    legal = _make_legal(spec)
    stats = DenseStats(legal.shape, years)
    last_entry_order = np.full(legal.shape, -10_000_000, dtype=np.int32)

    total = len(samples)
    for order, row in enumerate(samples.itertuples(index=False)):
        year_index = year_to_index[int(row.Year)]
        time_index = _time_segment_index(row.TimeSegment)
        for side in (SIDE_LONG, SIDE_SHORT):
            fam = _family_match(spec, row, side)
            if fam is None:
                continue
            fam_slc, fam_mask = fam
            gap = float(row.O0 - row.C1) if side == SIDE_LONG else float(row.C1 - row.O0)
            gap_bounds = _gap_bounds(gap)
            if gap_bounds is None:
                continue
            gap_min_slc, gap_max_slc = gap_bounds
            for exec_i, mode in enumerate(cfg.EXEC_MODE_LIST):
                ok, anchor, open_class_name = _exec_signal(row, mode, side, config.break_small_max)
                if not ok:
                    continue
                pen_distance = float(anchor - row.L0) if side == SIDE_LONG else float(row.H0 - anchor)
                pen_hi = _hi(pen_distance, cfg.PENETRATE_LIST)
                exec_slc = slice(exec_i, exec_i + 1)
                trigger_slc = (*fam_slc, gap_min_slc, gap_max_slc, slice(None), exec_slc)
                raw_mask = legal[trigger_slc].copy()
                if fam_mask is not None:
                    expanded = fam_mask.reshape(fam_mask.shape + (1, 1, 1, 1))
                    raw_mask &= expanded
                active = raw_mask & (last_entry_order[trigger_slc] < order - 1)
                if not raw_mask.any():
                    continue

                direct = costed_points(side, float(row.O0), float(row.NextOpen), config.cost)
                open_class_i = _open_class_index(open_class_name)
                stats.add_trigger(
                    trigger_slc,
                    raw_mask,
                    active,
                    side=side,
                    direct_raw=direct.raw_points,
                    direct_net=direct.net_points,
                    open_class=open_class_i,
                )

                if pen_hi >= 0:
                    fill_slc = (*fam_slc, gap_min_slc, gap_max_slc, slice(0, pen_hi + 1), exec_slc)
                    fill_active = legal[fill_slc].copy()
                    if fam_mask is not None:
                        expanded = fam_mask.reshape(fam_mask.shape + (1, 1, 1, 1))
                        fill_active &= expanded[:, :, :, :, :, :, : pen_hi + 1, :]
                    fill_active &= last_entry_order[fill_slc] < order - 1
                    pullback = costed_points(side, float(anchor), float(row.NextOpen), config.cost)
                    stats.add_trade(
                        fill_slc,
                        fill_active,
                        side=side,
                        raw_points=pullback.raw_points,
                        net_points=pullback.net_points,
                        fee_twd=pullback.fee_twd,
                        tax_twd=pullback.tax_twd,
                        slippage_twd=pullback.slippage_twd,
                        year_index=year_index,
                        open_class=open_class_i,
                        time_segment=time_index,
                        direct_raw=direct.raw_points,
                        direct_net=direct.net_points,
                    )
                    last_entry_order[fill_slc][fill_active] = order

                if pen_hi < len(cfg.PENETRATE_LIST) - 1:
                    start = max(pen_hi + 1, 0)
                    unf_slc = (*fam_slc, gap_min_slc, gap_max_slc, slice(start, len(cfg.PENETRATE_LIST)), exec_slc)
                    unf_active = legal[unf_slc].copy()
                    if fam_mask is not None:
                        expanded = fam_mask.reshape(fam_mask.shape + (1, 1, 1, 1))
                        unf_active &= expanded[:, :, :, :, :, :, start:, :]
                    unf_active &= last_entry_order[unf_slc] < order - 1
                    stats.add_unfilled(
                        unf_slc,
                        unf_active,
                        direct_raw=direct.raw_points,
                        direct_net=direct.net_points,
                        open_class=open_class_i,
                    )

        if progress_every and (order + 1) % progress_every == 0:
            print(f"R2A {spec.name}: {order + 1:,}/{total:,} samples")
    print(f"R2A {spec.name}: {total:,}/{total:,} samples")

    return _flatten_family_result(spec, legal, stats, years, rule_id_start, config)


def _flatten_family_result(
    spec: FamilySpec,
    legal: np.ndarray,
    stats: DenseStats,
    years: tuple[int, ...],
    rule_id_start: int,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coords = np.where(legal)
    family_dim_count = len(spec.dense_shape)
    fam_coords = coords[:family_dim_count]
    exec_coords = coords[family_dim_count:]
    n = len(coords[0])

    family_frame = family_param_columns(spec, fam_coords)
    exec_frame = exec_param_columns(exec_coords)
    summary = pd.concat([family_frame, exec_frame], axis=1)
    summary.insert(0, "Family", spec.name)
    summary.insert(0, "StrategyID", [f"R2A_{rule_id_start + i:07d}" for i in range(n)])
    summary.insert(0, "Stage", "R2A")
    summary.insert(0, "RuleID", np.arange(rule_id_start, rule_id_start + n, dtype=np.int64))

    idx = coords
    count = stats.trade_count[idx]
    net_points = stats.net_sum[idx]
    raw_points = stats.raw_sum[idx]
    gp = stats.gross_profit[idx]
    gl = stats.gross_loss_abs[idx]
    raw_gp = stats.raw_gross_profit[idx]
    raw_gl = stats.raw_gross_loss_abs[idx]
    avg = safe_divide(net_points, count)
    raw_avg = safe_divide(raw_points, count)
    variance = safe_divide(stats.net_sq_sum[idx], count) - avg * avg
    variance = np.where((count > 0) & (variance < 0) & (variance > -1e-9), 0, variance)
    std = np.where(count > 0, np.sqrt(variance), np.nan)

    year_points = np.vstack([stats.year_net_sum[year_i][idx] for year_i in range(len(years))])
    year_counts = np.vstack([stats.year_count[year_i][idx] for year_i in range(len(years))])
    active_year = year_counts > 0
    has_year = active_year.any(axis=0)
    positive_years = ((year_points > 0) & active_year).sum(axis=0)
    negative_years = ((year_points < 0) & active_year).sum(axis=0)
    year_count = active_year.sum(axis=0)
    worst_points = np.where(has_year, np.where(active_year, year_points, np.inf).min(axis=0), np.nan)
    best_points = np.where(has_year, np.where(active_year, year_points, -np.inf).max(axis=0), np.nan)
    worst_idx = np.where(has_year, np.where(active_year, year_points, np.inf).argmin(axis=0), -1)
    best_idx = np.where(has_year, np.where(active_year, year_points, -np.inf).argmax(axis=0), -1)
    year_labels = np.asarray(years, dtype=object)

    summary["EntrySlippagePoints"] = config.cost.entry_slippage_points
    summary["ExitSlippagePoints"] = config.cost.exit_slippage_points
    summary["RoundTripFeeTWD"] = config.cost.round_trip_fee_twd
    summary["TaxRate"] = config.cost.tax_rate
    summary["PointValueTWD"] = config.cost.point_value_twd
    summary["RawTriggerCount"] = stats.raw_trigger[idx]
    summary["EligibleTriggerCount"] = stats.eligible_trigger[idx]
    summary["FillCount"] = stats.fill_count[idx]
    summary["FillRate"] = safe_divide(stats.fill_count[idx], stats.eligible_trigger[idx])
    summary["LongTriggers"] = stats.long_trigger[idx]
    summary["ShortTriggers"] = stats.short_trigger[idx]
    summary["LongTrades"] = stats.long_trades[idx]
    summary["ShortTrades"] = stats.short_trades[idx]
    summary["TotalTrades"] = count
    summary["WinTrades"] = stats.win_count[idx]
    summary["LossTrades"] = stats.loss_count[idx]
    summary["FlatTrades"] = stats.flat_count[idx]
    summary["WinRate"] = safe_divide(stats.win_count[idx], count)
    summary["RawNetPoints"] = raw_points
    summary["RawGrossProfit"] = raw_gp
    summary["RawGrossLoss"] = -raw_gl
    summary["RawPF"] = profit_factor(raw_gp, raw_gl)
    summary["RawAvgPoints"] = raw_avg
    summary["RawMedianPoints"] = np.nan
    summary["NetPoints"] = net_points
    summary["NetProfitTWD"] = net_points * config.cost.point_value_twd
    summary["TotalReturnRate"] = summary["NetProfitTWD"] / config.cost.capital_twd
    summary["GrossProfitNet"] = gp
    summary["GrossLossNet"] = -gl
    summary["PFNet"] = profit_factor(gp, gl)
    summary["AvgNetPoints"] = avg
    summary["MedianNetPoints"] = np.nan
    summary["MaxTradeNetPoints"] = np.where(count > 0, stats.max_trade[idx], np.nan)
    summary["MinTradeNetPoints"] = np.where(count > 0, stats.min_trade[idx], np.nan)
    summary["StdNetPoints"] = std
    summary["MaxDrawdownNetPoints"] = np.where(count > 0, stats.max_drawdown[idx], np.nan)
    summary["MaxLosingStreak"] = stats.max_losing_streak[idx]
    summary["TotalFeeTWD"] = stats.total_fee_twd[idx]
    summary["TotalTaxTWD"] = stats.total_tax_twd[idx]
    summary["TotalSlippageTWD"] = stats.total_slippage_twd[idx]
    summary["AvgCostPointsPerTrade"] = safe_divide(raw_points - net_points, count)
    summary["DirectOpenRawNet_AllTriggers"] = stats.direct_raw_all_sum[idx]
    summary["DirectOpenRawAvg_AllTriggers"] = safe_divide(stats.direct_raw_all_sum[idx], stats.direct_all_count[idx])
    summary["DirectOpenNetPoints_AllTriggers"] = stats.direct_net_all_sum[idx]
    summary["DirectOpenAvgNet_AllTriggers"] = safe_divide(stats.direct_net_all_sum[idx], stats.direct_all_count[idx])
    summary["DirectOpenRawNet_FilledSubset"] = stats.direct_raw_filled_sum[idx]
    summary["DirectOpenRawAvg_FilledSubset"] = safe_divide(stats.direct_raw_filled_sum[idx], stats.direct_filled_count[idx])
    summary["DirectOpenNetPoints_FilledSubset"] = stats.direct_net_filled_sum[idx]
    summary["DirectOpenAvgNet_FilledSubset"] = safe_divide(stats.direct_net_filled_sum[idx], stats.direct_filled_count[idx])
    summary["DirectOpenRawNet_UnfilledSubset"] = stats.direct_raw_unfilled_sum[idx]
    summary["DirectOpenRawAvg_UnfilledSubset"] = safe_divide(stats.direct_raw_unfilled_sum[idx], stats.direct_unfilled_count[idx])
    summary["DirectOpenNetPoints_UnfilledSubset"] = stats.direct_net_unfilled_sum[idx]
    summary["DirectOpenAvgNet_UnfilledSubset"] = safe_divide(stats.direct_net_unfilled_sum[idx], stats.direct_unfilled_count[idx])
    summary["PullbackAdvantageRaw"] = safe_divide(raw_points, stats.eligible_trigger[idx]) - safe_divide(stats.direct_raw_all_sum[idx], stats.direct_all_count[idx])
    summary["PullbackAdvantageNet"] = safe_divide(net_points, stats.eligible_trigger[idx]) - safe_divide(stats.direct_net_all_sum[idx], stats.direct_all_count[idx])
    summary["YearCount"] = year_count
    summary["PositiveYears"] = positive_years
    summary["NegativeYears"] = negative_years
    summary["WorstYear"] = np.where(worst_idx >= 0, year_labels[np.maximum(worst_idx, 0)], "")
    summary["WorstYearNetPoints"] = worst_points
    summary["BestYear"] = np.where(best_idx >= 0, year_labels[np.maximum(best_idx, 0)], "")
    summary["BestYearNetPoints"] = best_points
    summary["LongNetPoints"] = stats.side_net_sum[0][idx]
    summary["LongPFNet"] = profit_factor(stats.side_gp[0][idx], stats.side_gl_abs[0][idx])
    summary["LongAvgNetPoints"] = safe_divide(stats.side_net_sum[0][idx], stats.side_count[0][idx])
    summary["ShortNetPoints"] = stats.side_net_sum[1][idx]
    summary["ShortPFNet"] = profit_factor(stats.side_gp[1][idx], stats.side_gl_abs[1][idx])
    summary["ShortAvgNetPoints"] = safe_divide(stats.side_net_sum[1][idx], stats.side_count[1][idx])

    for class_i, name in enumerate(cfg.OPEN_CLASSES):
        prefix = "Inside" if name == "INSIDE" else "BreakSmall" if name == "BREAK_SMALL" else "BreakLarge"
        summary[f"{prefix}Trades"] = stats.class_count[class_i][idx]
        summary[f"{prefix}NetPoints"] = stats.class_net_sum[class_i][idx]
        summary[f"{prefix}PFNet"] = profit_factor(stats.class_gp[class_i][idx], stats.class_gl_abs[class_i][idx])

    for seg_i, name in enumerate(cfg.TIME_SEGMENTS):
        summary[f"{name}Trades"] = stats.segment_count[seg_i][idx]
        summary[f"{name}NetPoints"] = stats.segment_net_sum[seg_i][idx]

    by_year = _make_by_year(summary[["RuleID", "StrategyID", "Family", "ExecMode"]], stats, idx, years, config)
    by_side = _make_by_side(summary[["RuleID", "StrategyID", "Family", "ExecMode"]], stats, idx)
    by_class = _make_by_open_class(summary[["RuleID", "StrategyID", "Family", "ExecMode"]], stats, idx)
    by_segment = _make_by_time_segment(summary[["RuleID", "StrategyID", "Family", "ExecMode"]], stats, idx)
    return summary, by_year, by_side, by_class, by_segment


def _make_by_year(keys: pd.DataFrame, stats: DenseStats, idx: tuple[np.ndarray, ...], years: tuple[int, ...], config: ResearchConfig) -> pd.DataFrame:
    parts = []
    for year_i, year in enumerate(years):
        count = stats.year_count[year_i][idx]
        net = stats.year_net_sum[year_i][idx]
        part = keys.copy()
        part["Year"] = year
        part["Trades"] = count
        part["WinRate"] = safe_divide(stats.year_win[year_i][idx], count)
        part["NetPoints"] = net
        part["NetProfitTWD"] = net * config.cost.point_value_twd
        part["TotalReturnRate"] = part["NetProfitTWD"] / config.cost.capital_twd
        part["PFNet"] = profit_factor(stats.year_gp[year_i][idx], stats.year_gl_abs[year_i][idx])
        part["AvgNetPoints"] = safe_divide(net, count)
        part["MaxDrawdownNetPoints"] = stats.year_mdd[year_i][idx]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _make_by_side(keys: pd.DataFrame, stats: DenseStats, idx: tuple[np.ndarray, ...]) -> pd.DataFrame:
    parts = []
    for side_i, side_name in enumerate(["LONG", "SHORT"]):
        count = stats.side_count[side_i][idx]
        net = stats.side_net_sum[side_i][idx]
        part = keys.copy()
        part["Side"] = side_name
        part["Trades"] = count
        part["NetPoints"] = net
        part["PFNet"] = profit_factor(stats.side_gp[side_i][idx], stats.side_gl_abs[side_i][idx])
        part["AvgNetPoints"] = safe_divide(net, count)
        part["WinRate"] = safe_divide(stats.side_win[side_i][idx], count)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _make_by_open_class(keys: pd.DataFrame, stats: DenseStats, idx: tuple[np.ndarray, ...]) -> pd.DataFrame:
    parts = []
    for class_i, name in enumerate(cfg.OPEN_CLASSES):
        count = stats.class_count[class_i][idx]
        net = stats.class_net_sum[class_i][idx]
        part = keys.copy()
        part["OpenClass"] = name
        part["EligibleTriggerCount"] = stats.class_eligible[class_i][idx]
        part["FillCount"] = stats.class_fill[class_i][idx]
        part["FillRate"] = safe_divide(stats.class_fill[class_i][idx], stats.class_eligible[class_i][idx])
        part["Trades"] = count
        part["NetPoints"] = net
        part["PFNet"] = profit_factor(stats.class_gp[class_i][idx], stats.class_gl_abs[class_i][idx])
        part["AvgNetPoints"] = safe_divide(net, count)
        part["WinRate"] = safe_divide(stats.class_win[class_i][idx], count)
        part["DirectOpenNetPoints_AllTriggers"] = stats.class_direct_net_all_sum[class_i][idx]
        part["DirectOpenNetPoints_UnfilledSubset"] = stats.class_direct_net_unfilled_sum[class_i][idx]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _make_by_time_segment(keys: pd.DataFrame, stats: DenseStats, idx: tuple[np.ndarray, ...]) -> pd.DataFrame:
    parts = []
    for seg_i, name in enumerate(cfg.TIME_SEGMENTS):
        count = stats.segment_count[seg_i][idx]
        net = stats.segment_net_sum[seg_i][idx]
        part = keys.copy()
        part["TimeSegment"] = name
        part["Trades"] = count
        part["NetPoints"] = net
        part["PFNet"] = profit_factor(stats.segment_gp[seg_i][idx], stats.segment_gl_abs[seg_i][idx])
        part["AvgNetPoints"] = safe_divide(net, count)
        part["WinRate"] = safe_divide(stats.segment_win[seg_i][idx], count)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def r2a_total_combo_count(config: ResearchConfig) -> int:
    assert_expected_r2a_count(config.r2a_expected_combo_count)
    return config.r2a_expected_combo_count


def materialize_r2a_trades(samples: pd.DataFrame, rule: pd.Series, config: ResearchConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    spec = next(s for s in family_specs() if s.name == rule["Family"])
    last_entry_order = -10_000_000
    for order, row in enumerate(samples.itertuples(index=False)):
        if last_entry_order >= order - 1:
            continue
        for side in (SIDE_LONG, SIDE_SHORT):
            if not _single_family_rule_match(spec, row, side, rule):
                continue
            gap = float(row.O0 - row.C1) if side == SIDE_LONG else float(row.C1 - row.O0)
            if not (float(rule.OpenGapMin) <= gap <= float(rule.OpenGapMax)):
                continue
            ok, anchor, open_class = _exec_signal(row, str(rule.ExecMode), side, config.break_small_max)
            if not ok:
                continue
            filled = row.L0 <= anchor - rule.Penetrate if side == SIDE_LONG else row.H0 >= anchor + rule.Penetrate
            if not filled:
                continue
            pullback = costed_points(side, float(anchor), float(row.NextOpen), config.cost)
            direct = costed_points(side, float(row.O0), float(row.NextOpen), config.cost)
            rows.append(
                {
                    "StrategyID": rule.StrategyID,
                    "RuleID": int(rule.RuleID),
                    "Family": rule.Family,
                    "ExecMode": rule.ExecMode,
                    "EntryIndex": int(row.EntryIndex),
                    "EntryDate": str(pd.Timestamp(row.DateTime).date()),
                    "EntryTime": pd.Timestamp(row.DateTime).strftime("%H:%M:%S"),
                    "ExitIndex": int(row.ExitIndex),
                    "ExitDate": str(pd.Timestamp(row.NextDateTime).date()),
                    "ExitTime": pd.Timestamp(row.NextDateTime).strftime("%H:%M:%S"),
                    "ExitReason": "NEXT_OPEN",
                    "Side": _side_name(side),
                    "EntryPx": anchor,
                    "ExitPx": float(row.NextOpen),
                    "RawPoints": pullback.raw_points,
                    "NetPoints": pullback.net_points,
                    "NetProfitTWD": pullback.net_profit_twd,
                    "FeeTWD": pullback.fee_twd,
                    "TaxTWD": pullback.tax_twd,
                    "SlippageTWD": pullback.slippage_twd,
                    "CostPoints": pullback.cost_points,
                    "O1": row.O1,
                    "H1": row.H1,
                    "L1": row.L1,
                    "C1": row.C1,
                    "O0": row.O0,
                    "H0": row.H0,
                    "L0": row.L0,
                    "NextOpen": row.NextOpen,
                    "Range1": row.Range1,
                    "Body1": row.BodyLong if side == SIDE_LONG else row.BodyShort,
                    "BodyPct": row.BodyPct,
                    "ClosePosPct": row.ClosePosPct,
                    "UpperTailPct": row.UpperTailPct,
                    "LowerTailPct": row.LowerTailPct,
                    "OppTailPct": row.UpperTailPct if side == SIDE_LONG else row.LowerTailPct,
                    "MainTailPct": row.LowerTailPct if side == SIDE_LONG else row.UpperTailPct,
                    "M1": row.M1,
                    "BM1": row.BM1,
                    "Anchor": anchor,
                    "Penetrate": rule.Penetrate,
                    "OpenGap": gap,
                    "OpenClass": open_class,
                    "TimeSegment": row.TimeSegment,
                    "DirectOpenRawPoints": direct.raw_points,
                    "DirectOpenNetPoints": direct.net_points,
                }
            )
            last_entry_order = order
            break
    return pd.DataFrame(rows)


def _single_family_rule_match(spec: FamilySpec, row: object, side: int, rule: pd.Series) -> bool:
    if row.Range1 <= 0:
        return False
    if side == SIDE_LONG and not row.LongDirection:
        return False
    if side == SIDE_SHORT and not row.ShortDirection:
        return False
    body = row.BodyLong if side == SIDE_LONG else row.BodyShort
    opp_tail = row.UpperTailPct if side == SIDE_LONG else row.LowerTailPct
    main_tail = row.LowerTailPct if side == SIDE_LONG else row.UpperTailPct
    close_ok = row.ClosePosPct if side == SIDE_LONG else 100 - row.ClosePosPct

    if spec.name in {"F01_ATTACK", "F02_STRONG_CLOSE", "F03_MARUBOZU", "F04_BODY_EFFICIENCY", "F05_TAIL_SUPPORT_CONTINUATION", "F06_BODY_CENTER"}:
        if body < rule.BodyMin or row.Range1 > rule.RangeMax:
            return False
    if spec.name == "F01_ATTACK":
        return row.BodyPct >= rule.BodyPctMin and opp_tail <= rule.EffOppTailMax
    if spec.name == "F02_STRONG_CLOSE":
        return row.BodyPct >= rule.BodyPctFloor and close_ok >= rule.ClosePosMin
    if spec.name == "F03_MARUBOZU":
        return row.BodyPct >= rule.MaruBodyPct and row.UpperTailPct <= rule.TailMax and row.LowerTailPct <= rule.TailMax
    if spec.name == "F04_BODY_EFFICIENCY":
        return row.BodyPct >= rule.BodyPctMin and close_ok >= rule.ClosePosMin
    if spec.name == "F05_TAIL_SUPPORT_CONTINUATION":
        fixed_close = row.ClosePosPct >= 60 if side == SIDE_LONG else row.ClosePosPct <= 40
        return fixed_close and main_tail >= rule.MainTailMin and opp_tail <= rule.EffOppTailMax
    if spec.name == "F06_BODY_CENTER":
        center = row.BM1 - row.M1 if side == SIDE_LONG else row.M1 - row.BM1
        return row.BodyPct >= rule.BodyPctMin and close_ok >= rule.ClosePosMin and center >= rule.CenterOffset
    if spec.name == "F07_LARGE_RANGE_ATTACK":
        if not (rule.RangeMinLarge <= row.Range1 <= rule.RangeMaxLarge):
            return False
        return body >= rule.BodyMinLarge and row.BodyPct >= rule.BodyPctMin and opp_tail <= rule.EffOppTailMax
    return False
