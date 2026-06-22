from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


SPLIT_TO_INDEX = {"train": 0, "valid": 1, "test": 2}
SPLIT_NAMES = ("train", "valid", "test")


def _wilson_vector(win_count: np.ndarray, trade_count: np.ndarray) -> np.ndarray:
    z = 1.959963984540054
    wins = win_count.astype(float)
    n = trade_count.astype(float)
    out = np.full(n.shape, np.nan, dtype=float)
    mask = n > 0
    if not mask.any():
        return out
    phat = wins[mask] / n[mask]
    denom = 1 + z * z / n[mask]
    centre = phat + z * z / (2 * n[mask])
    margin = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n[mask])) / n[mask])
    out[mask] = (centre - margin) / denom
    return out


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = numerator.astype(float, copy=False)
    denominator = denominator.astype(float, copy=False)
    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _profit_factor(gross_win: np.ndarray, loss_abs: np.ndarray) -> np.ndarray:
    out = np.full(gross_win.shape, np.nan, dtype=float)
    finite_loss = loss_abs > 0
    np.divide(gross_win, loss_abs, out=out, where=finite_loss)
    out[(loss_abs == 0) & (gross_win > 0)] = np.inf
    return out


def _round_twd(value: float) -> float:
    return float(np.floor(value + 0.5))


def _trade_result(c1: float, o_next: float, side: str) -> tuple[float, float, float, float, float, float, float, float]:
    entry_slip = float(config.ENTRY_SLIPPAGE_POINTS)
    exit_slip = float(config.EXIT_SLIPPAGE_POINTS)
    point_value = float(config.POINT_VALUE_TWD)
    fee = float(config.FEE_PER_SIDE_TWD * 2)
    if side == "LONG":
        entry_price = c1 + entry_slip
        exit_price = o_next - exit_slip
        raw_points = o_next - c1
        slippage_points = entry_slip + exit_slip
        after_slip_points = exit_price - entry_price
    else:
        entry_price = c1 - entry_slip
        exit_price = o_next + exit_slip
        raw_points = c1 - o_next
        slippage_points = entry_slip + exit_slip
        after_slip_points = entry_price - exit_price

    entry_tax = _round_twd(entry_price * point_value * float(config.TRANSACTION_TAX_RATE))
    exit_tax = _round_twd(exit_price * point_value * float(config.TRANSACTION_TAX_RATE))
    tax = entry_tax + exit_tax
    slippage_twd = slippage_points * point_value
    net_profit_twd = after_slip_points * point_value - fee - tax
    net_points = net_profit_twd / point_value
    return raw_points, after_slip_points, net_points, net_profit_twd, fee, tax, slippage_twd, exit_price


@dataclass
class OnlineStats:
    shape: tuple[int, int, int, int]
    years: tuple[int, ...]

    def __post_init__(self) -> None:
        self.trade_count = np.zeros(self.shape, dtype=np.int32)
        self.win_count = np.zeros(self.shape, dtype=np.int32)
        self.loss_count = np.zeros(self.shape, dtype=np.int32)
        self.tie_count = np.zeros(self.shape, dtype=np.int32)
        self.sum_points = np.zeros(self.shape, dtype=np.float64)
        self.gross_win_points = np.zeros(self.shape, dtype=np.float64)
        self.loss_abs_points = np.zeros(self.shape, dtype=np.float64)
        self.equity = np.zeros(self.shape, dtype=np.float64)
        self.peak = np.zeros(self.shape, dtype=np.float64)
        self.max_drawdown = np.zeros(self.shape, dtype=np.float64)
        self.current_win_streak = np.zeros(self.shape, dtype=np.int32)
        self.current_loss_streak = np.zeros(self.shape, dtype=np.int32)
        self.max_win_streak = np.zeros(self.shape, dtype=np.int32)
        self.max_loss_streak = np.zeros(self.shape, dtype=np.int32)
        self.first_order = np.full(self.shape, -1, dtype=np.int32)
        self.last_order = np.full(self.shape, -1, dtype=np.int32)

        split_shape = (len(SPLIT_NAMES), *self.shape)
        self.split_trade_count = np.zeros(split_shape, dtype=np.int32)
        self.split_win_count = np.zeros(split_shape, dtype=np.int32)
        self.split_sum_points = np.zeros(split_shape, dtype=np.float64)
        self.split_gross_win_points = np.zeros(split_shape, dtype=np.float64)
        self.split_loss_abs_points = np.zeros(split_shape, dtype=np.float64)
        self.total_fee_twd = np.zeros(self.shape, dtype=np.float64)
        self.total_tax_twd = np.zeros(self.shape, dtype=np.float64)
        self.total_slippage_twd = np.zeros(self.shape, dtype=np.float64)

        year_shape = (len(self.years), *self.shape)
        self.year_trade_count = np.zeros(year_shape, dtype=np.int32)
        self.year_win_count = np.zeros(year_shape, dtype=np.int32)
        self.year_sum_points = np.zeros(year_shape, dtype=np.float64)
        self.year_equity = np.zeros(year_shape, dtype=np.float64)
        self.year_peak = np.zeros(year_shape, dtype=np.float64)
        self.year_max_drawdown = np.zeros(year_shape, dtype=np.float64)

    def update(
        self,
        slc: tuple[slice, slice, slice, slice],
        point: float,
        order: int,
        split_index: int,
        year_index: int,
        *,
        fee_twd: float,
        tax_twd: float,
        slippage_twd: float,
    ) -> None:
        count_view = self.trade_count[slc]
        first_view = self.first_order[slc]
        first_view[count_view == 0] = order
        self.last_order[slc] = order

        count_view += 1
        self.sum_points[slc] += point
        self.equity[slc] += point
        self.total_fee_twd[slc] += fee_twd
        self.total_tax_twd[slc] += tax_twd
        self.total_slippage_twd[slc] += slippage_twd

        peak_view = self.peak[slc]
        equity_view = self.equity[slc]
        np.maximum(peak_view, equity_view, out=peak_view)
        drawdown_view = peak_view - equity_view
        mdd_view = self.max_drawdown[slc]
        np.maximum(mdd_view, drawdown_view, out=mdd_view)

        split_count = self.split_trade_count[split_index][slc]
        split_count += 1
        self.split_sum_points[split_index][slc] += point

        year_count = self.year_trade_count[year_index][slc]
        year_count += 1
        self.year_sum_points[year_index][slc] += point
        self.year_equity[year_index][slc] += point
        year_peak = self.year_peak[year_index][slc]
        year_equity = self.year_equity[year_index][slc]
        np.maximum(year_peak, year_equity, out=year_peak)
        year_dd = year_peak - year_equity
        year_mdd = self.year_max_drawdown[year_index][slc]
        np.maximum(year_mdd, year_dd, out=year_mdd)

        if point > 0:
            self.win_count[slc] += 1
            self.gross_win_points[slc] += point
            self.split_win_count[split_index][slc] += 1
            self.split_gross_win_points[split_index][slc] += point
            self.year_win_count[year_index][slc] += 1
            win_view = self.current_win_streak[slc]
            win_view += 1
            np.maximum(self.max_win_streak[slc], win_view, out=self.max_win_streak[slc])
            self.current_loss_streak[slc] = 0
        elif point < 0:
            loss = -point
            self.loss_count[slc] += 1
            self.loss_abs_points[slc] += loss
            self.split_loss_abs_points[split_index][slc] += loss
            loss_view = self.current_loss_streak[slc]
            loss_view += 1
            np.maximum(self.max_loss_streak[slc], loss_view, out=self.max_loss_streak[slc])
            self.current_win_streak[slc] = 0
        else:
            self.tie_count[slc] += 1
            self.current_win_streak[slc] = 0
            self.current_loss_streak[slc] = 0

    def to_frame(
        self,
        *,
        prefix: str,
        params: pd.DataFrame,
        min_idx: np.ndarray,
        max_idx: np.ndarray,
        pullback_idx: np.ndarray,
        body_idx: np.ndarray,
        datetime_strings: np.ndarray,
    ) -> pd.DataFrame:
        idx = (min_idx, max_idx, pullback_idx, body_idx)
        trade_count = self.trade_count[idx].astype(np.int32)
        win_count = self.win_count[idx].astype(np.int32)
        loss_count = self.loss_count[idx].astype(np.int32)
        tie_count = self.tie_count[idx].astype(np.int32)
        sum_points = self.sum_points[idx]
        gross_win = self.gross_win_points[idx]
        loss_abs = self.loss_abs_points[idx]
        pf = _profit_factor(gross_win, loss_abs)
        first_order = self.first_order[idx]
        last_order = self.last_order[idx]

        first_dt = np.full(len(params), "", dtype=object)
        last_dt = np.full(len(params), "", dtype=object)
        first_mask = first_order >= 0
        last_mask = last_order >= 0
        first_dt[first_mask] = datetime_strings[first_order[first_mask]]
        last_dt[last_mask] = datetime_strings[last_order[last_mask]]

        df = pd.DataFrame(
            {
                "rule_id": params["rule_id"].to_numpy(),
                f"{prefix}_trade_count": trade_count,
                f"{prefix}_win_count": win_count,
                f"{prefix}_loss_count": loss_count,
                f"{prefix}_tie_count": tie_count,
                f"{prefix}_win_rate": _safe_divide(win_count, trade_count),
                f"{prefix}_loss_rate": _safe_divide(loss_count, trade_count),
                f"{prefix}_tie_rate": _safe_divide(tie_count, trade_count),
                f"{prefix}_avg_points": _safe_divide(sum_points, trade_count),
                f"{prefix}_median_points": np.full(len(params), np.nan, dtype=float),
                f"{prefix}_gross_points": sum_points,
                f"{prefix}_net_profit_twd": sum_points * float(config.POINT_VALUE_TWD),
                f"{prefix}_return_rate": (sum_points * float(config.POINT_VALUE_TWD)) / float(config.CAPITAL_TWD),
                f"{prefix}_avg_win_points": _safe_divide(gross_win, win_count),
                f"{prefix}_avg_loss_points": -_safe_divide(loss_abs, loss_count),
                f"{prefix}_profit_factor": pf,
                f"{prefix}_profit_factor_capped": np.minimum(np.where(np.isfinite(pf), pf, 10.0), 10.0),
                f"{prefix}_max_drawdown_points": np.where(trade_count > 0, self.max_drawdown[idx], np.nan),
                f"{prefix}_max_drawdown_twd": np.where(
                    trade_count > 0, self.max_drawdown[idx] * float(config.POINT_VALUE_TWD), np.nan
                ),
                f"{prefix}_total_fee_twd": self.total_fee_twd[idx],
                f"{prefix}_total_tax_twd": self.total_tax_twd[idx],
                f"{prefix}_total_slippage_twd": self.total_slippage_twd[idx],
                f"{prefix}_max_losing_streak": self.max_loss_streak[idx].astype(np.int32),
                f"{prefix}_max_winning_streak": self.max_win_streak[idx].astype(np.int32),
                f"{prefix}_first_trade_datetime": first_dt,
                f"{prefix}_last_trade_datetime": last_dt,
                f"{prefix}_wilson_win_rate_lower_95": _wilson_vector(win_count, trade_count),
            }
        )

        for split_i, split in enumerate(SPLIT_NAMES):
            split_count = self.split_trade_count[split_i][idx].astype(np.int32)
            split_win = self.split_win_count[split_i][idx].astype(np.int32)
            split_sum = self.split_sum_points[split_i][idx]
            split_gross_win = self.split_gross_win_points[split_i][idx]
            split_loss_abs = self.split_loss_abs_points[split_i][idx]
            df[f"{prefix}_{split}_trade_count"] = split_count
            df[f"{prefix}_{split}_win_rate"] = _safe_divide(split_win, split_count)
            df[f"{prefix}_{split}_avg_points"] = _safe_divide(split_sum, split_count)
            df[f"{prefix}_{split}_profit_factor"] = _profit_factor(split_gross_win, split_loss_abs)

        for year_i, year in enumerate(self.years):
            year_count = self.year_trade_count[year_i][idx].astype(np.int32)
            year_win = self.year_win_count[year_i][idx].astype(np.int32)
            year_sum = self.year_sum_points[year_i][idx]
            year_mdd = self.year_max_drawdown[year_i][idx]
            df[f"{prefix}_{year}_trade_count"] = year_count
            df[f"{prefix}_{year}_win_rate"] = _safe_divide(year_win, year_count)
            df[f"{prefix}_{year}_net_points"] = year_sum
            df[f"{prefix}_{year}_return_rate"] = (year_sum * float(config.POINT_VALUE_TWD)) / float(config.CAPITAL_TWD)
            df[f"{prefix}_{year}_max_drawdown_twd"] = year_mdd * float(config.POINT_VALUE_TWD)

        return df


def _threshold_high_index(values: np.ndarray, thresholds: list[int]) -> np.ndarray:
    return np.searchsorted(np.asarray(thresholds, dtype=float), values, side="right") - 1


def _gap_bounds(gap: float, min_values: np.ndarray, max_values: np.ndarray) -> tuple[int, int] | None:
    min_hi = int(np.searchsorted(min_values, gap, side="right") - 1)
    max_lo = int(np.searchsorted(max_values, gap, side="left"))
    if min_hi < 0 or max_lo >= len(max_values):
        return None
    return min_hi, max_lo


def _add_robust_score(df: pd.DataFrame, prefix: str, min_trades: int) -> None:
    pf = df[f"{prefix}_profit_factor"].to_numpy(dtype=float)
    pf_score = np.where(np.isnan(pf), 0.0, np.where(np.isfinite(pf), np.minimum(pf, 3.0), 3.0))
    wilson = df[f"{prefix}_wilson_win_rate_lower_95"].fillna(0).to_numpy(dtype=float)
    avg = df[f"{prefix}_avg_points"].fillna(0).to_numpy(dtype=float)
    valid_win = df[f"{prefix}_valid_win_rate"].fillna(0).to_numpy(dtype=float)
    test_win = df[f"{prefix}_test_win_rate"].fillna(0).to_numpy(dtype=float)
    valid_avg = df[f"{prefix}_valid_avg_points"].to_numpy(dtype=float)
    test_avg = df[f"{prefix}_test_avg_points"].to_numpy(dtype=float)
    count = df[f"{prefix}_trade_count"].to_numpy(dtype=float)
    score = wilson * 100 + avg * 5 + pf_score * 5 + valid_win * 10 + test_win * 20
    score -= np.where(np.isnan(valid_avg) | (valid_avg <= 0), 20.0, 0.0)
    score -= np.where(np.isnan(test_avg) | (test_avg <= 0), 30.0, 0.0)
    score -= np.where(count < min_trades, 50.0, 0.0)
    df[f"{prefix}_robust_score"] = score


def _setup_counts(body_values: np.ndarray, body_thresholds: list[int]) -> np.ndarray:
    high_idx = _threshold_high_index(body_values, body_thresholds)
    out = np.zeros(len(body_thresholds), dtype=np.int32)
    for idx in high_idx:
        if idx >= 0:
            out[: idx + 1] += 1
    return out


def _resolve_param_indices(params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_map = {value: idx for idx, value in enumerate(config.OPEN_GAP_MIN_LIST)}
    max_map = {value: idx for idx, value in enumerate(config.OPEN_GAP_MAX_LIST)}
    body_map = {value: idx for idx, value in enumerate(config.BODY_MIN_LIST)}
    pullback_map = {value: idx for idx, value in enumerate(config.PULLBACK_DEPTH_LIST)}
    return (
        params["open_gap_min"].map(min_map).to_numpy(dtype=np.int16),
        params["open_gap_max"].map(max_map).to_numpy(dtype=np.int16),
        params["pullback_depth"].map(pullback_map).to_numpy(dtype=np.int16),
        params["body_min"].map(body_map).to_numpy(dtype=np.int16),
    )


def scan_l1(
    samples: pd.DataFrame,
    params: pd.DataFrame,
    *,
    min_trades: int,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    min_values = np.asarray(config.OPEN_GAP_MIN_LIST, dtype=float)
    max_values = np.asarray(config.OPEN_GAP_MAX_LIST, dtype=float)
    body_thresholds = config.BODY_MIN_LIST
    pullback_thresholds = config.PULLBACK_DEPTH_LIST
    shape = (
        len(config.OPEN_GAP_MIN_LIST),
        len(config.OPEN_GAP_MAX_LIST),
        len(config.PULLBACK_DEPTH_LIST),
        len(config.BODY_MIN_LIST),
    )

    long_signal_count = np.zeros(shape, dtype=np.int32)
    short_signal_count = np.zeros(shape, dtype=np.int32)
    years = tuple(int(year) for year in sorted(samples["year"].unique()))
    year_to_index = {year: idx for idx, year in enumerate(years)}
    long_stats = OnlineStats(shape, years)
    short_stats = OnlineStats(shape, years)
    combined_stats = OnlineStats(shape, years)

    o1 = samples["O1"].to_numpy(dtype=float)
    c1 = samples["C1"].to_numpy(dtype=float)
    o0 = samples["O0"].to_numpy(dtype=float)
    h0 = samples["H0"].to_numpy(dtype=float)
    l0 = samples["L0"].to_numpy(dtype=float)
    o_next = samples["O_NEXT"].to_numpy(dtype=float)
    splits = samples["split"].map(SPLIT_TO_INDEX).to_numpy(dtype=np.int8)
    year_indices = samples["year"].map(year_to_index).to_numpy(dtype=np.int8)
    datetime_strings = pd.to_datetime(samples["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

    long_body = c1 - o1
    short_body = o1 - c1
    long_setup_by_body = _setup_counts(long_body, body_thresholds)
    short_setup_by_body = _setup_counts(short_body, body_thresholds)
    long_body_hi = _threshold_high_index(long_body, body_thresholds)
    short_body_hi = _threshold_high_index(short_body, body_thresholds)
    long_pull_hi = _threshold_high_index(c1 - l0, pullback_thresholds)
    short_pull_hi = _threshold_high_index(h0 - c1, pullback_thresholds)

    total = len(samples)
    for order in range(total):
        split_index = int(splits[order])
        year_index = int(year_indices[order])

        body_hi = int(long_body_hi[order])
        if body_hi >= 0:
            bounds = _gap_bounds(float(o0[order] - c1[order]), min_values, max_values)
            if bounds is not None:
                min_hi, max_lo = bounds
                signal_slc = (slice(0, min_hi + 1), slice(max_lo, len(max_values)), slice(None), slice(0, body_hi + 1))
                long_signal_count[signal_slc] += 1
                pull_hi = int(long_pull_hi[order])
                if pull_hi >= 0:
                    trade_slc = (
                        slice(0, min_hi + 1),
                        slice(max_lo, len(max_values)),
                        slice(0, pull_hi + 1),
                        slice(0, body_hi + 1),
                    )
                    _, _, point, _, fee, tax, slippage, _ = _trade_result(c1[order], o_next[order], "LONG")
                    long_stats.update(
                        trade_slc,
                        point,
                        order,
                        split_index,
                        year_index,
                        fee_twd=fee,
                        tax_twd=tax,
                        slippage_twd=slippage,
                    )
                    combined_stats.update(
                        trade_slc,
                        point,
                        order,
                        split_index,
                        year_index,
                        fee_twd=fee,
                        tax_twd=tax,
                        slippage_twd=slippage,
                    )

        body_hi = int(short_body_hi[order])
        if body_hi >= 0:
            bounds = _gap_bounds(float(c1[order] - o0[order]), min_values, max_values)
            if bounds is not None:
                min_hi, max_lo = bounds
                signal_slc = (slice(0, min_hi + 1), slice(max_lo, len(max_values)), slice(None), slice(0, body_hi + 1))
                short_signal_count[signal_slc] += 1
                pull_hi = int(short_pull_hi[order])
                if pull_hi >= 0:
                    trade_slc = (
                        slice(0, min_hi + 1),
                        slice(max_lo, len(max_values)),
                        slice(0, pull_hi + 1),
                        slice(0, body_hi + 1),
                    )
                    _, _, point, _, fee, tax, slippage, _ = _trade_result(c1[order], o_next[order], "SHORT")
                    short_stats.update(
                        trade_slc,
                        point,
                        order,
                        split_index,
                        year_index,
                        fee_twd=fee,
                        tax_twd=tax,
                        slippage_twd=slippage,
                    )
                    combined_stats.update(
                        trade_slc,
                        point,
                        order,
                        split_index,
                        year_index,
                        fee_twd=fee,
                        tax_twd=tax,
                        slippage_twd=slippage,
                    )

        if chunk_size > 0 and (order + 1) % chunk_size == 0:
            print(f"scan progress: {order + 1:,}/{total:,} samples")
    print(f"scan progress: {total:,}/{total:,} samples")

    min_idx, max_idx, pullback_idx, body_idx = _resolve_param_indices(params)
    result = params[
        ["rule_id", "open_gap_min", "open_gap_max", "pullback_depth", "body_min"]
    ].copy()
    result["long_setup_count"] = long_setup_by_body[body_idx]
    result["long_signal_count"] = long_signal_count[min_idx, max_idx, pullback_idx, body_idx]
    result["long_fill_count"] = long_stats.trade_count[min_idx, max_idx, pullback_idx, body_idx]
    result["long_fill_rate"] = _safe_divide(
        result["long_fill_count"].to_numpy(), result["long_signal_count"].to_numpy()
    )
    result = result.merge(
        long_stats.to_frame(
            prefix="long",
            params=params,
            min_idx=min_idx,
            max_idx=max_idx,
            pullback_idx=pullback_idx,
            body_idx=body_idx,
            datetime_strings=datetime_strings,
        ),
        on="rule_id",
        how="left",
    )

    result["short_setup_count"] = short_setup_by_body[body_idx]
    result["short_signal_count"] = short_signal_count[min_idx, max_idx, pullback_idx, body_idx]
    result["short_fill_count"] = short_stats.trade_count[min_idx, max_idx, pullback_idx, body_idx]
    result["short_fill_rate"] = _safe_divide(
        result["short_fill_count"].to_numpy(), result["short_signal_count"].to_numpy()
    )
    result = result.merge(
        short_stats.to_frame(
            prefix="short",
            params=params,
            min_idx=min_idx,
            max_idx=max_idx,
            pullback_idx=pullback_idx,
            body_idx=body_idx,
            datetime_strings=datetime_strings,
        ),
        on="rule_id",
        how="left",
    )
    result = result.merge(
        combined_stats.to_frame(
            prefix="combined",
            params=params,
            min_idx=min_idx,
            max_idx=max_idx,
            pullback_idx=pullback_idx,
            body_idx=body_idx,
            datetime_strings=datetime_strings,
        ),
        on="rule_id",
        how="left",
    )

    for prefix in ("long", "short", "combined"):
        _add_robust_score(result, prefix, min_trades)
    if len(result) != config.EXPECTED_PARAM_COUNT:
        raise RuntimeError(f"main result row count {len(result):,} != {config.EXPECTED_PARAM_COUNT:,}")
    return result


def materialize_rule_trades(samples: pd.DataFrame, rule: pd.Series) -> pd.DataFrame:
    gap_min = float(rule["open_gap_min"])
    gap_max = float(rule["open_gap_max"])
    pullback = float(rule["pullback_depth"])
    body_min = float(rule["body_min"])

    long_mask = (
        (samples["C1"] - samples["O1"] >= body_min)
        & (samples["O0"] - samples["C1"] >= gap_min)
        & (samples["O0"] - samples["C1"] <= gap_max)
        & (samples["L0"] <= samples["C1"] - pullback)
    )
    short_mask = (
        (samples["O1"] - samples["C1"] >= body_min)
        & (samples["C1"] - samples["O0"] >= gap_min)
        & (samples["C1"] - samples["O0"] <= gap_max)
        & (samples["H0"] >= samples["C1"] + pullback)
    )

    rows = []
    for side, mask in (("LONG", long_mask), ("SHORT", short_mask)):
        part = samples.loc[mask, ["datetime", "year", "O1", "H1", "L1", "C1", "O0", "H0", "L0", "C0", "O_NEXT"]].copy()
        if part.empty:
            continue
        part["side"] = side
        part["rule_id"] = int(rule["rule_id"])
        part["open_gap_min"] = int(rule["open_gap_min"])
        part["open_gap_max"] = int(rule["open_gap_max"])
        part["pullback_depth"] = int(rule["pullback_depth"])
        part["body_min"] = int(rule["body_min"])
        part["entry_price"] = part["C1"]
        if side == "LONG":
            result = part.apply(lambda row: _trade_result(row["C1"], row["O_NEXT"], "LONG"), axis=1, result_type="expand")
        else:
            result = part.apply(lambda row: _trade_result(row["C1"], row["O_NEXT"], "SHORT"), axis=1, result_type="expand")
        result.columns = [
            "raw_points",
            "after_slippage_points",
            "net_points",
            "net_profit_twd",
            "fee_twd",
            "tax_twd",
            "slippage_twd",
            "exit_price",
        ]
        part = pd.concat([part.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
        part["points"] = part["net_points"]
        part["win_loss"] = np.where(part["net_points"] > 0, "WIN", np.where(part["net_points"] < 0, "LOSS", "TIE"))
        rows.append(part)
    if not rows:
        return pd.DataFrame(
            columns=[
                "datetime",
                "side",
                "open_gap_min",
                "open_gap_max",
                "pullback_depth",
                "body_min",
                "O1",
                "H1",
                "L1",
                "C1",
                "O0",
                "H0",
                "L0",
                "C0",
                "O_NEXT",
                "entry_price",
                "exit_price",
                "raw_points",
                "after_slippage_points",
                "net_points",
                "net_profit_twd",
                "fee_twd",
                "tax_twd",
                "slippage_twd",
                "points",
                "win_loss",
            ]
        )
    trades = pd.concat(rows, ignore_index=True).sort_values(["datetime", "side"], kind="mergesort")
    return trades[
        [
            "datetime",
            "side",
            "open_gap_min",
            "open_gap_max",
            "pullback_depth",
            "body_min",
            "O1",
            "H1",
            "L1",
            "C1",
            "O0",
            "H0",
            "L0",
            "C0",
            "O_NEXT",
            "entry_price",
            "exit_price",
            "raw_points",
            "after_slippage_points",
            "net_points",
            "net_profit_twd",
            "fee_twd",
            "tax_twd",
            "slippage_twd",
            "points",
            "win_loss",
            "year",
            "rule_id",
        ]
    ].reset_index(drop=True)
