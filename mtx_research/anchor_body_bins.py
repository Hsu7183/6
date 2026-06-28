from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CostConfig
from .data_loader import load_ohlcv
from .metrics import costed_points, profit_factor, safe_divide
from .xs_anchor_rod import XSParams, _anchor_values, _int_anchor_values, _prepare_samples


STRATEGY_ID = "A01_A08_BODY_GAP_BIN_ROD_P1"
ANCHOR_MODES = list(range(1, 9))
PENETRATE = 1
WINRATE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)

# 前 K 實體區間：0~10, 11~20, ... 391~400, 401以上，共 41 組。
BODY_BINS: list[tuple[int, int | None]] = (
    [(0, 10)]
    + [(start, start + 9) for start in range(11, 401, 10)]
    + [(401, None)]
)

# OpenGap 區間：2~4, 5~7, ... 98~100, 101以上，共 34 組。
GAP_BINS: list[tuple[int, int | None]] = (
    [(start, start + 2) for start in range(2, 101, 3)]
    + [(101, None)]
)

EXPECTED_COMBOS = len(ANCHOR_MODES) * len(BODY_BINS) * len(GAP_BINS)

ANCHOR_LABELS = {
    1: "C[1] 前收",
    2: "M[1] 前高低中位",
    3: "BM[1] 前實體中位",
    4: "H/L[1] 前高/前低",
    5: "O[1] 前開",
    6: "BodyTop/BodyBot 前實體頂底",
    7: "BodyBot/BodyTop 前實體底頂",
    8: "Q75/Q25 前K四分位",
}

ANCHOR_LONG_FORMULAS = {
    1: "C[1]",
    2: "M[1]",
    3: "BM[1]",
    4: "H[1]",
    5: "O[1]",
    6: "BodyTop[1]",
    7: "BodyBot[1]",
    8: "Q75[1]",
}

ANCHOR_SHORT_FORMULAS = {
    1: "C[1]",
    2: "M[1]",
    3: "BM[1]",
    4: "L[1]",
    5: "O[1]",
    6: "BodyBot[1]",
    7: "BodyTop[1]",
    8: "Q25[1]",
}


def _coord_rule_id(coord: tuple[int, int, int]) -> int:
    anchor_i, body_i, gap_i = coord
    return anchor_i * len(BODY_BINS) * len(GAP_BINS) + body_i * len(GAP_BINS) + gap_i + 1


def _threshold_label(threshold: float) -> str:
    if threshold >= 1.0:
        return "勝率 = 100%"
    return f"勝率 > {int(threshold * 100)}%"


def _select_by_threshold(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    valid = summary["TotalTrades"] > 0
    if threshold >= 1.0:
        return summary[valid & (summary["WinRate"] >= 1.0 - 1e-12)].copy()
    return summary[valid & (summary["WinRate"] > threshold)].copy()


@dataclass
class BodyGapStats:
    shape: tuple[int, int, int]
    years: tuple[int, ...]

    def __post_init__(self) -> None:
        self.triggers = np.zeros(self.shape, dtype=np.int32)
        self.trades = np.zeros(self.shape, dtype=np.int32)
        self.long_trades = np.zeros(self.shape, dtype=np.int32)
        self.short_trades = np.zeros(self.shape, dtype=np.int32)
        self.wins = np.zeros(self.shape, dtype=np.int32)
        self.losses = np.zeros(self.shape, dtype=np.int32)
        self.flats = np.zeros(self.shape, dtype=np.int32)
        self.net_points = np.zeros(self.shape, dtype=np.float64)
        self.raw_points = np.zeros(self.shape, dtype=np.float64)
        self.gp = np.zeros(self.shape, dtype=np.float64)
        self.gl_abs = np.zeros(self.shape, dtype=np.float64)
        self.equity = np.zeros(self.shape, dtype=np.float64)
        self.peak = np.zeros(self.shape, dtype=np.float64)
        self.mdd = np.zeros(self.shape, dtype=np.float64)
        self.fees = np.zeros(self.shape, dtype=np.float64)
        self.taxes = np.zeros(self.shape, dtype=np.float64)
        self.slippage = np.zeros(self.shape, dtype=np.float64)
        self.trade_rows: list[dict[str, object]] = []

        year_shape = (len(self.years), *self.shape)
        self.year_trades = np.zeros(year_shape, dtype=np.int32)
        self.year_wins = np.zeros(year_shape, dtype=np.int32)
        self.year_net = np.zeros(year_shape, dtype=np.float64)
        self.year_gp = np.zeros(year_shape, dtype=np.float64)
        self.year_gl_abs = np.zeros(year_shape, dtype=np.float64)
        self.year_equity = np.zeros(year_shape, dtype=np.float64)
        self.year_peak = np.zeros(year_shape, dtype=np.float64)
        self.year_mdd = np.zeros(year_shape, dtype=np.float64)
        self.daily_net_twd: dict[tuple[int, int, int], float] = {}

    def add_trigger(self, mask: np.ndarray) -> None:
        if mask.any():
            self.triggers[mask] += 1

    def add_trigger_at(self, coord: tuple[int, int, int]) -> None:
        self.triggers[coord] += 1

    def add_trade(
        self,
        mask: np.ndarray,
        *,
        side: int,
        raw_points: float,
        net_points: float,
        fee_twd: int,
        tax_twd: int,
        slippage_twd: float,
        year_index: int,
        date_int: int | None = None,
        net_profit_twd: float | None = None,
        entry_time: object | None = None,
        exit_time: object | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        effective_entry: float | None = None,
        effective_exit: float | None = None,
    ) -> None:
        if not mask.any():
            return
        self.trades[mask] += 1
        if side == 1:
            self.long_trades[mask] += 1
        else:
            self.short_trades[mask] += 1
        self.raw_points[mask] += raw_points
        self.net_points[mask] += net_points
        self.fees[mask] += fee_twd
        self.taxes[mask] += tax_twd
        self.slippage[mask] += slippage_twd

        self.equity[mask] += net_points
        self.peak[mask] = np.maximum(self.peak[mask], self.equity[mask])
        self.mdd[mask] = np.maximum(self.mdd[mask], self.peak[mask] - self.equity[mask])

        year_trade = self.year_trades[year_index]
        year_trade[mask] += 1
        self.year_net[year_index][mask] += net_points
        self.year_equity[year_index][mask] += net_points
        self.year_peak[year_index][mask] = np.maximum(
            self.year_peak[year_index][mask],
            self.year_equity[year_index][mask],
        )
        self.year_mdd[year_index][mask] = np.maximum(
            self.year_mdd[year_index][mask],
            self.year_peak[year_index][mask] - self.year_equity[year_index][mask],
        )

        if net_points > 0:
            self.wins[mask] += 1
            self.gp[mask] += net_points
            self.year_wins[year_index][mask] += 1
            self.year_gp[year_index][mask] += net_points
        elif net_points < 0:
            loss = -net_points
            self.losses[mask] += 1
            self.gl_abs[mask] += loss
            self.year_gl_abs[year_index][mask] += loss
        else:
            self.flats[mask] += 1

    def add_trade_at(
        self,
        coord: tuple[int, int, int],
        *,
        side: int,
        raw_points: float,
        net_points: float,
        fee_twd: int,
        tax_twd: int,
        slippage_twd: float,
        year_index: int,
        date_int: int | None = None,
        net_profit_twd: float | None = None,
        entry_time: object | None = None,
        exit_time: object | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        effective_entry: float | None = None,
        effective_exit: float | None = None,
    ) -> None:
        self.trades[coord] += 1
        if side == 1:
            self.long_trades[coord] += 1
        else:
            self.short_trades[coord] += 1
        self.raw_points[coord] += raw_points
        self.net_points[coord] += net_points
        self.fees[coord] += fee_twd
        self.taxes[coord] += tax_twd
        self.slippage[coord] += slippage_twd

        self.equity[coord] += net_points
        self.peak[coord] = max(self.peak[coord], self.equity[coord])
        self.mdd[coord] = max(self.mdd[coord], self.peak[coord] - self.equity[coord])

        self.year_trades[year_index][coord] += 1
        self.year_net[year_index][coord] += net_points
        self.year_equity[year_index][coord] += net_points
        self.year_peak[year_index][coord] = max(
            self.year_peak[year_index][coord],
            self.year_equity[year_index][coord],
        )
        self.year_mdd[year_index][coord] = max(
            self.year_mdd[year_index][coord],
            self.year_peak[year_index][coord] - self.year_equity[year_index][coord],
        )

        if net_points > 0:
            self.wins[coord] += 1
            self.gp[coord] += net_points
            self.year_wins[year_index][coord] += 1
            self.year_gp[year_index][coord] += net_points
        elif net_points < 0:
            loss = -net_points
            self.losses[coord] += 1
            self.gl_abs[coord] += loss
            self.year_gl_abs[year_index][coord] += loss
        else:
            self.flats[coord] += 1

        if date_int is not None and net_profit_twd is not None:
            key = (_coord_rule_id(coord), int(date_int), int(side))
            self.daily_net_twd[key] = self.daily_net_twd.get(key, 0.0) + float(net_profit_twd)

        if (
            date_int is not None
            and net_profit_twd is not None
            and entry_time is not None
            and exit_time is not None
            and entry_price is not None
            and exit_price is not None
        ):
            self.trade_rows.append(
                {
                    "RuleID": _coord_rule_id(coord),
                    "Date": int(date_int),
                    "EntryTime": pd.Timestamp(entry_time).strftime("%Y-%m-%d %H:%M"),
                    "ExitTime": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"),
                    "Side": int(side),
                    "SideLabel": "做多" if side == 1 else "做空",
                    "EntryPrice": float(entry_price),
                    "ExitPrice": float(exit_price),
                    "EffectiveEntry": float(effective_entry if effective_entry is not None else entry_price),
                    "EffectiveExit": float(effective_exit if effective_exit is not None else exit_price),
                    "RawPoints": float(raw_points),
                    "NetPoints": float(net_points),
                    "NetProfitTWD": float(net_profit_twd),
                    "FeeTWD": int(fee_twd),
                    "TaxTWD": int(tax_twd),
                    "SlippageTWD": float(slippage_twd),
                }
            )


def _range_labels(bins: list[tuple[int, int | None]]) -> list[str]:
    return [f"{lo}以上" if hi is None else f"{lo}~{hi}" for lo, hi in bins]


def _single_bin_mask(value: float, bins: list[tuple[int, int | None]]) -> np.ndarray:
    out = np.zeros(len(bins), dtype=bool)
    for i, (lo, hi) in enumerate(bins):
        if value >= lo and (hi is None or value <= hi):
            out[i] = True
            break
    return out


def _bin_index(value: float, bins: list[tuple[int, int | None]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if value >= lo and (hi is None or value <= hi):
            return i
    return -1


def _gap_bin_mask(distance: np.ndarray) -> np.ndarray:
    out = np.zeros((len(distance), len(GAP_BINS)), dtype=bool)
    for i, (lo, hi) in enumerate(GAP_BINS):
        out[:, i] = distance >= lo
        if hi is not None:
            out[:, i] &= distance <= hi
    return out


def _candidate_coords(body_bin: int, gap_bins: list[int]) -> set[tuple[int, int, int]]:
    if body_bin < 0:
        return set()
    return {
        (anchor_i, body_bin, gap_i)
        for anchor_i, gap_i in enumerate(gap_bins)
        if gap_i >= 0
    }


def scan(
    data_path: Path,
    outdir: Path,
    *,
    params: XSParams | None = None,
    cost: CostConfig | None = None,
    progress_every: int = 100_000,
) -> dict[str, Path]:
    params = params or XSParams()
    cost = cost or CostConfig()
    if EXPECTED_COMBOS != 11_152:
        raise RuntimeError(f"combo count {EXPECTED_COMBOS:,} != 11,152")

    outdir.mkdir(parents=True, exist_ok=True)
    df, data_report = load_ohlcv(data_path)
    samples = _prepare_samples(df, params)
    years = tuple(int(y) for y in sorted(samples["Year"].unique()))
    year_to_index = {year: i for i, year in enumerate(years)}

    shape = (len(ANCHOR_MODES), len(BODY_BINS), len(GAP_BINS))
    stats = BodyGapStats(shape, years)
    last_entry_raw_index = np.full(shape, -10_000_000, dtype=np.int32)
    pending_raw_index = np.full(shape, -1, dtype=np.int32)

    for order, row in enumerate(samples.itertuples(index=False)):
        year_index = year_to_index[int(row.Year)]
        raw_index = int(row.EntryRawIndex)
        raw_long_anchors, raw_short_anchors = _anchor_values(row)
        int_long_anchors, int_short_anchors = _int_anchor_values(raw_long_anchors, raw_short_anchors, params)

        pending_expired = (pending_raw_index >= 0) & (raw_index > pending_raw_index)
        cancel_block = pending_expired & (raw_index == pending_raw_index + 1)
        if pending_expired.any():
            pending_raw_index[pending_expired] = -1

        long_body_bin = _bin_index(float(row.C1 - row.O1), BODY_BINS)
        short_body_bin = _bin_index(float(row.O1 - row.C1), BODY_BINS)
        long_gap_bins = [
            _bin_index(float(row.O0) - float(anchor), GAP_BINS)
            for anchor in raw_long_anchors
        ]
        short_gap_bins = [
            _bin_index(float(anchor) - float(row.O0), GAP_BINS)
            for anchor in raw_short_anchors
        ]

        long_triggers = _candidate_coords(long_body_bin, long_gap_bins)
        short_triggers = _candidate_coords(short_body_bin, short_gap_bins)
        both_triggers = long_triggers & short_triggers
        if both_triggers:
            long_triggers -= both_triggers
            short_triggers -= both_triggers

        active_long: list[tuple[int, int, int]] = []
        active_short: list[tuple[int, int, int]] = []
        for coord in sorted(long_triggers):
            if (
                last_entry_raw_index[coord] < raw_index - 1
                and pending_raw_index[coord] < 0
                and not bool(cancel_block[coord])
            ):
                stats.add_trigger_at(coord)
                active_long.append(coord)
        for coord in sorted(short_triggers):
            if (
                last_entry_raw_index[coord] < raw_index - 1
                and pending_raw_index[coord] < 0
                and not bool(cancel_block[coord])
            ):
                stats.add_trigger_at(coord)
                active_short.append(coord)

        for coord in active_long:
            anchor_i = coord[0]
            if (float(int_long_anchors[anchor_i]) - float(row.L0)) >= PENETRATE:
                entry_price = float(int_long_anchors[anchor_i])
                exit_price = float(row.NextOpen)
                trade = costed_points(1, entry_price, exit_price, cost)
                stats.add_trade_at(
                    coord,
                    side=1,
                    raw_points=trade.raw_points,
                    net_points=trade.net_points,
                    fee_twd=trade.fee_twd,
                    tax_twd=trade.tax_twd,
                    slippage_twd=trade.slippage_twd,
                    year_index=year_index,
                    date_int=int(row.DateInt),
                    net_profit_twd=trade.net_profit_twd,
                    entry_time=row.DateTime,
                    exit_time=row.NextDateTime,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    effective_entry=trade.effective_entry,
                    effective_exit=trade.effective_exit,
                )
                last_entry_raw_index[coord] = raw_index
                pending_raw_index[coord] = -1
            else:
                pending_raw_index[coord] = raw_index

        for coord in active_short:
            anchor_i = coord[0]
            if (float(row.H0) - float(int_short_anchors[anchor_i])) >= PENETRATE:
                entry_price = float(int_short_anchors[anchor_i])
                exit_price = float(row.NextOpen)
                trade = costed_points(-1, entry_price, exit_price, cost)
                stats.add_trade_at(
                    coord,
                    side=-1,
                    raw_points=trade.raw_points,
                    net_points=trade.net_points,
                    fee_twd=trade.fee_twd,
                    tax_twd=trade.tax_twd,
                    slippage_twd=trade.slippage_twd,
                    year_index=year_index,
                    date_int=int(row.DateInt),
                    net_profit_twd=trade.net_profit_twd,
                    entry_time=row.DateTime,
                    exit_time=row.NextDateTime,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    effective_entry=trade.effective_entry,
                    effective_exit=trade.effective_exit,
                )
                last_entry_raw_index[coord] = raw_index
                pending_raw_index[coord] = -1
            else:
                pending_raw_index[coord] = raw_index

        if progress_every and (order + 1) % progress_every == 0:
            print(f"Anchor body-gap progress: {order + 1:,}/{len(samples):,}")
    print(f"Anchor body-gap progress: {len(samples):,}/{len(samples):,}")

    summary, by_year = _flatten(stats, years, cost)
    threshold_daily = _threshold_daily(summary, stats.daily_net_twd)
    threshold_trades = _threshold_trades(summary, stats.trade_rows)
    summary_path = outdir / "summary_anchor_body_gap_bins.csv"
    by_year_path = outdir / "by_year_anchor_body_gap_bins.csv"
    daily_path = outdir / "winrate_threshold_daily.csv"
    trades_path = outdir / "winrate_threshold_trades.csv"
    html_path = outdir / "anchor_body_gap_bins_report.html"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")
    threshold_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    threshold_trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    write_html(summary, by_year, html_path, data_report=data_report, cost=cost)
    return {"summary": summary_path, "by_year": by_year_path, "daily": daily_path, "trades": trades_path, "html": html_path}


def _flatten(
    stats: BodyGapStats,
    years: tuple[int, ...],
    cost: CostConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = np.where(np.ones(stats.shape, dtype=bool))
    anchor_idx, body_idx, gap_idx = coords
    anchor_modes = np.asarray(ANCHOR_MODES)
    body_labels = _range_labels(BODY_BINS)
    gap_labels = _range_labels(GAP_BINS)

    rows = pd.DataFrame(
        {
            "RuleID": np.arange(1, len(anchor_idx) + 1),
            "RunID": [f"G{x:04d}" for x in range(1, len(anchor_idx) + 1)],
            "StrategyID": STRATEGY_ID,
            "AnchorMode": anchor_modes[anchor_idx],
            "AnchorID": [f"A{int(anchor_modes[i]):02d}" for i in anchor_idx],
            "AnchorLabel": [ANCHOR_LABELS[int(anchor_modes[i])] for i in anchor_idx],
            "BodyBinIndex": body_idx + 1,
            "BodyBin": [body_labels[i] for i in body_idx],
            "BodyMin": [BODY_BINS[i][0] for i in body_idx],
            "BodyMax": [BODY_BINS[i][1] if BODY_BINS[i][1] is not None else np.nan for i in body_idx],
            "GapBinIndex": gap_idx + 1,
            "GapBin": [gap_labels[i] for i in gap_idx],
            "GapMin": [GAP_BINS[i][0] for i in gap_idx],
            "GapMax": [GAP_BINS[i][1] if GAP_BINS[i][1] is not None else np.nan for i in gap_idx],
            "Penetrate": PENETRATE,
        }
    )

    idx = coords
    count = stats.trades[idx]
    net = stats.net_points[idx]
    rows["TriggerCount"] = stats.triggers[idx]
    rows["FillCount"] = count
    rows["FillRate"] = safe_divide(count, stats.triggers[idx])
    rows["TotalTrades"] = count
    rows["LongTrades"] = stats.long_trades[idx]
    rows["ShortTrades"] = stats.short_trades[idx]
    rows["WinTrades"] = stats.wins[idx]
    rows["LossTrades"] = stats.losses[idx]
    rows["FlatTrades"] = stats.flats[idx]
    rows["WinRate"] = safe_divide(stats.wins[idx], count)
    rows["RawNetPoints"] = stats.raw_points[idx]
    rows["NetPoints"] = net
    rows["NetProfitTWD"] = net * cost.point_value_twd
    rows["TotalReturnRate"] = rows["NetProfitTWD"] / cost.capital_twd
    rows["GrossProfitNetPoints"] = stats.gp[idx]
    rows["GrossLossNetPoints"] = -stats.gl_abs[idx]
    rows["PFNet"] = profit_factor(stats.gp[idx], stats.gl_abs[idx])
    rows["AvgNetPoints"] = safe_divide(net, count)
    rows["MaxDrawdownNetPoints"] = stats.mdd[idx]
    rows["MaxDrawdownRate"] = rows["MaxDrawdownNetPoints"] * cost.point_value_twd / cost.capital_twd
    rows["TotalFeeTWD"] = stats.fees[idx]
    rows["TotalTaxTWD"] = stats.taxes[idx]
    rows["TotalSlippageTWD"] = stats.slippage[idx]

    by_parts = []
    key_cols = [
        "RuleID",
        "RunID",
        "StrategyID",
        "AnchorMode",
        "AnchorID",
        "BodyBinIndex",
        "BodyBin",
        "GapBinIndex",
        "GapBin",
        "Penetrate",
    ]
    for year_i, year in enumerate(years):
        part = rows[key_cols].copy()
        c = stats.year_trades[year_i][idx]
        n = stats.year_net[year_i][idx]
        part["Year"] = year
        part["Trades"] = c
        part["WinRate"] = safe_divide(stats.year_wins[year_i][idx], c)
        part["NetPoints"] = n
        part["NetProfitTWD"] = n * cost.point_value_twd
        part["TotalReturnRate"] = part["NetProfitTWD"] / cost.capital_twd
        part["PFNet"] = profit_factor(stats.year_gp[year_i][idx], stats.year_gl_abs[year_i][idx])
        part["AvgNetPoints"] = safe_divide(n, c)
        part["MaxDrawdownNetPoints"] = stats.year_mdd[year_i][idx]
        by_parts.append(part)

    by_year = pd.concat(by_parts, ignore_index=True)
    return rows, by_year


def _threshold_daily(
    summary: pd.DataFrame,
    daily_net_twd: dict[tuple[int, int, int], float],
) -> pd.DataFrame:
    columns = [
        "Threshold",
        "ThresholdLabel",
        "Date",
        "DailyNetTWD",
        "DailyLongTWD",
        "DailyShortTWD",
        "CumNetTWD",
        "CumLongTWD",
        "CumShortTWD",
    ]
    if not daily_net_twd:
        return pd.DataFrame(columns=columns)

    daily = pd.DataFrame(
        [
            {
                "RuleID": rule_id,
                "Date": date_int,
                "Side": side,
                "NetProfitTWD": net,
            }
            for (rule_id, date_int, side), net in daily_net_twd.items()
        ]
    )

    parts: list[pd.DataFrame] = []
    for threshold in WINRATE_THRESHOLDS:
        selected = _select_by_threshold(summary, threshold)
        if selected.empty:
            continue
        selected_ids = set(int(rule_id) for rule_id in selected["RuleID"])
        part = daily[daily["RuleID"].isin(selected_ids)]
        if part.empty:
            continue
        grouped = (
            part.pivot_table(
                index="Date",
                columns="Side",
                values="NetProfitTWD",
                aggfunc="sum",
                fill_value=0.0,
            )
            .rename(columns={1: "DailyLongTWD", -1: "DailyShortTWD"})
            .reset_index()
        )
        if "DailyLongTWD" not in grouped.columns:
            grouped["DailyLongTWD"] = 0.0
        if "DailyShortTWD" not in grouped.columns:
            grouped["DailyShortTWD"] = 0.0
        grouped = grouped[["Date", "DailyLongTWD", "DailyShortTWD"]].sort_values("Date")
        grouped["DailyNetTWD"] = grouped["DailyLongTWD"] + grouped["DailyShortTWD"]
        grouped["CumNetTWD"] = grouped["DailyNetTWD"].cumsum()
        grouped["CumLongTWD"] = grouped["DailyLongTWD"].cumsum()
        grouped["CumShortTWD"] = grouped["DailyShortTWD"].cumsum()
        grouped.insert(0, "ThresholdLabel", _threshold_label(threshold))
        grouped.insert(0, "Threshold", threshold)
        parts.append(grouped[columns])

    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def _threshold_trades(
    summary: pd.DataFrame,
    trade_rows: list[dict[str, object]],
) -> pd.DataFrame:
    columns = [
        "Threshold",
        "ThresholdLabel",
        "RuleID",
        "RunID",
        "AnchorID",
        "BodyBin",
        "GapBin",
        "Date",
        "EntryTime",
        "ExitTime",
        "Side",
        "SideLabel",
        "EntryPrice",
        "ExitPrice",
        "EffectiveEntry",
        "EffectiveExit",
        "RawPoints",
        "NetPoints",
        "NetProfitTWD",
        "FeeTWD",
        "TaxTWD",
        "SlippageTWD",
        "CumNetProfitTWD",
    ]
    if not trade_rows:
        return pd.DataFrame(columns=columns)

    trades = pd.DataFrame(trade_rows)
    meta = summary[["RuleID", "RunID", "AnchorID", "BodyBin", "GapBin"]].copy()
    trades = trades.merge(meta, on="RuleID", how="left")
    parts: list[pd.DataFrame] = []
    for threshold in WINRATE_THRESHOLDS:
        selected = _select_by_threshold(summary, threshold)
        if selected.empty:
            continue
        selected_ids = set(int(rule_id) for rule_id in selected["RuleID"])
        part = trades[trades["RuleID"].isin(selected_ids)].copy()
        if part.empty:
            continue
        part = part.sort_values(["EntryTime", "ExitTime", "RuleID", "Side"]).reset_index(drop=True)
        part["CumNetProfitTWD"] = part["NetProfitTWD"].cumsum()
        part.insert(0, "ThresholdLabel", _threshold_label(threshold))
        part.insert(0, "Threshold", threshold)
        parts.append(part[columns])

    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def write_html(
    summary: pd.DataFrame,
    by_year: pd.DataFrame,
    output: Path,
    *,
    data_report: object | None = None,
    cost: CostConfig | None = None,
) -> None:
    cost = cost or CostConfig()
    years = sorted(int(y) for y in by_year["Year"].unique())
    year_map = {(int(r.RuleID), int(r.Year)): r for r in by_year.itertuples(index=False)}
    body_labels = _range_labels(BODY_BINS)
    gap_labels = _range_labels(GAP_BINS)

    def fmt_num(v: object, d: int = 1) -> str:
        if v is None or pd.isna(v):
            return ""
        if v == np.inf:
            return "∞"
        value = float(v)
        if d == 0 or abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}"
        return f"{value:,.{d}f}"

    def fmt_pct(v: object, d: int = 1) -> str:
        if v is None or pd.isna(v):
            return ""
        return f"{float(v) * 100:.{d}f}%"

    def cls(v: object) -> str:
        if v is None or pd.isna(v):
            return ""
        value = float(v)
        return "pos" if value > 0 else "neg" if value < 0 else ""

    def sort_num(v: object) -> str:
        if v is None or pd.isna(v):
            return ""
        if v == np.inf:
            return "999999999999"
        return f"{float(v):.10f}"

    def long_formula(row: object) -> str:
        return (
            f"B=C1-O1 in {row.BodyBin}; "
            f"O>=A+{int(row.GapMin)}; "
            f"O<=A+{fmt_num(row.GapMax, 0) if not pd.isna(row.GapMax) else '∞'}; "
            f"L<=A-1"
        )

    def short_formula(row: object) -> str:
        return (
            f"B=O1-C1 in {row.BodyBin}; "
            f"O<=A-{int(row.GapMin)}; "
            f"O>=A-{fmt_num(row.GapMax, 0) if not pd.isna(row.GapMax) else '∞'}; "
            f"H>=A+1"
        )

    summary_by_key = {
        (int(r.AnchorMode), int(r.BodyBinIndex), int(r.GapBinIndex)): r
        for r in summary.itertuples(index=False)
    }

    matrix_sections = []
    for anchor_no in ANCHOR_MODES:
        rows_html = []
        for body_i, body_label in enumerate(body_labels, start=1):
            cells = []
            for gap_i, gap_label in enumerate(gap_labels, start=1):
                row = summary_by_key[(anchor_no, body_i, gap_i)]
                cells.append(
                    f'<td class="{cls(row.TotalReturnRate)}" '
                    f'data-rule="{int(row.RuleID)}">'
                    f'<b>{fmt_pct(row.TotalReturnRate, 2)}</b><br>'
                    f'次 {fmt_num(row.TotalTrades, 0)}<br>'
                    f'MDD {fmt_num(row.MaxDrawdownNetPoints, 0)}<br>'
                    f'勝 {fmt_pct(row.WinRate, 1)}</td>'
                )
            rows_html.append(f"<tr><th>{escape(body_label)}</th>{''.join(cells)}</tr>")
        gap_headers = "".join(f"<th>{escape(label)}</th>" for label in gap_labels)
        matrix_sections.append(
            f'<section class="anchor-block"><h3>A{anchor_no:02d} {escape(ANCHOR_LABELS[anchor_no])}</h3>'
            f'<div class="wrap"><table class="matrix"><thead><tr><th>Body \\ Gap</th>{gap_headers}</tr></thead>'
            f"<tbody>{''.join(rows_html)}</tbody></table></div></section>"
        )

    detail_rows = []
    for row in summary.sort_values("RuleID").itertuples(index=False):
        year_cells = []
        for year in years:
            yr = year_map.get((int(row.RuleID), year))
            if yr is None or int(yr.Trades) == 0:
                year_cells.append('<td class="empty">無</td>')
            else:
                year_cells.append(
                    f'<td class="{cls(yr.TotalReturnRate)}">'
                    f'次 {fmt_num(yr.Trades, 0)}<br>'
                    f'勝 {fmt_pct(yr.WinRate, 1)}<br>'
                    f'報 {fmt_pct(yr.TotalReturnRate, 2)}'
                    "</td>"
                )

        detail_rows.append(
            f'<tr data-anchor="{escape(row.AnchorID)}" data-body="{escape(row.BodyBin)}" '
            f'data-gap="{escape(row.GapBin)}">'
            f'<td data-sort="{int(row.RuleID)}">{int(row.RuleID)}</td>'
            f"<td>{escape(row.RunID)}</td>"
            f"<td>{escape(row.AnchorID)}</td>"
            f'<td class="left">{escape(row.AnchorLabel)}<br>'
            f'<span class="muted">多 A={escape(ANCHOR_LONG_FORMULAS[int(row.AnchorMode)])}；'
            f'空 A={escape(ANCHOR_SHORT_FORMULAS[int(row.AnchorMode)])}</span></td>'
            f"<td>{escape(row.BodyBin)}</td>"
            f"<td>{escape(row.GapBin)}</td>"
            f'<td class="formula long">{escape(long_formula(row))}</td>'
            f'<td class="formula short">{escape(short_formula(row))}</td>'
            + "".join(year_cells)
            + f'<td data-sort="{sort_num(row.TriggerCount)}">{fmt_num(row.TriggerCount, 0)}</td>'
            f'<td data-sort="{sort_num(row.TotalTrades)}">{fmt_num(row.TotalTrades, 0)}</td>'
            f'<td data-sort="{sort_num(row.FillRate)}">{fmt_pct(row.FillRate, 1)}</td>'
            f'<td data-sort="{sort_num(row.WinRate)}">{fmt_pct(row.WinRate, 1)}</td>'
            f'<td class="{cls(row.NetPoints)}" data-sort="{sort_num(row.NetPoints)}">{fmt_num(row.NetPoints, 1)}</td>'
            f'<td class="{cls(row.NetProfitTWD)}" data-sort="{sort_num(row.NetProfitTWD)}">{fmt_num(row.NetProfitTWD, 0)}</td>'
            f'<td class="{cls(row.TotalReturnRate)}" data-sort="{sort_num(row.TotalReturnRate)}">{fmt_pct(row.TotalReturnRate, 2)}</td>'
            f'<td data-sort="{sort_num(row.PFNet)}">{fmt_num(row.PFNet, 2)}</td>'
            f'<td data-sort="{sort_num(row.MaxDrawdownNetPoints)}">{fmt_num(row.MaxDrawdownNetPoints, 1)}</td>'
            f'<td>{fmt_num(row.TotalFeeTWD, 0)}</td>'
            f'<td>{fmt_num(row.TotalTaxTWD, 0)}</td>'
            f'<td>{fmt_num(row.TotalSlippageTWD, 0)}</td>'
            "</tr>"
        )

    data_range = ""
    if data_report is not None:
        data_range = (
            f"資料期間：{escape(str(data_report.datetime_min))} ~ {escape(str(data_report.datetime_max))}；"
            f"有效 K：{fmt_num(getattr(data_report, 'cleaned_rows', 0), 0)} 根"
        )

    year_headers = "".join(f"<th>{year}</th>" for year in years)
    output.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>A01~A08 Body x Gap 11,152 組報表</title>
<style>
body{{font-family:"Microsoft JhengHei",Arial,sans-serif;margin:0;background:#f7faf8;color:#1d2823;font-size:16px}}
header{{padding:18px 10px;background:white;border-bottom:1px solid #dce7e1}}
h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:24px;margin:26px 0 8px}} h3{{font-size:20px;margin:20px 0 6px}}
.sub{{color:#60716a;line-height:1.6;max-width:1500px}} main{{padding:10px 8px}}
.cards{{display:grid;grid-template-columns:repeat(8,minmax(180px,1fr));gap:8px;margin-top:16px}}
.card{{background:#fbfdfc;border:1px solid #dce7e1;border-radius:6px;padding:12px}}
.label{{color:#71817a;font-size:13px}} .value{{font-size:18px;font-weight:800;margin-top:5px}}
.wrap{{overflow:auto;border:1px solid #dce7e1;background:white;margin-bottom:14px;max-height:72vh}}
table{{border-collapse:separate;border-spacing:0;font-size:13px;width:max-content}}
th,td{{border-right:1px solid #dfe8e3;border-bottom:1px solid #e7efea;padding:7px 8px;text-align:right;vertical-align:middle;white-space:nowrap}}
th{{background:#dfece6;position:sticky;top:0;z-index:2}} tbody tr:nth-child(even) td{{background:#f6faf8}}
.matrix th:first-child,.matrix td:first-child{{position:sticky;left:0;z-index:1;background:#eef5f1}}
.matrix td{{min-width:94px;line-height:1.35}} .detail{{min-width:2600px}}
.detail th{{cursor:pointer}} .left{{text-align:left}} .formula{{text-align:left;white-space:normal;min-width:210px;font-family:Consolas,monospace;font-size:12px;line-height:1.35}}
.long{{background:#fff3ef}} .short{{background:#eef9f2}} .pos{{color:#bd3e31;font-weight:800}} .neg{{color:#2f8b58;font-weight:800}}
.empty{{color:#9aa7a0;text-align:center}} .muted{{color:#71817a;font-size:12px}}
.filters{{display:grid;grid-template-columns:repeat(8,minmax(130px,1fr));gap:8px;align-items:end;margin:10px 0}}
.filters input,.filters select,.filters button{{height:34px;border:1px solid #dce7e1;border-radius:5px;background:white;padding:0 8px;font:inherit}}
.filters button{{font-weight:800;color:#2f668a}}
</style>
</head>
<body>
<header>
<h1>A01~A08 第0層 Body x OpenGap 11,152 組報表</h1>
<div class="sub">
{data_range}<br>
公式簡碼：做多 B=C1-O1 in 前K實體區間；O>=A+Gap下限；O<=A+Gap上限；L<=A-1。做空鏡像 B=O1-C1；O<=A-Gap下限；O>=A-Gap上限；H>=A+1。<br>
成本已扣：進場滑點 {cost.entry_slippage_points:g} 點、出場滑點 {cost.exit_slippage_points:g} 點、手續費來回 {cost.round_trip_fee_twd} 元、期交稅單邊 {cost.tax_rate:g}、小台每點 {cost.point_value_twd} 元、本金 {cost.capital_twd:,} 元。
</div>
<div class="cards">
<div class="card"><div class="label">總組合</div><div class="value">{EXPECTED_COMBOS:,} 組</div></div>
<div class="card"><div class="label">Anchor</div><div class="value">8 種</div></div>
<div class="card"><div class="label">前K實體區間</div><div class="value">41 組</div></div>
<div class="card"><div class="label">OpenGap區間</div><div class="value">34 組</div></div>
<div class="card"><div class="label">Penetrate</div><div class="value">固定 1</div></div>
<div class="card"><div class="label">輸出</div><div class="value">{escape(str(output.parent))}</div></div>
<div class="card"><div class="label">CSV</div><div class="value">summary / by_year</div></div>
<div class="card"><div class="label">排序</div><div class="value">點表頭切換</div></div>
</div>
</header>
<main>
<h2>分層總表：Anchor x 前K實體 x OpenGap</h2>
<div class="sub">每個格子顯示總報酬率、成交次數、MDD、勝率。紅色為正報酬，綠色為負報酬。</div>
{''.join(matrix_sections)}
<h2>條列總表：11,152 組</h2>
<div class="filters">
<label><div class="label">搜尋</div><input id="q" placeholder="A01 / 0~10 / 2~4 / 公式"></label>
<label><div class="label">最小成交</div><input id="minTrades" type="number" placeholder="不限"></label>
<label><div class="label">最小總報酬率 %</div><input id="minRet" type="number" step="0.01" placeholder="不限"></label>
<label><div class="label">最小 PF</div><input id="minPf" type="number" step="0.01" placeholder="不限"></label>
<label><div class="label">最大 MDD</div><input id="maxMdd" type="number" step="1" placeholder="不限"></label>
<button id="apply">套用</button>
<button id="clear">清除</button>
<div class="value" id="count"></div>
</div>
<div class="wrap"><table class="detail" id="detail"><thead><tr>
<th>編號</th><th>RunID</th><th>AnchorID</th><th>Anchor</th><th>前K實體</th><th>Gap</th><th>做多公式</th><th>做空公式</th>
{year_headers}<th>觸發</th><th>成交</th><th>成交率</th><th>勝率</th><th>淨點</th><th>淨損益</th><th>總報酬率</th><th>PF</th><th>MDD</th><th>手續費</th><th>期交稅</th><th>滑點成本</th>
</tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div>
</main>
<script>
const table = document.getElementById('detail');
let sortCol = null, sortAsc = true;
function cellValue(row, idx) {{
  const cell = row.children[idx];
  const raw = cell?.dataset?.sort;
  if (raw !== undefined && raw !== '') return Number(raw);
  return (cell?.innerText || '').trim();
}}
function applyFilters() {{
  const q = document.getElementById('q').value.trim().toLowerCase();
  const minTrades = Number(document.getElementById('minTrades').value || -Infinity);
  const minRet = Number(document.getElementById('minRet').value || -Infinity) / 100;
  const minPf = Number(document.getElementById('minPf').value || -Infinity);
  const maxMddInput = document.getElementById('maxMdd').value;
  const maxMdd = maxMddInput === '' ? Infinity : Number(maxMddInput);
  let shown = 0;
  for (const row of table.tBodies[0].rows) {{
    const text = row.innerText.toLowerCase();
    const trades = Number(row.children[row.children.length - 11].dataset.sort || 0);
    const ret = Number(row.children[row.children.length - 6].dataset.sort || 0);
    const pf = Number(row.children[row.children.length - 5].dataset.sort || 0);
    const mdd = Number(row.children[row.children.length - 4].dataset.sort || 0);
    const ok = (!q || text.includes(q)) && trades >= minTrades && ret >= minRet && pf >= minPf && mdd <= maxMdd;
    row.style.display = ok ? '' : 'none';
    if (ok) shown++;
  }}
  document.getElementById('count').textContent = `顯示 ${{shown.toLocaleString()}} / {EXPECTED_COMBOS:,} 組`;
}}
for (const [idx, th] of Array.from(table.tHead.rows[0].cells).entries()) {{
  th.addEventListener('click', () => {{
    if (sortCol === idx) sortAsc = !sortAsc; else {{ sortCol = idx; sortAsc = false; }}
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((a, b) => {{
      const av = cellValue(a, idx), bv = cellValue(b, idx);
      if (typeof av === 'number' && typeof bv === 'number') return sortAsc ? av - bv : bv - av;
      return sortAsc ? String(av).localeCompare(String(bv), 'zh-Hant') : String(bv).localeCompare(String(av), 'zh-Hant');
    }});
    table.tBodies[0].append(...rows);
    applyFilters();
  }});
}}
document.getElementById('apply').addEventListener('click', applyFilters);
document.getElementById('clear').addEventListener('click', () => {{
  for (const id of ['q','minTrades','minRet','minPf','maxMdd']) document.getElementById(id).value = '';
  applyFilters();
}});
for (const id of ['q','minTrades','minRet','minPf','maxMdd']) document.getElementById(id).addEventListener('input', applyFilters);
applyFilters();
</script>
</body>
</html>""",
        encoding="utf-8",
    )
