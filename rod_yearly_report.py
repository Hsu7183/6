from __future__ import annotations

import argparse
import base64
import html
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import pandas as pd

from mtx_research.data_sources import MTX_FULL_DATA
from rod_backtest import ANCHORS, Bar, iter_bars


@dataclass(frozen=True)
class TradeEconomics:
    raw_points: float
    net_points: float
    net_profit: float
    entry_price: float
    exit_price: float
    fees: float
    tax: float


@dataclass
class Stats:
    trades: int = 0
    wins: int = 0
    raw_points: float = 0.0
    net_points: float = 0.0
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    fees: float = 0.0
    tax: float = 0.0
    equity: float = 0.0
    peak: float = 0.0
    mdd: float = 0.0

    def add(self, economics: TradeEconomics) -> None:
        self.trades += 1
        self.wins += int(economics.net_profit > 0)
        self.raw_points += economics.raw_points
        self.net_points += economics.net_points
        self.net_profit += economics.net_profit
        self.fees += economics.fees
        self.tax += economics.tax

        if economics.net_profit > 0:
            self.gross_profit += economics.net_profit
        elif economics.net_profit < 0:
            self.gross_loss += economics.net_profit

        self.equity += economics.net_profit
        self.peak = max(self.peak, self.equity)
        self.mdd = max(self.mdd, self.peak - self.equity)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def avg_net_profit(self) -> float:
        return self.net_profit / self.trades if self.trades else 0.0

    @property
    def pf(self) -> float:
        if self.gross_loss == 0:
            return math.inf if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)


@dataclass
class PositionState:
    side: str | None = None
    anchor: float = 0.0
    entry_year: str = ""
    entry_index: int = -1


def parse_number_series(raw: str) -> list[float]:
    raw = raw.strip()
    if not raw:
        raise ValueError("number series cannot be empty")

    if ":" in raw:
        parts = [float(part) for part in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("range series must be start:stop:step")
        start, stop, step = parts
        if step <= 0:
            raise ValueError("range step must be > 0")
        values = []
        value = start
        limit = stop + step / 1_000_000
        while value <= limit:
            values.append(round(value, 10))
            value += step
        return values

    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("number series cannot be empty")
    return sorted(dict.fromkeys(values))


def fmt_number(value: float, digits: int = 2) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def describe_number_series(values: list[float]) -> str:
    if not values:
        return ""
    unique_values = sorted(dict.fromkeys(values))
    if len(unique_values) <= 12:
        return ", ".join(fmt_number(value) for value in unique_values)

    steps = [
        round(unique_values[index] - unique_values[index - 1], 10)
        for index in range(1, len(unique_values))
    ]
    if steps and all(math.isclose(step, steps[0]) for step in steps):
        return (
            f"{fmt_number(unique_values[0])} ~ {fmt_number(unique_values[-1])}"
            f"（每 {fmt_number(steps[0])}）"
        )
    return ", ".join(fmt_number(value) for value in unique_values)


def describe_open_gap_bands(open_gap_bands: list[tuple[float, float]]) -> str:
    if not open_gap_bands:
        return ""
    lows = sorted({low for low, _ in open_gap_bands})
    highs = sorted({high for _, high in open_gap_bands})
    return (
        f"OG_L {describe_number_series(lows)}；"
        f"OG_H {describe_number_series(highs)}；"
        f"有效區間 {len(open_gap_bands):,} 組"
    )


def fmt_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def open_gap_band_label(open_gap_low: float, open_gap_high: float) -> str:
    return f"{fmt_number(open_gap_low)}~{fmt_number(open_gap_high)}"


def condition_label(open_gap_low: float, open_gap_high: float, penetrate: float) -> str:
    return f"OG {open_gap_band_label(open_gap_low, open_gap_high)} / P {fmt_number(penetrate)}"


def fmt_hhmm(value: int) -> str:
    text = f"{value:04d}"
    return f"{text[:2]}:{text[2:]}"


def round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def calc_tax(price: float, point_value: float, tax_rate: float, round_tax: bool) -> float:
    tax = price * point_value * tax_rate
    return float(round_half_up(tax)) if round_tax else tax


def compute_trade_economics(
    *,
    side: str,
    anchor: float,
    next_open: float,
    entry_slippage: float,
    exit_slippage: float,
    point_value: float,
    fee_per_side: float,
    tax_rate: float,
    round_tax: bool,
) -> TradeEconomics:
    if side == "long":
        entry_price = anchor + entry_slippage
        exit_price = next_open - exit_slippage
        raw_points = next_open - anchor
        net_points = exit_price - entry_price
    elif side == "short":
        entry_price = anchor - entry_slippage
        exit_price = next_open + exit_slippage
        raw_points = anchor - next_open
        net_points = entry_price - exit_price
    else:
        raise ValueError(f"unsupported side: {side}")

    fees = fee_per_side * 2
    tax = calc_tax(entry_price, point_value, tax_rate, round_tax) + calc_tax(
        exit_price, point_value, tax_rate, round_tax
    )
    net_profit = net_points * point_value - fees - tax
    return TradeEconomics(
        raw_points=raw_points,
        net_points=net_points,
        net_profit=net_profit,
        entry_price=entry_price,
        exit_price=exit_price,
        fees=fees,
        tax=tax,
    )


def hhmm_to_int(raw: str) -> int:
    digits = "".join(char for char in raw if char.isdigit())
    if len(digits) != 4:
        raise ValueError(f"time must be HHMM or HH:MM, got {raw!r}")
    hour = int(digits[:2])
    minute = int(digits[2:])
    if hour > 23 or minute > 59:
        raise ValueError(f"invalid time: {raw!r}")
    return hour * 100 + minute


def bar_date(timestamp: str) -> str:
    return timestamp[:8]


def bar_hhmm(timestamp: str) -> int:
    return int(timestamp[8:12])


def build_force_exit_indices(bars: list[Bar], force_exit: int) -> dict[str, int]:
    force_indices: dict[str, int] = {}
    for index, bar in enumerate(bars):
        date = bar_date(bar.timestamp)
        if date not in force_indices and bar_hhmm(bar.timestamp) >= force_exit:
            force_indices[date] = index
    return force_indices


def choose_exit_index(
    bars: list[Bar],
    *,
    entry_index: int,
    force_exit_indices: dict[str, int],
    force_exit: int,
) -> int | None:
    date = bar_date(bars[entry_index].timestamp)
    next_index = entry_index + 1
    if next_index >= len(bars):
        return None

    next_bar = bars[next_index]
    if bar_date(next_bar.timestamp) == date and bar_hhmm(next_bar.timestamp) <= force_exit:
        return next_index

    force_index = force_exit_indices.get(date)
    if force_index is not None and force_index > entry_index:
        return force_index
    return None


def walk_trade_contexts(bars: Iterable[Bar]) -> Iterator[tuple[Bar, Bar, Bar]]:
    iterator = iter(bars)
    try:
        previous = next(iterator)
        current = next(iterator)
    except StopIteration:
        return

    for next_bar in iterator:
        yield previous, current, next_bar
        previous = current
        current = next_bar


def build_yearly_stats(
    bars: Iterable[Bar],
    *,
    open_gap_bands: list[tuple[float, float]],
    penetrates: list[float],
    anchor_name: str,
    entry_start: int,
    entry_end: int,
    force_exit: int,
    entry_slippage: float,
    exit_slippage: float,
    point_value: float,
    fee_per_side: float,
    tax_rate: float,
    round_tax: bool,
) -> dict[tuple[str, float, float, float, str], Stats]:
    anchor_fn = ANCHORS[anchor_name]
    stats: dict[tuple[str, float, float, float, str], Stats] = defaultdict(Stats)
    bar_list = list(bars)
    open_gap_band_values = sorted(
        dict.fromkeys(
            (float(open_gap_low), float(open_gap_high))
            for open_gap_low, open_gap_high in open_gap_bands
            if open_gap_high >= open_gap_low
        )
    )
    penetrate_values = sorted(dict.fromkeys(penetrates))
    last_entry_index: dict[tuple[float, float, float], int] = {}
    open_positions: list[tuple[float, float, float, PositionState]] = []

    for index in range(1, len(bar_list)):
        previous = bar_list[index - 1]
        current = bar_list[index]
        current_time = bar_hhmm(current.timestamp)

        if open_positions:
            exiting_positions = open_positions
            open_positions = []
            for open_gap_low, open_gap_high, penetrate, state in exiting_positions:
                if state.side is None:
                    continue
                economics = compute_trade_economics(
                    side=state.side,
                    anchor=state.anchor,
                    next_open=current.open,
                    entry_slippage=entry_slippage,
                    exit_slippage=exit_slippage,
                    point_value=point_value,
                    fee_per_side=fee_per_side,
                    tax_rate=tax_rate,
                    round_tax=round_tax,
                )
                for bucket_year in (state.entry_year, "ALL"):
                    stats[(bucket_year, open_gap_low, open_gap_high, penetrate, state.side)].add(economics)
                    stats[(bucket_year, open_gap_low, open_gap_high, penetrate, "all")].add(economics)

        can_enter = entry_start <= current_time <= entry_end and current_time < force_exit
        if not can_enter:
            continue

        anchor = anchor_fn(previous, current.open)

        if current.open >= anchor:
            open_distance = current.open - anchor
            penetration_distance = anchor - current.low
            candidate_side = "long"
        else:
            open_distance = anchor - current.open
            penetration_distance = current.high - anchor
            candidate_side = "short"

        matching_open_gap_bands = [
            (open_gap_low, open_gap_high)
            for open_gap_low, open_gap_high in open_gap_band_values
            if open_gap_low <= open_distance <= open_gap_high
        ]
        penetrate_count = bisect_right(penetrate_values, penetration_distance)
        if not matching_open_gap_bands or penetrate_count == 0:
            continue

        for open_gap_low, open_gap_high in matching_open_gap_bands:
            for penetrate in penetrate_values[:penetrate_count]:
                condition = (open_gap_low, open_gap_high, penetrate)
                if index <= last_entry_index.get(condition, -10) + 1:
                    continue
                last_entry_index[condition] = index
                open_positions.append(
                    (
                        open_gap_low,
                        open_gap_high,
                        penetrate,
                        PositionState(
                            side=candidate_side,
                            anchor=anchor,
                            entry_year=current.timestamp[:4],
                            entry_index=index,
                        ),
                    )
                )

    return stats


def stats_to_dataframe(
    stats: dict[tuple[str, float, float, float, str], Stats],
    *,
    capital: float,
    open_gap_bands: list[tuple[float, float]] | None = None,
    penetrates: list[float] | None = None,
) -> pd.DataFrame:
    all_keys = set(stats)
    if open_gap_bands is not None and penetrates is not None:
        for open_gap_low, open_gap_high in open_gap_bands:
            for penetrate in penetrates:
                all_keys.add(("ALL", open_gap_low, open_gap_high, penetrate, "all"))

    rows = []
    for year, open_gap_low, open_gap_high, penetrate, side in all_keys:
        item = stats.get((year, open_gap_low, open_gap_high, penetrate, side), Stats())
        rows.append(
            {
                "year": year,
                "open_gap": open_gap_low,
                "open_gap_low": open_gap_low,
                "open_gap_high": open_gap_high,
                "penetrate": penetrate,
                "side": side,
                "trades": item.trades,
                "wins": item.wins,
                "win_rate": item.win_rate,
                "raw_points": item.raw_points,
                "net_points": item.net_points,
                "net_profit": item.net_profit,
                "avg_net_profit": item.avg_net_profit,
                "mdd": item.mdd,
                "total_return": item.net_profit / capital if capital else 0.0,
                "pf": item.pf,
                "fees": item.fees,
                "tax": item.tax,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["side", "year", "open_gap_low", "open_gap_high", "penetrate"]
    )


def best_win_rate_by_year(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    usable = df[(df["side"] == "all") & (df["year"] != "ALL") & (df["trades"] >= min_trades)].copy()
    if usable.empty:
        return usable
    usable = usable.sort_values(
        ["year", "win_rate", "trades", "net_profit"],
        ascending=[True, False, False, False],
    )
    return usable.groupby("year", as_index=False).head(1).sort_values("year")


def best_return_by_year(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    usable = df[(df["side"] == "all") & (df["year"] != "ALL") & (df["trades"] >= min_trades)].copy()
    if usable.empty:
        return usable
    usable = usable.sort_values(
        ["year", "total_return", "net_profit", "mdd"],
        ascending=[True, False, False, True],
    )
    return usable.groupby("year", as_index=False).head(1).sort_values("year")


def top_overall_conditions(df: pd.DataFrame, min_trades: int, top_n: int) -> pd.DataFrame:
    usable = df[(df["side"] == "all") & (df["year"] == "ALL") & (df["trades"] >= min_trades)].copy()
    if usable.empty:
        return usable
    return usable.sort_values(
        ["total_return", "net_profit", "mdd"],
        ascending=[False, False, True],
    ).head(top_n)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6f")


def metric_text(metric: str, value: float, trades: int) -> str:
    if metric == "win_rate":
        return f"{value:.0%}\n{trades}"
    if metric == "total_return":
        return f"{value:.0%}\n{trades}"
    if metric == "mdd":
        return f"{fmt_money(value)}\n{trades}"
    return f"{value:.1f}\n{trades}"


def metric_color_bounds(values: list[float], metric: str) -> tuple[float, float]:
    if metric == "win_rate":
        return 0.0, 1.0
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if metric == "total_return":
        low = min(0.0, low)
    if metric == "mdd":
        low = 0.0
    if math.isclose(low, high):
        high = low + 1.0
    return low, high


def save_metric_heatmap_figure(
    df: pd.DataFrame,
    *,
    years: list[str],
    open_gap_bands: list[tuple[float, float]],
    penetrates: list[float],
    metric: str,
    title: str,
    path: Path,
    cmap: str,
) -> None:
    side_df = df[(df["side"] == "all") & (df["year"].isin(years))]
    metric_values = [float(value) for value in side_df[metric].dropna().tolist()]
    vmin, vmax = metric_color_bounds(metric_values, metric)
    rows = math.ceil(len(years) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(16, max(4.5, rows * 4.2)), squeeze=False)
    image = None

    for ax, year in zip(axes.ravel(), years):
        year_df = side_df[side_df["year"] == year]
        band_labels = [open_gap_band_label(low, high) for low, high in open_gap_bands]
        matrix = pd.DataFrame(index=band_labels, columns=penetrates, dtype=float)
        trades = pd.DataFrame(index=band_labels, columns=penetrates, dtype=float)
        for row in year_df.itertuples(index=False):
            band_label = open_gap_band_label(row.open_gap_low, row.open_gap_high)
            matrix.loc[band_label, row.penetrate] = getattr(row, metric)
            trades.loc[band_label, row.penetrate] = row.trades

        image = ax.imshow(matrix.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_title(f"{year} {title}")
        ax.set_xlabel("Penetrate")
        ax.set_ylabel("OpenGap band")
        ax.set_xticks(range(len(penetrates)), [fmt_number(value) for value in penetrates])
        ax.set_yticks(range(len(band_labels)), band_labels)

        for y_index, band_label in enumerate(band_labels):
            for x_index, penetrate in enumerate(penetrates):
                value = matrix.loc[band_label, penetrate]
                count = trades.loc[band_label, penetrate]
                if pd.isna(value):
                    continue
                ax.text(
                    x_index,
                    y_index,
                    metric_text(metric, float(value), int(count)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

    for ax in axes.ravel()[len(years) :]:
        ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.88, top=0.93, bottom=0.05, wspace=0.18, hspace=0.38)
    colorbar_axis = fig.add_axes((0.91, 0.16, 0.018, 0.68))
    fig.colorbar(image, cax=colorbar_axis, label=title)
    fig.suptitle(f"ROD {title} Heatmaps by Year", fontsize=16)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_best_return_figure(best: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    if best.empty:
        ax.text(0.5, 0.5, "No condition reached min trades", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [str(row.year) for row in best.itertuples(index=False)]
        values = [row.total_return for row in best.itertuples(index=False)]
        bars = ax.bar(labels, values, color="#2f7d5c")
        low = min(0.0, min(values) * 1.1)
        high = max(values) * 1.2 if max(values) > 0 else 0.1
        ax.set_ylim(low, high)
        ax.set_ylabel("Total Return")
        ax.set_title("Best Total Return by Year")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        for bar, row in zip(bars, best.itertuples(index=False)):
            label = (
                f"{condition_label(row.open_gap_low, row.open_gap_high, row.penetrate)}\n"
                f"{row.trades:,} trades"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_top_lines_figure(df: pd.DataFrame, top: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    if top.empty:
        ax.text(0.5, 0.5, "No condition reached min trades", ha="center", va="center")
        ax.axis("off")
    else:
        years = sorted(df[(df["side"] == "all") & (df["year"] != "ALL")]["year"].unique())
        yearly = df[(df["side"] == "all") & (df["year"].isin(years))]
        for row in top.itertuples(index=False):
            condition = yearly[
                (yearly["open_gap_low"] == row.open_gap_low)
                & (yearly["open_gap_high"] == row.open_gap_high)
                & (yearly["penetrate"] == row.penetrate)
            ].sort_values("year")
            label = condition_label(row.open_gap_low, row.open_gap_high, row.penetrate)
            ax.plot(condition["year"], condition["total_return"], marker="o", linewidth=2, label=label)
        ax.set_ylabel("Total Return")
        ax.set_title("Top Overall Conditions, Total Return by Year")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def format_signed(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def format_pf(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


COLUMN_LABELS = {
    "year": "年份",
    "condition": "條件",
    "open_gap": "OpenGap 開盤跳空",
    "penetrate": "Penetrate 回踩/回抽",
    "trades": "次數",
    "wins": "獲勝次數",
    "win_rate": "勝率",
    "raw_points": "原始點數",
    "net_points": "淨點數",
    "net_profit": "淨損益",
    "avg_net_profit": "平均損益",
    "mdd": "MDD 最大回撤",
    "total_return": "總報酬率",
    "pf": "PF 獲利因子",
    "fees": "手續費",
    "tax": "期交稅",
}


METRIC_LABELS = {
    "win_rate": "勝率",
    "total_return": "總報酬率",
    "mdd": "MDD 最大回撤",
}


def format_metric_for_heatmap(metric: str, value: float) -> str:
    if metric in {"win_rate", "total_return"}:
        return f"{value:.1%}"
    if metric == "mdd":
        return f"{value:,.0f}"
    return format_signed(value)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def heatmap_cell_style(value: float, *, vmin: float, vmax: float, metric: str) -> str:
    if math.isclose(vmin, vmax):
        ratio = 0.5
    else:
        ratio = clamp((value - vmin) / (vmax - vmin))

    if metric == "mdd":
        hue = 48 + ratio * 78
    else:
        hue = 126 - ratio * 116

    return f"background: hsl({hue:.0f} 72% 84%);"


def html_metric_heatmaps(
    df: pd.DataFrame,
    *,
    years: list[str],
    open_gap_bands: list[tuple[float, float]],
    penetrates: list[float],
    metric: str,
) -> str:
    side_df = df[(df["side"] == "all") & (df["year"].isin(years))]
    metric_values = [float(value) for value in side_df[metric].dropna().tolist()]
    vmin, vmax = metric_color_bounds(metric_values, metric)
    title = METRIC_LABELS[metric]

    year_cards = []
    band_labels = [open_gap_band_label(low, high) for low, high in open_gap_bands]
    for year in years:
        year_df = side_df[side_df["year"] == year]
        matrix = pd.DataFrame(index=band_labels, columns=penetrates, dtype=float)
        trades = pd.DataFrame(index=band_labels, columns=penetrates, dtype=float)
        for row in year_df.itertuples(index=False):
            band_label = open_gap_band_label(row.open_gap_low, row.open_gap_high)
            matrix.loc[band_label, row.penetrate] = getattr(row, metric)
            trades.loc[band_label, row.penetrate] = row.trades

        header_cells = ['<th class="corner">OG區間 \\ P</th>'] + [
            f"<th>{html.escape(fmt_number(value))}</th>" for value in penetrates
        ]
        body_rows = []
        for band_label in band_labels:
            cells = [f'<th class="axis">{html.escape(band_label)}</th>']
            for penetrate in penetrates:
                value = matrix.loc[band_label, penetrate]
                trade_count = trades.loc[band_label, penetrate]
                if pd.isna(value):
                    cells.append('<td class="heat-empty">無資料</td>')
                    continue
                style = heatmap_cell_style(float(value), vmin=vmin, vmax=vmax, metric=metric)
                cells.append(
                    f'<td class="heat-cell" style="{style}">'
                    f'<div class="cell-main">{html.escape(format_metric_for_heatmap(metric, float(value)))}</div>'
                    f'<div class="cell-sub">次數 {int(trade_count):,}</div>'
                    "</td>"
                )
            body_rows.append("<tr>" + "".join(cells) + "</tr>")

        table = (
            '<table class="heatmap-table"><thead><tr>'
            + "".join(header_cells)
            + "</tr></thead><tbody>"
            + "".join(body_rows)
            + "</tbody></table>"
        )
        year_cards.append(
            f'<article class="year-card"><h3>{html.escape(str(year))} 年</h3>{table}</article>'
        )

    return (
        f'<section class="report-section" id="{html.escape(metric)}">'
        f"<h2>{html.escape(title)}年度熱圖</h2>"
        '<p class="section-note">每格上方是指標值，下方是交易次數；列為 OpenGap 區間，欄為 Penetrate。</p>'
        '<div class="year-grid">'
        + "".join(year_cards)
        + "</div></section>"
    )


ANCHOR_CODES = {
    "prev_close": "PC",
    "prev_open": "PO",
    "prev_high": "PH",
    "prev_low": "PL",
    "prev_hlc3": "HLC3",
    "prev_ohlc4": "OHLC4",
}


ANCHOR_EXPRESSIONS = {
    "prev_close": "C[1]",
    "prev_open": "O[1]",
    "prev_high": "H[1]",
    "prev_low": "L[1]",
    "prev_hlc3": "HLC3[1]",
    "prev_ohlc4": "OHLC4[1]",
}


ANCHOR_ZH = {
    "prev_close": "前一根收盤價 C[1]",
    "prev_open": "前一根開盤價 O[1]",
    "prev_high": "前一根最高價 H[1]",
    "prev_low": "前一根最低價 L[1]",
    "prev_hlc3": "前一根 HLC3",
    "prev_ohlc4": "前一根 OHLC4",
}


def rod_kline_svg() -> str:
    return """<svg class="mini-kline" viewBox="0 0 150 58" aria-label="???OHLC???Open???" role="img">
  <line x1="8" y1="31" x2="142" y2="31" stroke="#53645c" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="8" y="28" font-size="7" fill="#53645c">C[1]</text>
  <line x1="43" y1="8" x2="43" y2="50" stroke="#b5362e" stroke-width="2.4"/>
  <rect x="34" y="17" width="18" height="20" rx="2" fill="#f19a8d" stroke="#b5362e" stroke-width="1.4"/>
  <text x="32" y="7" font-size="7" fill="#27352f">?1?K</text>
  <text x="22" y="14" font-size="7" fill="#53645c">H[1]</text>
  <text x="22" y="53" font-size="7" fill="#53645c">L[1]</text>
  <text x="55" y="22" font-size="7" fill="#53645c">O[1]</text>
  <text x="55" y="37" font-size="7" fill="#53645c">C[1]</text>
  <line x1="96" y1="14" x2="128" y2="14" stroke="#b5362e" stroke-width="3.2"/>
  <text x="91" y="8" font-size="7" fill="#b5362e">?? O</text>
  <path d="M112 16 L112 29" stroke="#b5362e" stroke-width="1.8" marker-end="url(#arrowR)"/>
  <text x="94" y="43" font-size="7" fill="#b5362e">????</text>
  <defs>
    <marker id="arrowR" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0 0 L6 3 L0 6 Z" fill="#b5362e"/>
    </marker>
  </defs>
</svg>"""


def formula_pair_html(
    anchor_name: str,
    open_gap_low: float,
    open_gap_high: float,
    penetrate: float,
) -> str:
    anchor = ANCHOR_EXPRESSIONS.get(anchor_name, "Anchor")
    og_low = fmt_number(open_gap_low)
    og_high = fmt_number(open_gap_high)
    pen = fmt_number(penetrate)
    long_lines = [
        f"O >= {anchor} + {og_low}",
        f"AND O <= {anchor} + {og_high}",
        f"AND L <= {anchor} - {pen}",
    ]
    short_lines = [
        f"O <= {anchor} - {og_low}",
        f"AND O >= {anchor} - {og_high}",
        f"AND H >= {anchor} + {pen}",
    ]
    long_formula = "<br>".join(html.escape(line) for line in long_lines)
    short_formula = "<br>".join(html.escape(line) for line in short_lines)
    return (
        '<div class="formula-pair">'
        '<div class="formula-box long-formula">'
        '<div class="formula-title">做多</div>'
        f"<code>{long_formula}</code>"
        "</div>"
        '<div class="formula-box short-formula">'
        '<div class="formula-title">做空</div>'
        f"<code>{short_formula}</code>"
        "</div>"
        "</div>"
    )


def format_ratio(value: float) -> str:
    return f"{value:.1%}"


def format_compact_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}{value / 1_000:.0f}K"
    return f"{sign}{value:.0f}"


def data_number(value: float | int | None) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if math.isnan(numeric):
        return ""
    if math.isinf(numeric):
        return "Infinity" if numeric > 0 else "-Infinity"
    return f"{numeric:.10g}"


def sort_header(label: str, key: str, class_name: str = "") -> str:
    classes = f' class="{class_name}"' if class_name else ""
    return (
        f"<th{classes}>"
        f'<button type="button" class="sort-button" data-sort-key="{html.escape(key)}">'
        f"<span>{html.escape(label)}</span><span class=\"sort-mark\" aria-hidden=\"true\"></span>"
        "</button></th>"
    )


def excel_year_cell(row: pd.Series | None) -> str:
    if row is None:
        return '<td class="year-metrics muted-cell">無資料</td>'
    ret_class = "pos" if float(row["total_return"]) >= 0 else "neg"
    return (
        '<td class="year-metrics">'
        f'<div><span>次</span><b>{int(row["trades"]):,}</b></div>'
        f'<div><span>勝</span><b>{format_ratio(float(row["win_rate"]))}</b></div>'
        f'<div><span>MDD</span><b>{format_compact_money(float(row["mdd"]))}</b></div>'
        f'<div class="{ret_class}"><span>報酬</span><b>{format_ratio(float(row["total_return"]))}</b></div>'
        "</td>"
    )


def html_excel_condition_table(
    df: pd.DataFrame,
    *,
    years: list[str],
    anchor_name: str,
    min_trades: int,
) -> str:
    side_df = df[df["side"] == "all"].copy()
    annual_lookup: dict[tuple[str, float, float, float], pd.Series] = {}
    for _, row in side_df[side_df["year"] != "ALL"].iterrows():
        annual_lookup[
            (
                str(row["year"]),
                float(row["open_gap_low"]),
                float(row["open_gap_high"]),
                float(row["penetrate"]),
            )
        ] = row

    overall = side_df[side_df["year"] == "ALL"].copy()
    if overall.empty:
        return '<p class="empty">沒有可顯示的條件。</p>'

    overall = overall.sort_values(
        ["total_return", "net_profit", "mdd", "trades"],
        ascending=[False, False, True, False],
    )
    kline = rod_kline_svg()

    header = [
        sort_header("編號", "original", "sticky-col col-no"),
        '<th class="sticky-col col-kline">K線簡圖</th>',
        '<th class="sticky-col col-code">公式簡碼</th>',
        sort_header("參數", "param", "sticky-col col-param"),
    ]
    header.extend(sort_header(str(year), f"year-{year}", "year-head") for year in years)
    header.extend(
        [
            sort_header("總次數", "total-trades", "summary-head"),
            sort_header("總勝率", "total-win-rate", "summary-head"),
            sort_header("MDD", "mdd", "summary-head"),
            sort_header("總報酬率", "total-return", "summary-head"),
            sort_header("淨點數", "net-points", "summary-head"),
            sort_header("淨損益", "net-profit", "summary-head"),
            sort_header("PF", "pf", "summary-head"),
            sort_header("手續費", "fees", "summary-head"),
            sort_header("期交稅", "tax", "summary-head"),
        ]
    )

    body_rows = []
    for index, row in enumerate(overall.itertuples(index=False), start=1):
        open_gap_low = float(row.open_gap_low)
        open_gap_high = float(row.open_gap_high)
        penetrate = float(row.penetrate)
        formula_html = formula_pair_html(anchor_name, open_gap_low, open_gap_high, penetrate)
        year_rows = [
            annual_lookup.get((str(year), open_gap_low, open_gap_high, penetrate))
            for year in years
        ]
        year_cells = [excel_year_cell(year_row) for year_row in year_rows]
        row_attrs = [
            f'data-sort-original="{index}"',
            f'data-sort-param="{data_number(open_gap_low * 1000000 + open_gap_high * 1000 + penetrate)}"',
            f'data-filter-open-gap-low="{data_number(open_gap_low)}"',
            f'data-filter-open-gap-high="{data_number(open_gap_high)}"',
            f'data-filter-penetrate="{data_number(penetrate)}"',
            f'data-sort-total-trades="{data_number(row.trades)}"',
            f'data-filter-total-trades="{data_number(row.trades)}"',
            f'data-sort-total-win-rate="{data_number(row.win_rate)}"',
            f'data-filter-total-win-rate="{data_number(row.win_rate)}"',
            f'data-sort-mdd="{data_number(row.mdd)}"',
            f'data-filter-mdd="{data_number(row.mdd)}"',
            f'data-sort-total-return="{data_number(row.total_return)}"',
            f'data-filter-total-return="{data_number(row.total_return)}"',
            f'data-sort-net-points="{data_number(row.net_points)}"',
            f'data-sort-net-profit="{data_number(row.net_profit)}"',
            f'data-sort-pf="{data_number(row.pf)}"',
            f'data-filter-pf="{data_number(row.pf)}"',
            f'data-sort-fees="{data_number(row.fees)}"',
            f'data-sort-tax="{data_number(row.tax)}"',
        ]
        for year, year_row in zip(years, year_rows):
            year_value = None if year_row is None else float(year_row["total_return"])
            row_attrs.append(f'data-sort-year-{html.escape(str(year))}="{data_number(year_value)}"')
        if int(row.trades) < min_trades:
            row_attrs.append("hidden")
        ret_class = "pos" if float(row.total_return) >= 0 else "neg"
        body_rows.append(
            "<tr " + " ".join(row_attrs) + ">"
            f'<td class="sticky-col col-no row-no">{index}</td>'
            f'<td class="sticky-col col-kline">{kline}</td>'
            f'<td class="sticky-col col-code">{formula_html}</td>'
            f'<td class="sticky-col col-param"><b>OG={open_gap_band_label(open_gap_low, open_gap_high)}</b><br><b>P={fmt_number(penetrate)}</b></td>'
            + "".join(year_cells)
            + f'<td class="num summary-num">{int(row.trades):,}</td>'
            + f'<td class="num summary-num">{format_ratio(float(row.win_rate))}</td>'
            + f'<td class="num summary-num">{format_signed(float(row.mdd))}</td>'
            + f'<td class="num summary-num {ret_class}">{format_ratio(float(row.total_return))}</td>'
            + f'<td class="num summary-num">{format_signed(float(row.net_points))}</td>'
            + f'<td class="num summary-num">{format_signed(float(row.net_profit))}</td>'
            + f'<td class="num summary-num">{format_pf(float(row.pf))}</td>'
            + f'<td class="num summary-num">{format_signed(float(row.fees))}</td>'
            + f'<td class="num summary-num">{format_signed(float(row.tax))}</td>'
            + "</tr>"
        )

    return (
        '<div class="condition-toolbar" data-role="condition-toolbar">'
        '<label>最小 OG_L<input type="number" step="1" min="2" data-filter-input="open-gap-low" placeholder="不限"></label>'
        '<label>最大 OG_H<input type="number" step="1" min="2" data-filter-input="open-gap-high-max" placeholder="不限"></label>'
        '<label>最小 P<input type="number" step="1" min="1" data-filter-input="penetrate" placeholder="不限"></label>'
        f'<label>最小總次數<input type="number" step="1" min="0" data-filter-input="total-trades" value="{min_trades}"></label>'
        '<label>最小總勝率 %<input type="number" step="0.1" data-filter-input="total-win-rate" placeholder="不限"></label>'
        '<label>最小總報酬率 %<input type="number" step="0.1" data-filter-input="total-return" placeholder="不限"></label>'
        '<label>最小 PF<input type="number" step="0.01" min="0" data-filter-input="pf" placeholder="不限"></label>'
        '<label>最大 MDD<input type="number" step="1" min="0" data-filter-input="mdd-max" placeholder="不限制"></label>'
        '<button type="button" class="filter-reset" data-filter-reset>清除</button>'
        '<div class="filter-count">顯示 <b data-visible-count>0</b> / <b data-total-count>0</b> 組</div>'
        "</div>"
        '<div class="excel-wrap">'
        '<table class="excel-table" data-role="condition-table"><thead><tr>'
        + "".join(header)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def html_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return '<p class="empty">沒有符合條件的資料。</p>'

    header = "".join(f"<th>{html.escape(COLUMN_LABELS.get(column, column))}</th>" for column in columns)
    body_rows = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        cells = []
        for column in columns:
            value = row_dict[column]
            if column in {"win_rate", "total_return"}:
                text = format_percent(float(value))
            elif column in {"net_profit", "avg_net_profit", "mdd", "fees", "tax"}:
                text = format_signed(float(value))
            elif column in {"raw_points", "net_points"}:
                text = format_signed(float(value))
            elif column in {"open_gap", "penetrate"}:
                text = fmt_number(float(value))
            elif column in {"trades", "wins"}:
                text = f"{int(value):,}"
            elif column == "pf":
                text = format_pf(float(value))
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def table_with_condition(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if output.empty:
        return output
    output = output.assign(
        condition=[
            condition_label(row.open_gap_low, row.open_gap_high, row.penetrate)
            for row in output.itertuples(index=False)
        ]
    )
    return output


def save_html_report(
    *,
    path: Path,
    input_path: Path,
    anchor_name: str,
    open_gap_bands: list[tuple[float, float]],
    penetrates: list[float],
    min_trades: int,
    capital: float,
    entry_start: int,
    entry_end: int,
    force_exit: int,
    entry_slippage: float,
    exit_slippage: float,
    point_value: float,
    fee_per_side: float,
    tax_rate: float,
    round_tax: bool,
    best_win: pd.DataFrame,
    best_return: pd.DataFrame,
    top: pd.DataFrame,
    win_heatmap_image: Path,
    return_heatmap_image: Path,
    mdd_heatmap_image: Path,
    best_image: Path,
    lines_image: Path,
    csv_path: Path,
    include_heatmaps: bool = True,
) -> None:
    report_dir = path.parent
    rel_csv = csv_path.relative_to(report_dir) if csv_path.is_relative_to(report_dir) else csv_path
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    open_gap_text = describe_open_gap_bands(open_gap_bands)
    penetrate_text = describe_number_series(penetrates)
    tax_rounding = "nearest TWD per side" if round_tax else "exact decimal"

    summary_columns = [
        "year",
        "condition",
        "trades",
        "win_rate",
        "net_points",
        "net_profit",
        "mdd",
        "total_return",
        "pf",
    ]
    top_columns = [
        "condition",
        "trades",
        "win_rate",
        "net_points",
        "net_profit",
        "mdd",
        "total_return",
        "pf",
        "fees",
        "tax",
    ]

    best_win_table = table_with_condition(best_win)
    best_return_table = table_with_condition(best_return)
    top_table = table_with_condition(top)

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROD yearly report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #5f6c66;
      --line: #d9dfdc;
      --panel: #f7f9f8;
      --accent: #2f7d5c;
      --accent-2: #2c5f8f;
    }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft JhengHei", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 30px 0 12px;
      font-size: 20px;
    }}
    p {{
      line-height: 1.6;
      color: var(--muted);
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 20px 0 26px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--panel);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .value {{
      font-size: 15px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 8px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: var(--panel);
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    a {{
      color: var(--accent-2);
      font-weight: 700;
    }}
    .empty {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .table-wrap table {{
      margin-top: 0;
    }}
  </style>
</head>
<body>
  <main>
    <h1>ROD yearly condition report</h1>
    <p>Each filled trade uses entry slippage {fmt_number(entry_slippage)} pt, exit slippage {fmt_number(exit_slippage)} pt, MTX point value {fmt_number(point_value)} TWD, fee {fmt_number(fee_per_side)} TWD per side, and futures transaction tax {tax_rate:g} per side. Win rate, MDD, PF, and return are cost-adjusted.</p>

    <section class="meta">
      <div class="metric"><div class="label">Input</div><div class="value">{html.escape(str(input_path))}</div></div>
      <div class="metric"><div class="label">Anchor</div><div class="value">{html.escape(anchor_name)}</div></div>
      <div class="metric"><div class="label">OpenGap</div><div class="value">{html.escape(open_gap_text)}</div></div>
      <div class="metric"><div class="label">Penetrate</div><div class="value">{html.escape(penetrate_text)}</div></div>
      <div class="metric"><div class="label">Capital for return</div><div class="value">{capital:,.0f} TWD</div></div>
      <div class="metric"><div class="label">Tax rounding</div><div class="value">{html.escape(tax_rounding)}</div></div>
      <div class="metric"><div class="label">Min trades for ranking</div><div class="value">{min_trades:,}</div></div>
      <div class="metric"><div class="label">Generated</div><div class="value">{html.escape(generated)}</div></div>
    </section>

    <h2>Win rate heatmap</h2>
    <img src="{image_data_uri(win_heatmap_image)}" alt="Win rate heatmaps by year">

    <h2>Total return heatmap</h2>
    <img src="{image_data_uri(return_heatmap_image)}" alt="Total return heatmaps by year">

    <h2>MDD heatmap</h2>
    <img src="{image_data_uri(mdd_heatmap_image)}" alt="MDD heatmaps by year">

    <h2>Best total return by year</h2>
    <img src="{image_data_uri(best_image)}" alt="Best total return by year">
    <div class="table-wrap">
      {html_table(best_return_table, summary_columns)}
    </div>

    <h2>Best win rate by year</h2>
    <div class="table-wrap">
      {html_table(best_win_table, summary_columns)}
    </div>

    <h2>Top overall conditions</h2>
    <img src="{image_data_uri(lines_image)}" alt="Top conditions by total return over years">
    <div class="table-wrap">
      {html_table(top_table, top_columns)}
    </div>

    <p>Full CSV detail: <a href="{html.escape(str(rel_csv))}">{html.escape(str(rel_csv))}</a></p>
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def save_html_report(
    *,
    path: Path,
    df: pd.DataFrame,
    input_path: Path,
    source_rows: int,
    used_bars: int,
    anchor_name: str,
    open_gap_bands: list[tuple[float, float]],
    penetrates: list[float],
    min_trades: int,
    capital: float,
    entry_start: int,
    entry_end: int,
    force_exit: int,
    entry_slippage: float,
    exit_slippage: float,
    point_value: float,
    fee_per_side: float,
    tax_rate: float,
    round_tax: bool,
    best_win: pd.DataFrame,
    best_return: pd.DataFrame,
    top: pd.DataFrame,
    win_heatmap_image: Path,
    return_heatmap_image: Path,
    mdd_heatmap_image: Path,
    best_image: Path,
    lines_image: Path,
    csv_path: Path,
    include_heatmaps: bool = True,
) -> None:
    report_dir = path.parent
    rel_csv = csv_path.relative_to(report_dir) if csv_path.is_relative_to(report_dir) else csv_path
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    open_gap_text = describe_open_gap_bands(open_gap_bands)
    penetrate_text = describe_number_series(penetrates)
    removed_rows = max(source_rows - used_bars, 0)
    tax_rounding = "單邊四捨五入到元" if round_tax else "保留小數精算"
    anchor_display = f"{anchor_name}（{ANCHOR_ZH.get(anchor_name, '自訂基準價')}）"
    trading_time_text = f"{fmt_hhmm(entry_start)}～{fmt_hhmm(entry_end)}，{fmt_hhmm(force_exit)} 強制平倉"
    years = sorted(df[(df["side"] == "all") & (df["year"] != "ALL")]["year"].unique())

    summary_columns = [
        "year",
        "condition",
        "trades",
        "win_rate",
        "net_points",
        "net_profit",
        "mdd",
        "total_return",
        "pf",
    ]
    top_columns = [
        "condition",
        "trades",
        "win_rate",
        "net_points",
        "net_profit",
        "mdd",
        "total_return",
        "pf",
        "fees",
        "tax",
    ]

    best_win_table = table_with_condition(best_win)
    best_return_table = table_with_condition(best_return)
    top_table = table_with_condition(top)
    if include_heatmaps:
        heatmap_html = "\n".join(
            [
                html_metric_heatmaps(
                    df,
                    years=years,
                    open_gap_bands=open_gap_bands,
                    penetrates=penetrates,
                    metric="win_rate",
                ),
                html_metric_heatmaps(
                    df,
                    years=years,
                    open_gap_bands=open_gap_bands,
                    penetrates=penetrates,
                    metric="total_return",
                ),
                html_metric_heatmaps(
                    df,
                    years=years,
                    open_gap_bands=open_gap_bands,
                    penetrates=penetrates,
                    metric="mdd",
                ),
            ]
        )
    else:
        heatmap_html = (
            '<p class="empty">條件組合較多，已略過輔助熱圖；'
            '請以條列總表與 CSV 明細為主。</p>'
        )
    excel_table = html_excel_condition_table(
        df,
        years=years,
        anchor_name=anchor_name,
        min_trades=min_trades,
    )

    table_script = """
  <script>
  (() => {
    const table = document.querySelector('[data-role="condition-table"]');
    const toolbar = document.querySelector('[data-role="condition-toolbar"]');
    if (!table || !toolbar || !table.tBodies.length) return;

    const tbody = table.tBodies[0];
    const sortButtons = Array.from(table.querySelectorAll('[data-sort-key]'));
    const inputs = Array.from(toolbar.querySelectorAll('[data-filter-input]'));
    const resetButton = toolbar.querySelector('[data-filter-reset]');
    const visibleCount = toolbar.querySelector('[data-visible-count]');
    const totalCount = toolbar.querySelector('[data-total-count]');
    let activeSort = { key: null, direction: null };

    const parseNumber = (value) => {
      if (value === null || value === undefined || value === '') return null;
      const parsed = Number(value);
      return Number.isNaN(parsed) ? null : parsed;
    };

    const rowValue = (row, prefix, key) => parseNumber(row.getAttribute(`data-${prefix}-${key}`));
    const originalIndex = (row) => rowValue(row, 'sort', 'original') ?? 0;

    const thresholdValue = (input) => {
      const parsed = parseNumber(input.value.trim());
      if (parsed === null) return null;
      const key = input.getAttribute('data-filter-input');
      if (key === 'total-win-rate' || key === 'total-return') return parsed / 100;
      return parsed;
    };

    const passesFilters = (row) => {
      for (const input of inputs) {
        const limit = thresholdValue(input);
        if (limit === null) continue;

        const key = input.getAttribute('data-filter-input');
        if (key.endsWith('-max')) {
          const value = rowValue(row, 'filter', key.slice(0, -4));
          if (value === null || value > limit) return false;
          continue;
        }

        const value = rowValue(row, 'filter', key);
        if (value === null || value < limit) return false;
      }
      return true;
    };

    const updateRowNumbers = () => {
      let visibleIndex = 1;
      for (const row of Array.from(tbody.rows)) {
        const numberCell = row.querySelector('.row-no');
        if (!row.hidden && numberCell) {
          numberCell.textContent = visibleIndex.toLocaleString('zh-TW');
          visibleIndex += 1;
        }
      }
    };

    const applyFilters = () => {
      let visible = 0;
      const rows = Array.from(tbody.rows);
      for (const row of rows) {
        const show = passesFilters(row);
        row.hidden = !show;
        if (show) visible += 1;
      }
      if (visibleCount) visibleCount.textContent = visible.toLocaleString('zh-TW');
      if (totalCount) totalCount.textContent = rows.length.toLocaleString('zh-TW');
      updateRowNumbers();
    };

    const updateSortMarks = () => {
      for (const button of sortButtons) {
        const mark = button.querySelector('.sort-mark');
        if (!mark) continue;
        const key = button.getAttribute('data-sort-key');
        mark.textContent = key === activeSort.key ? (activeSort.direction === 'desc' ? '▼' : '▲') : '';
      }
    };

    const compareRows = (a, b, key, direction) => {
      const av = rowValue(a, 'sort', key);
      const bv = rowValue(b, 'sort', key);
      if (av === null && bv === null) return originalIndex(a) - originalIndex(b);
      if (av === null) return 1;
      if (bv === null) return -1;

      const diff = direction === 'desc' ? bv - av : av - bv;
      if (diff !== 0) return diff;
      return originalIndex(a) - originalIndex(b);
    };

    const sortBy = (key) => {
      const direction = activeSort.key === key && activeSort.direction === 'desc' ? 'asc' : 'desc';
      activeSort = { key, direction };
      const rows = Array.from(tbody.rows);
      rows.sort((a, b) => compareRows(a, b, key, direction));
      for (const row of rows) tbody.appendChild(row);
      updateSortMarks();
      applyFilters();
    };

    for (const button of sortButtons) {
      button.addEventListener('click', () => sortBy(button.getAttribute('data-sort-key')));
    }

    for (const input of inputs) {
      input.addEventListener('input', applyFilters);
    }

    if (resetButton) {
      resetButton.addEventListener('click', () => {
        for (const input of inputs) input.value = '';
        applyFilters();
      });
    }

    applyFilters();
  })();
  </script>
"""

    content = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROD 年度條件報表</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #586760;
      --line: #cfd8d4;
      --panel: #f7f9f8;
      --panel-strong: #eef3f1;
      --accent: #2f7d5c;
      --accent-2: #245f8f;
      --shadow: 0 10px 26px rgba(23, 32, 27, 0.08);
    }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft JhengHei", "Noto Sans TC", sans-serif;
      color: var(--ink);
      background: #ffffff;
      font-size: 18px;
    }}
    main {{
      width: 100%;
      max-width: none;
      margin: 0;
      padding: 28px 10px 56px;
      box-sizing: border-box;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 42px;
      line-height: 1.18;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 38px 0 12px;
      font-size: 30px;
      line-height: 1.25;
    }}
    h3 {{
      margin: 0 0 14px;
      font-size: 24px;
      line-height: 1.25;
    }}
    p {{
      line-height: 1.6;
      color: var(--muted);
      font-size: 18px;
    }}
    .lead {{
      max-width: 1320px;
      font-size: 20px;
      color: #415149;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 24px 0;
    }}
    .nav a {{
      display: inline-flex;
      align-items: center;
      min-height: 42px;
      padding: 0 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--accent-2);
      text-decoration: none;
      font-size: 17px;
      font-weight: 700;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 24px 0 30px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 18px;
      background: var(--panel);
    }}
    .label {{
      color: var(--muted);
      font-size: 15px;
      margin-bottom: 7px;
    }}
    .value {{
      font-size: 20px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .report-section {{
      margin-top: 34px;
      padding-top: 6px;
    }}
    .section-note {{
      margin: 0 0 16px;
      font-size: 17px;
    }}
    .year-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(720px, 1fr));
      gap: 22px;
      align-items: start;
    }}
    .year-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 18px;
      box-shadow: var(--shadow);
      overflow-x: auto;
    }}
    .heatmap-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 4px;
      table-layout: fixed;
      font-size: 18px;
      margin: 0;
    }}
    .heatmap-table th,
    .heatmap-table td {{
      border: 0;
      border-radius: 7px;
      text-align: center;
      vertical-align: middle;
    }}
    .heatmap-table th {{
      background: var(--panel-strong);
      color: #33423b;
      padding: 10px 8px;
      font-size: 17px;
      font-weight: 800;
    }}
    .heatmap-table .axis,
    .heatmap-table .corner {{
      width: 76px;
      min-width: 76px;
    }}
    .heat-cell {{
      min-width: 108px;
      height: 74px;
      padding: 8px 7px;
      color: #17201b;
      box-shadow: inset 0 0 0 1px rgba(23, 32, 27, 0.09);
    }}
    .heat-empty {{
      min-width: 108px;
      height: 74px;
      padding: 8px 7px;
      color: var(--muted);
      background: #f2f5f4;
    }}
    .cell-main {{
      font-size: 21px;
      font-weight: 900;
      line-height: 1.1;
    }}
    .cell-sub {{
      margin-top: 7px;
      font-size: 14px;
      font-weight: 700;
      color: #425047;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 18px;
      margin-top: 8px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: var(--panel);
      font-size: 16px;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    a {{
      color: var(--accent-2);
      font-weight: 700;
    }}
    .empty {{
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    .table-wrap table {{
      margin-top: 0;
    }}
    .condition-toolbar {{
      display: flex;
      align-items: end;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 8px;
      font-size: 14px;
    }}
    .condition-toolbar label {{
      display: grid;
      gap: 3px;
      color: var(--muted);
      font-weight: 800;
    }}
    .condition-toolbar input {{
      width: 92px;
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 7px;
      font: inherit;
      font-weight: 800;
      color: var(--text);
      background: #ffffff;
    }}
    .filter-reset {{
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 12px;
      font: inherit;
      font-weight: 900;
      color: var(--accent-2);
      background: #f7faf9;
      cursor: pointer;
    }}
    .filter-count {{
      min-height: 32px;
      display: flex;
      align-items: center;
      color: var(--muted);
      font-weight: 800;
      padding: 0 6px;
    }}
    .excel-wrap {{
      max-height: 76vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      box-shadow: var(--shadow);
      width: calc(100vw - 20px);
    }}
    .excel-table {{
      width: max-content;
      min-width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 14px;
      margin: 0;
    }}
    .excel-table th,
    .excel-table td {{
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: middle;
      background: #ffffff;
    }}
    .excel-table thead th {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: #e8efec;
      color: #24342c;
      font-size: 14px;
      text-align: center;
      white-space: nowrap;
    }}
    .sort-button {{
      width: 100%;
      min-height: 24px;
      border: 0;
      padding: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      font: inherit;
      font-weight: 900;
      color: inherit;
      background: transparent;
      cursor: pointer;
      white-space: nowrap;
    }}
    .sort-button:hover {{
      color: var(--accent-2);
    }}
    .sort-mark {{
      display: inline-block;
      min-width: 12px;
      color: var(--accent-2);
      font-size: 12px;
      line-height: 1;
    }}
    .excel-table tbody tr:nth-child(even) td {{
      background: #fbfcfc;
    }}
    .excel-table tbody tr:hover td {{
      background: #eef6f1;
    }}
    .excel-table .sticky-col {{
      position: sticky;
      z-index: 4;
      background: #ffffff;
    }}
    .excel-table thead .sticky-col {{
      z-index: 8;
      background: #dce8e3;
    }}
    .excel-table .col-no {{
      left: 0;
      min-width: 48px;
      width: 48px;
      text-align: center;
    }}
    .excel-table .col-kline {{
      left: 48px;
      min-width: 132px;
      width: 132px;
      text-align: center;
    }}
    .excel-table .col-code {{
      left: 180px;
      min-width: 348px;
      width: 348px;
      text-align: center;
    }}
    .excel-table .col-param {{
      left: 528px;
      min-width: 86px;
      width: 86px;
      text-align: left;
    }}
    .excel-table .year-head {{
      min-width: 112px;
      width: 112px;
      padding-left: 4px !important;
      padding-right: 4px !important;
    }}
    .year-metrics {{
      min-width: 112px;
      width: 112px;
      line-height: 1.14;
      padding-left: 4px !important;
      padding-right: 4px !important;
    }}
    .year-metrics div {{
      display: flex;
      justify-content: space-between;
      gap: 4px;
      white-space: nowrap;
    }}
    .year-metrics span {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }}
    .year-metrics b {{
      font-size: 12px;
    }}
    .muted-cell {{
      color: var(--muted);
      text-align: center;
    }}
    .row-no {{
      font-size: 16px;
      font-weight: 800;
    }}
    .mini-kline {{
      display: block;
      width: 124px;
      height: 46px;
      margin: 0 auto;
    }}
    .formula-pair {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      min-width: 322px;
    }}
    .formula-box {{
      border: 1px solid #cfd8d4;
      border-radius: 7px;
      padding: 5px 6px;
      text-align: left;
      background: #ffffff;
    }}
    .formula-title {{
      margin-bottom: 3px;
      font-size: 13px;
      font-weight: 900;
    }}
    .formula-box code {{
      display: block;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.28;
      color: #1d2b25;
      white-space: nowrap;
    }}
    .long-formula {{
      border-color: #efb0a8;
      background: #fff3f1;
    }}
    .long-formula .formula-title {{
      color: #b5352c;
    }}
    .short-formula {{
      border-color: #b9d7c5;
      background: #f0f8f3;
    }}
    .short-formula .formula-title {{
      color: #15734d;
    }}
    .num {{
      min-width: 96px;
      text-align: right;
      white-space: nowrap;
      font-weight: 700;
    }}
    .summary-head {{
      min-width: 72px;
      width: 72px;
      padding-left: 5px !important;
      padding-right: 5px !important;
    }}
    .summary-num {{
      min-width: 72px;
      width: 72px;
      padding-left: 5px !important;
      padding-right: 5px !important;
      font-size: 13px;
    }}
    .pos {{
      color: #b5352c;
    }}
    .neg {{
      color: #0f7b4c;
    }}
    details.visual-details {{
      margin-top: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 0 18px 18px;
    }}
    details.visual-details summary {{
      cursor: pointer;
      padding: 18px 0;
      font-size: 22px;
      font-weight: 800;
      color: var(--accent-2);
    }}
    .fine-print {{
      margin-top: 30px;
      font-size: 16px;
    }}
    @media (max-width: 840px) {{
      main {{
        padding: 20px 6px 44px;
      }}
      h1 {{
        font-size: 32px;
      }}
      .lead {{
        font-size: 18px;
      }}
      .year-grid {{
        grid-template-columns: 1fr;
      }}
      .year-card {{
        padding: 12px;
      }}
      .heatmap-table {{
        min-width: 760px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>ROD 年度條件報表</h1>
    <p class="lead">每筆成交都用進場滑點 {fmt_number(entry_slippage)} 點、出場滑點 {fmt_number(exit_slippage)} 點、小台每點 {fmt_number(point_value)} 元、單邊手續費 {fmt_number(fee_per_side)} 元，以及單邊期交稅 {tax_rate:g} 計算。只允許 {html.escape(trading_time_text)}；進出場採 XS 式順序：每根 K 先強制平倉、再一般出場、最後才判斷新進場，單根不反手也不出後立刻進。勝率、MDD、PF 與總報酬率都已扣除成本。</p>

    <nav class="nav" aria-label="報表導覽">
      <a href="#condition-list">條列總表</a>
      <a href="#best-return">年度最佳總報酬</a>
      <a href="#best-win">年度最佳勝率</a>
      <a href="#top-overall">整體最佳條件</a>
      <a href="#visual-heatmaps">輔助熱圖</a>
    </nav>

    <section class="meta">
      <div class="metric"><div class="label">資料檔</div><div class="value">{html.escape(str(input_path))}</div></div>
      <div class="metric"><div class="label">有效 K 棒</div><div class="value">{used_bars:,} 根</div></div>
      <div class="metric"><div class="label">去除重複列</div><div class="value">{removed_rows:,} 列</div></div>
      <div class="metric"><div class="label">Anchor（基準價 / 錨點）</div><div class="value">{html.escape(anchor_display)}</div></div>
      <div class="metric"><div class="label">OpenGap（開盤跳空區間）</div><div class="value">{html.escape(open_gap_text)}</div></div>
      <div class="metric"><div class="label">Penetrate（回踩 / 回抽穿越）</div><div class="value">{html.escape(penetrate_text)}</div></div>
      <div class="metric"><div class="label">報酬率本金</div><div class="value">{capital:,.0f} 元</div></div>
      <div class="metric"><div class="label">期交稅取整</div><div class="value">{html.escape(tax_rounding)}</div></div>
      <div class="metric"><div class="label">交易時間 / 強制平倉</div><div class="value">{html.escape(trading_time_text)}</div></div>
      <div class="metric"><div class="label">排名最低次數</div><div class="value">{min_trades:,}</div></div>
      <div class="metric"><div class="label">產生時間</div><div class="value">{html.escape(generated)}</div></div>
    </section>

    <section class="report-section" id="condition-list">
      <h2>條列總表</h2>
      <p class="section-note">每列是一組條件，依整體總報酬率由高到低排列。左側四欄會固定，往右捲可以看每個年度與整體彙總。</p>
      {excel_table}
    </section>

    <section class="report-section" id="best-return">
      <h2>年度最佳總報酬</h2>
      <div class="table-wrap">
        {html_table(best_return_table, summary_columns)}
      </div>
    </section>

    <section class="report-section" id="best-win">
      <h2>年度最佳勝率</h2>
      <div class="table-wrap">
        {html_table(best_win_table, summary_columns)}
      </div>
    </section>

    <section class="report-section" id="top-overall">
      <h2>整體最佳條件（依總報酬率排序）</h2>
      <div class="table-wrap">
        {html_table(top_table, top_columns)}
      </div>
    </section>

    <details class="visual-details" id="visual-heatmaps">
      <summary>輔助熱圖</summary>
      {heatmap_html}
    </details>

    <p class="fine-print">完整 CSV 明細：<a href="{html.escape(str(rel_csv))}">{html.escape(str(rel_csv))}</a>。本報表只沿用你提供的 XS 進出場時序，未套用該指標板的 Mode 與 L2 濾網。MDD 是扣除滑點、手續費與期交稅後的逐筆資金曲線最大回撤；PF 是 Profit Factor（總獲利 / 總虧損絕對值）。</p>
{table_script}
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate yearly ROD condition charts and cost-adjusted metrics.")
    parser.add_argument(
        "input",
        nargs="?",
        default=MTX_FULL_DATA.path,
        type=Path,
        help="Input text file with columns: timestamp open high low close volume.",
    )
    parser.add_argument(
        "--open-gaps",
        default="2:14:1",
        help="OpenGap lower-bound list or inclusive start:stop:step range. Values must be >= 2. Default: 2:14:1.",
    )
    parser.add_argument(
        "--open-gap-highs",
        default="2:14:1",
        help="OpenGap upper-bound list or inclusive start:stop:step range. Values must be >= 2. Default: 2:14:1.",
    )
    parser.add_argument(
        "--penetrates",
        default="1:100:1",
        help="Comma list or inclusive start:stop:step range. Values must be > 0. Default: 1:100:1.",
    )
    parser.add_argument("--anchor", choices=sorted(ANCHORS), default="prev_close")
    parser.add_argument("--output-dir", type=Path, default=Path("rod_yearly_report"))
    parser.add_argument("--html-path", type=Path, default=Path("rod_yearly_report.html"))
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--capital", type=float, default=250_000)
    parser.add_argument("--entry-start", default="0905")
    parser.add_argument("--entry-end", default="1310")
    parser.add_argument("--force-exit", default="1312")
    parser.add_argument("--entry-slippage", type=float, default=0)
    parser.add_argument("--exit-slippage", type=float, default=2)
    parser.add_argument("--point-value", type=float, default=50)
    parser.add_argument("--fee-per-side", type=float, default=18)
    parser.add_argument("--tax-rate", type=float, default=0.00002)
    parser.add_argument(
        "--no-round-tax",
        dest="round_tax",
        action="store_false",
        help="Use exact decimal tax instead of rounding each side to nearest TWD.",
    )
    parser.set_defaults(round_tax=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    open_gap_lows = parse_number_series(args.open_gaps)
    open_gap_highs = parse_number_series(args.open_gap_highs)
    penetrates = parse_number_series(args.penetrates)
    if any(value < 2 for value in open_gap_lows):
        parser.error("--open-gaps values must be >= 2.")
    if any(value < 2 for value in open_gap_highs):
        parser.error("--open-gap-highs values must be >= 2.")
    open_gap_bands = [
        (open_gap_low, open_gap_high)
        for open_gap_low in open_gap_lows
        for open_gap_high in open_gap_highs
        if open_gap_high >= open_gap_low
    ]
    if not open_gap_bands:
        parser.error("No valid OpenGap bands. At least one OG_H must be >= OG_L.")
    if any(value <= 0 for value in penetrates):
        parser.error("--penetrates values must be > 0 because Penetrate=0 has no pullback/pull-up.")
    entry_start = hhmm_to_int(args.entry_start)
    entry_end = hhmm_to_int(args.entry_end)
    force_exit = hhmm_to_int(args.force_exit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = sum(1 for line in args.input.open("r", encoding="utf-8") if line.split())
    bars = list(iter_bars(args.input))

    stats = build_yearly_stats(
        bars,
        open_gap_bands=open_gap_bands,
        penetrates=penetrates,
        anchor_name=args.anchor,
        entry_start=entry_start,
        entry_end=entry_end,
        force_exit=force_exit,
        entry_slippage=args.entry_slippage,
        exit_slippage=args.exit_slippage,
        point_value=args.point_value,
        fee_per_side=args.fee_per_side,
        tax_rate=args.tax_rate,
        round_tax=args.round_tax,
    )
    df = stats_to_dataframe(
        stats,
        capital=args.capital,
        open_gap_bands=open_gap_bands,
        penetrates=penetrates,
    )
    csv_path = args.output_dir / "yearly_conditions.csv"
    save_csv(df, csv_path)

    years = sorted(df[(df["side"] == "all") & (df["year"] != "ALL")]["year"].unique())
    best_win = best_win_rate_by_year(df, args.min_trades)
    best_return = best_return_by_year(df, args.min_trades)
    top = top_overall_conditions(df, args.min_trades, args.top_n)

    win_heatmap_image = args.output_dir / "win_rate_heatmaps.png"
    return_heatmap_image = args.output_dir / "total_return_heatmaps.png"
    mdd_heatmap_image = args.output_dir / "mdd_heatmaps.png"
    best_image = args.output_dir / "best_total_return_by_year.png"
    lines_image = args.output_dir / "top_conditions_total_return_lines.png"
    index_html_path = args.output_dir / "index.html"
    include_heatmaps = len(open_gap_bands) * len(penetrates) <= 2_500

    if include_heatmaps:
        save_metric_heatmap_figure(
            df,
            years=years,
            open_gap_bands=open_gap_bands,
            penetrates=penetrates,
            metric="win_rate",
            title="Win Rate",
            path=win_heatmap_image,
            cmap="RdYlGn",
        )
        save_metric_heatmap_figure(
            df,
            years=years,
            open_gap_bands=open_gap_bands,
            penetrates=penetrates,
            metric="total_return",
            title="Total Return",
            path=return_heatmap_image,
            cmap="RdYlGn",
        )
        save_metric_heatmap_figure(
            df,
            years=years,
            open_gap_bands=open_gap_bands,
            penetrates=penetrates,
            metric="mdd",
            title="MDD",
            path=mdd_heatmap_image,
            cmap="YlOrRd",
        )
    save_best_return_figure(best_return, best_image)
    save_top_lines_figure(df, top, lines_image)

    for html_path in (args.html_path, index_html_path):
        html_path.parent.mkdir(parents=True, exist_ok=True)
        save_html_report(
            path=html_path,
            df=df,
            input_path=args.input,
            source_rows=source_rows,
            used_bars=len(bars),
            anchor_name=args.anchor,
            open_gap_bands=open_gap_bands,
            penetrates=penetrates,
            min_trades=args.min_trades,
            capital=args.capital,
            entry_start=entry_start,
            entry_end=entry_end,
            force_exit=force_exit,
            entry_slippage=args.entry_slippage,
            exit_slippage=args.exit_slippage,
            point_value=args.point_value,
            fee_per_side=args.fee_per_side,
            tax_rate=args.tax_rate,
            round_tax=args.round_tax,
            best_win=best_win,
            best_return=best_return,
            top=top,
            win_heatmap_image=win_heatmap_image,
            return_heatmap_image=return_heatmap_image,
            mdd_heatmap_image=mdd_heatmap_image,
            best_image=best_image,
            lines_image=lines_image,
            csv_path=csv_path,
            include_heatmaps=include_heatmaps,
        )

    print(f"rows={len(df)}")
    print(f"csv={csv_path}")
    print(f"html={args.html_path}")
    print(f"index={index_html_path}")
    if include_heatmaps:
        print(f"win_heatmap={win_heatmap_image}")
        print(f"return_heatmap={return_heatmap_image}")
        print(f"mdd_heatmap={mdd_heatmap_image}")
    else:
        print("heatmaps=skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
