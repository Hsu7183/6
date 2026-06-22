from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, TextIO


@dataclass(frozen=True)
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Trade:
    index: int
    timestamp: str
    side: str
    anchor: float
    entry: float
    exit: float
    pnl: float
    open: float
    high: float
    low: float
    exit_timestamp: str


AnchorFn = Callable[[Bar, float], float]


def prev_close_anchor(previous: Bar, current_open: float) -> float:
    return previous.close


def prev_open_anchor(previous: Bar, current_open: float) -> float:
    return previous.open


def prev_high_anchor(previous: Bar, current_open: float) -> float:
    return previous.high


def prev_low_anchor(previous: Bar, current_open: float) -> float:
    return previous.low


def prev_hlc3_anchor(previous: Bar, current_open: float) -> float:
    return (previous.high + previous.low + previous.close) / 3.0


def prev_ohlc4_anchor(previous: Bar, current_open: float) -> float:
    return (previous.open + previous.high + previous.low + previous.close) / 4.0


ANCHORS: dict[str, AnchorFn] = {
    "prev_close": prev_close_anchor,
    "prev_open": prev_open_anchor,
    "prev_high": prev_high_anchor,
    "prev_low": prev_low_anchor,
    "prev_hlc3": prev_hlc3_anchor,
    "prev_ohlc4": prev_ohlc4_anchor,
}


def parse_bar(line: str) -> Bar | None:
    parts = line.split()
    if not parts:
        return None
    if len(parts) < 6:
        raise ValueError(f"expected at least 6 columns, got {len(parts)}: {line!r}")

    return Bar(
        timestamp=parts[0],
        open=float(parts[1]),
        high=float(parts[2]),
        low=float(parts[3]),
        close=float(parts[4]),
        volume=float(parts[5]),
    )


def iter_bars(path: Path, *, deduplicate: bool = True) -> Iterator[Bar]:
    previous_bar: Bar | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                bar = parse_bar(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if bar is not None:
                if deduplicate and bar == previous_bar:
                    continue
                previous_bar = bar
                yield bar


def iter_rod_trades(
    bars: Iterable[Bar],
    *,
    open_gap: float,
    penetrate: float,
    anchor_fn: AnchorFn = prev_close_anchor,
) -> Iterator[Trade]:
    if open_gap < 0:
        raise ValueError("open_gap must be >= 0")
    if penetrate < 0:
        raise ValueError("penetrate must be >= 0")

    iterator = iter(bars)
    try:
        previous = next(iterator)
        current = next(iterator)
    except StopIteration:
        return

    for next_index, next_bar in enumerate(iterator, start=2):
        current_index = next_index - 1
        anchor = anchor_fn(previous, current.open)

        # The order decision is made when current.open appears. The ROD order
        # can only fill inside this same bar, then exits at the next bar open.
        if current.open >= anchor + open_gap:
            if current.low <= anchor - penetrate:
                yield Trade(
                    index=current_index,
                    timestamp=current.timestamp,
                    side="long",
                    anchor=anchor,
                    entry=anchor,
                    exit=next_bar.open,
                    pnl=next_bar.open - anchor,
                    open=current.open,
                    high=current.high,
                    low=current.low,
                    exit_timestamp=next_bar.timestamp,
                )
        elif current.open <= anchor - open_gap:
            if current.high >= anchor + penetrate:
                yield Trade(
                    index=current_index,
                    timestamp=current.timestamp,
                    side="short",
                    anchor=anchor,
                    entry=anchor,
                    exit=next_bar.open,
                    pnl=anchor - next_bar.open,
                    open=current.open,
                    high=current.high,
                    low=current.low,
                    exit_timestamp=next_bar.timestamp,
                )

        previous = current
        current = next_bar


def write_trade_header(handle: TextIO) -> None:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "index",
            "timestamp",
            "side",
            "anchor",
            "entry",
            "exit",
            "pnl",
            "open",
            "high",
            "low",
            "exit_timestamp",
        ]
    )


def write_trade(handle: TextIO, trade: Trade) -> None:
    writer = csv.writer(handle)
    writer.writerow(
        [
            trade.index,
            trade.timestamp,
            trade.side,
            trade.anchor,
            trade.entry,
            trade.exit,
            trade.pnl,
            trade.open,
            trade.high,
            trade.low,
            trade.exit_timestamp,
        ]
    )


def run_backtest(
    *,
    input_path: Path,
    open_gap: float,
    penetrate: float,
    anchor_name: str,
    output_path: Path | None,
) -> dict[str, float]:
    anchor_fn = ANCHORS[anchor_name]
    trades = iter_rod_trades(
        iter_bars(input_path),
        open_gap=open_gap,
        penetrate=penetrate,
        anchor_fn=anchor_fn,
    )

    count = 0
    wins = 0
    gross_pnl = 0.0

    if output_path is None:
        for trade in trades:
            count += 1
            wins += int(trade.pnl > 0)
            gross_pnl += trade.pnl
    else:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            write_trade_header(handle)
            for trade in trades:
                write_trade(handle, trade)
                count += 1
                wins += int(trade.pnl > 0)
                gross_pnl += trade.pnl

    return {
        "trades": float(count),
        "wins": float(wins),
        "gross_pnl": gross_pnl,
        "avg_pnl": gross_pnl / count if count else 0.0,
        "win_rate": wins / count if count else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest ROD pullback/pull-up fills from minute OHLCV data."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="FIMTX_M1_202001020845.txt",
        type=Path,
        help="Input text file with columns: timestamp open high low close volume.",
    )
    parser.add_argument("--open-gap", type=float, required=True)
    parser.add_argument("--penetrate", type=float, required=True)
    parser.add_argument(
        "--anchor",
        choices=sorted(ANCHORS),
        default="prev_close",
        help="Anchor formula. Default: previous close.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path for filled trades.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_backtest(
        input_path=args.input,
        open_gap=args.open_gap,
        penetrate=args.penetrate,
        anchor_name=args.anchor,
        output_path=args.output,
    )
    print(f"trades={int(summary['trades'])}")
    print(f"wins={int(summary['wins'])}")
    print(f"gross_pnl={summary['gross_pnl']:.2f}")
    print(f"avg_pnl={summary['avg_pnl']:.4f}")
    print(f"win_rate={summary['win_rate']:.2%}")
    if args.output:
        print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
