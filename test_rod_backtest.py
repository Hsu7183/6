from rod_backtest import Bar, iter_bars, iter_rod_trades, prev_close_anchor
from rod_yearly_report import (
    Stats,
    build_yearly_stats,
    compute_trade_economics,
    parse_number_series,
)


def bar(ts, open_, high, low, close, volume=1):
    return Bar(ts, open_, high, low, close, volume)


def test_long_gap_then_same_bar_pullback_fills_at_anchor_and_exits_next_open():
    trades = list(
        iter_rod_trades(
            [
                bar("t0", 99, 101, 98, 100),
                bar("t1", 105, 106, 98, 102),
                bar("t2", 103, 104, 101, 102),
            ],
            open_gap=3,
            penetrate=1,
            anchor_fn=prev_close_anchor,
        )
    )

    assert len(trades) == 1
    assert trades[0].side == "long"
    assert trades[0].entry == 100
    assert trades[0].exit == 103
    assert trades[0].pnl == 3


def test_long_gap_without_same_bar_penetration_does_not_fill():
    trades = list(
        iter_rod_trades(
            [
                bar("t0", 99, 101, 98, 100),
                bar("t1", 105, 106, 99.5, 102),
                bar("t2", 103, 104, 101, 102),
            ],
            open_gap=3,
            penetrate=1,
            anchor_fn=prev_close_anchor,
        )
    )

    assert trades == []


def test_short_gap_then_same_bar_pullup_fills_at_anchor_and_exits_next_open():
    trades = list(
        iter_rod_trades(
            [
                bar("t0", 101, 102, 99, 100),
                bar("t1", 95, 102, 94, 98),
                bar("t2", 97, 99, 96, 98),
            ],
            open_gap=3,
            penetrate=1,
            anchor_fn=prev_close_anchor,
        )
    )

    assert len(trades) == 1
    assert trades[0].side == "short"
    assert trades[0].entry == 100
    assert trades[0].exit == 97
    assert trades[0].pnl == 3


def test_short_gap_without_same_bar_penetration_does_not_fill():
    trades = list(
        iter_rod_trades(
            [
                bar("t0", 101, 102, 99, 100),
                bar("t1", 95, 100.5, 94, 98),
                bar("t2", 97, 99, 96, 98),
            ],
            open_gap=3,
            penetrate=1,
            anchor_fn=prev_close_anchor,
        )
    )

    assert trades == []


def test_parse_number_series_accepts_lists_and_ranges():
    assert parse_number_series("1,2,2,3.5") == [1.0, 2.0, 3.5]
    assert parse_number_series("0:10:5") == [0.0, 5.0, 10.0]


def test_iter_bars_deduplicates_consecutive_identical_rows(tmp_path):
    path = tmp_path / "bars.txt"
    path.write_text(
        "\n".join(
            [
                "202001020845 12045 12048 12040 12046 1952",
                "202001020845 12045 12048 12040 12046 1952",
                "202001020846 12047 12055 12044 12052 1479",
            ]
        ),
        encoding="utf-8",
    )

    bars = list(iter_bars(path))

    assert [item.timestamp for item in bars] == ["202001020845", "202001020846"]


def test_trade_economics_applies_exit_slippage_and_round_turn_fee_for_long():
    economics = compute_trade_economics(
        side="long",
        anchor=100,
        next_open=110,
        entry_slippage=0,
        exit_slippage=2,
        point_value=50,
        fee_per_side=18,
        tax_rate=0,
        round_tax=True,
    )

    assert economics.raw_points == 10
    assert economics.net_points == 8
    assert economics.net_profit == 364


def test_trade_economics_applies_exit_slippage_and_round_turn_fee_for_short():
    economics = compute_trade_economics(
        side="short",
        anchor=100,
        next_open=90,
        entry_slippage=0,
        exit_slippage=2,
        point_value=50,
        fee_per_side=18,
        tax_rate=0,
        round_tax=True,
    )

    assert economics.raw_points == 10
    assert economics.net_points == 8
    assert economics.net_profit == 364


def test_stats_tracks_cost_adjusted_mdd_and_pf():
    stats = Stats()
    for net_profit in [100, -30, -80, 50]:
        economics = compute_trade_economics(
            side="long",
            anchor=100,
            next_open=100,
            entry_slippage=0,
            exit_slippage=0,
            point_value=1,
            fee_per_side=0,
            tax_rate=0,
            round_tax=True,
        )
        object.__setattr__(economics, "net_profit", net_profit)
        stats.add(economics)

    assert stats.trades == 4
    assert stats.mdd == 110
    assert round(stats.pf, 6) == round(150 / 110, 6)


def test_yearly_stats_waits_until_bar_after_exit_before_next_entry():
    stats = build_yearly_stats(
        [
            bar("202001020904", 100, 100, 100, 100),
            bar("202001020905", 100, 101, 99, 100),
            bar("202001020906", 100, 101, 99, 100),
            bar("202001020907", 100, 101, 99, 100),
            bar("202001020908", 100, 101, 99, 100),
        ],
        open_gap_bands=[(0, 0)],
        penetrates=[1],
        anchor_name="prev_close",
        entry_start=905,
        entry_end=1310,
        force_exit=1312,
        entry_slippage=0,
        exit_slippage=0,
        point_value=1,
        fee_per_side=0,
        tax_rate=0,
        round_tax=True,
    )

    assert stats[("2020", 0, 0, 1, "all")].trades == 2


def test_yearly_stats_uses_separate_open_gap_low_and_high_bounds():
    stats = build_yearly_stats(
        [
            bar("202001020904", 99, 101, 98, 100),
            bar("202001020905", 105, 106, 98, 102),
            bar("202001020906", 103, 104, 101, 102),
        ],
        open_gap_bands=[(3, 3), (3, 5), (5, 5)],
        penetrates=[1],
        anchor_name="prev_close",
        entry_start=905,
        entry_end=1310,
        force_exit=1312,
        entry_slippage=0,
        exit_slippage=0,
        point_value=1,
        fee_per_side=0,
        tax_rate=0,
        round_tax=True,
    )

    assert stats[("2020", 3, 5, 1, "all")].trades == 1
    assert stats[("2020", 3, 3, 1, "all")].trades == 0
    assert stats[("2020", 5, 5, 1, "all")].trades == 1
