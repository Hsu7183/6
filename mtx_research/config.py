from __future__ import annotations

from dataclasses import dataclass


STAGE_R2A = "r2a"
STAGE_R2B = "r2b"

STRATEGY_R2A = "R2A_1K_TREND_PULLBACK_ALL_FAMILIES"


@dataclass(frozen=True)
class CostConfig:
    capital_twd: int = 250_000
    point_value_twd: int = 50
    entry_slippage_points: float = 0.0
    exit_slippage_points: float = 2.0
    fee_per_side_twd: int = 18
    tax_rate: float = 0.00002

    @property
    def round_trip_fee_twd(self) -> int:
        return self.fee_per_side_twd * 2

    @property
    def slippage_points(self) -> float:
        return self.entry_slippage_points + self.exit_slippage_points


@dataclass(frozen=True)
class TimeConfig:
    begin_time: int = 90500
    end_time: int = 131000
    force_exit_time: int = 131200


@dataclass(frozen=True)
class ResearchConfig:
    cost: CostConfig = CostConfig()
    time: TimeConfig = TimeConfig()
    break_small_max: int = 8
    batch_size: int = 5000
    r2a_expected_combo_count: int = 3_515_850
    r2b_max_candidates: int = 2000
    top_trade_logs: int = 50


BODY_MIN_LIST = [2, 5, 8, 12, 18, 25, 35]
RANGE_MAX_LIST = [30, 50, 80, 120, 180]
BODY_PCT_MIN_LIST = [35, 45, 55, 65, 75]
BODY_PCT_FLOOR_LIST = [15, 25, 35]
BODY_PCT_CENTER_LIST = [25, 35, 45]
CLOSE_POS_MIN_LIST = [60, 70, 80, 90]
CLOSE_POS_STRONG_LIST = [70, 80, 90, 95]
CLOSE_POS_EFF_LIST = [55, 65, 75]
CLOSE_POS_CENTER_LIST = [55, 65, 75, 85]
EFF_OPP_TAIL_MAX_LIST = [10, 20, 30, 40]
MARU_BODY_PCT_LIST = [60, 70, 80, 90]
TAIL_MAX_LIST = [5, 10, 15, 20]
MAIN_TAIL_MIN_LIST = [10, 20, 30, 40]
CENTER_OFFSET_LIST = [0, 2, 5]

LARGE_RANGE_PAIR_LIST = [
    (20, 50),
    (20, 80),
    (20, 120),
    (20, 180),
    (20, 240),
    (30, 50),
    (30, 80),
    (30, 120),
    (30, 180),
    (30, 240),
    (50, 80),
    (50, 120),
    (50, 180),
    (50, 240),
    (80, 120),
    (80, 180),
    (80, 240),
    (120, 180),
    (120, 240),
    (180, 240),
    (240, 999999),
]
BODY_MIN_LARGE_LIST = [5, 12, 25, 35, 50]

OPEN_GAP_MIN_LIST = [1, 2, 3, 5, 8, 12]
OPEN_GAP_MAX_LIST = [5, 8, 12, 18, 25, 35, 50]
PENETRATE_LIST = [1, 2, 3]

EXEC_MODE_LIST = [
    "X01_INSIDE_C1",
    "X02_BREAK_SMALL_C1",
    "X03_BREAK_SMALL_HL",
    "X04_AUTO_C1_HL",
    "X05_INSIDE_BM1",
]

FAMILY_NAMES = [
    "F01_ATTACK",
    "F02_STRONG_CLOSE",
    "F03_MARUBOZU",
    "F04_BODY_EFFICIENCY",
    "F05_TAIL_SUPPORT_CONTINUATION",
    "F06_BODY_CENTER",
    "F07_LARGE_RANGE_ATTACK",
]

OPEN_CLASSES = ["INSIDE", "BREAK_SMALL", "BREAK_LARGE"]
TIME_SEGMENTS = ["T1", "T2", "T3", "T4"]

R2B_EXIT_BARS_LIST = [1, 2, 3, 5, 8]
R2B_TP_LIST = [8, 12, 18, 25, 35]
R2B_SL_LIST = [6, 8, 12, 18, 25]
R2B_MAX_BARS_LIST = [2, 3, 5, 8]
