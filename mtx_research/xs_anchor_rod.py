from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CostConfig
from .data_loader import load_ohlcv
from .metrics import costed_points, profit_factor, safe_divide


STRATEGY_ID = "XS_ANCHOR_ROD_PULLBACK_16P"
ANCHOR_MODES = list(range(1, 9))
PATTERN_MODES = list(range(1, 17))
PENETRATE_LIST = [1, 2, 3]

# Match naked_anchor_all_combinations_runplan.xlsx:
# GapMin x GapMax, skip invalid GapMax < GapMin. This yields 49 legal pairs.
GAP_MIN_LIST = [1, 2, 3, 5, 8, 10, 15, 20]
GAP_MAX_LIST = [3, 5, 8, 10, 15, 20, 30, 40]
GAP_PAIRS = [(mn, mx) for mn in GAP_MIN_LIST for mx in GAP_MAX_LIST if mx >= mn]
EXPECTED_COMBOS = 18_816


ANCHOR_LABELS = {
    1: "C[1] 前收定錨",
    2: "M[1] 高低中位定錨",
    3: "BM[1] 實體中位定錨",
    4: "前高/前低突破定錨",
    5: "O[1] 前開定錨",
    6: "BodyTop/BodyBot 順向實體邊",
    7: "BodyBot/BodyTop 深回踩實體邊",
    8: "Q75/Q25 分位定錨",
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

PATTERN_LABELS = {
    1: "強力陽線 / 強力陰線",
    2: "光頭光腳陽線 / 陰線",
    3: "收高普通陽線 / 收低普通陰線",
    4: "長下影陽線 / 長上影陰線",
    5: "蜻蜓十字 / 墓碑鏡像",
    6: "中位十字線",
    7: "長雙影小實體",
    8: "小振幅小實體",
    9: "長上影陽線 / 長下影陰線",
    10: "墓碑十字 / 蜻蜓鏡像",
    11: "強力陰線否定多 / 強力陽線否定空",
    12: "長下影陰線 / 長上影陽線反向吸收",
    13: "長上影陰線 / 長下影陽線做空鏡像",
    14: "大振幅大實體",
    15: "大振幅小實體 / 換手",
    16: "低量小振幅",
}

PATTERN_FAMILIES = {
    1: "順勢攻擊型",
    2: "極乾淨攻擊型",
    3: "次強續行型",
    4: "吸收續行/影線",
    5: "長單邊影線吸收型",
    6: "本根Open表態型",
    7: "多空拉鋸表態型",
    8: "雜訊/對照組",
    9: "拒絕被否定型",
    10: "強拒絕被否定型",
    11: "否定反轉型",
    12: "吸收反彈/反壓型",
    13: "順向拒絕型",
    14: "過熱/強趨勢",
    15: "大量換手表態型",
    16: "風控排除/對照組",
}

PATTERN_LONG_TEXT = {
    1: "C[1]>O[1]; BodyRatio>=0.60; ClosePos>=0.75; UpperWickRatio<=0.20",
    2: "C[1]>O[1]; BodyRatio>=0.75; UpperWickRatio<=0.10; LowerWickRatio<=0.10",
    3: "C[1]>O[1]; 0.30<=BodyRatio<=0.60; ClosePos>=0.60",
    4: "C[1]>O[1]; LowerWickRatio>=0.40; ClosePos>=0.60",
    5: "BodyRatio<=0.15; LowerWickRatio>=0.50; UpperWickRatio<=0.20; ClosePos>=0.75",
    6: "BodyRatio<=0.15; 0.40<=ClosePos<=0.60",
    7: "BodyRatio<=0.25; UpperWickRatio>=0.25; LowerWickRatio>=0.25",
    8: "Range<=10; BodyRatio<=0.30",
    9: "C[1]>O[1]; UpperWickRatio>=0.40",
    10: "BodyRatio<=0.15; UpperWickRatio>=0.50; ClosePos<=0.25",
    11: "C[1]<O[1]; BodyRatio>=0.60; ClosePos<=0.25; LowerWickRatio<=0.20",
    12: "C[1]<O[1]; LowerWickRatio>=0.40",
    13: "C[1]<O[1]; UpperWickRatio>=0.40",
    14: "Range>=30; BodyRatio>=0.60",
    15: "Range>=30; BodyRatio<=0.25; VolRatio>=1.20",
    16: "Range<=10; VolRatio<=0.80",
}

PATTERN_SHORT_TEXT = {
    1: "C[1]<O[1]; BodyRatio>=0.60; ClosePos<=0.25; LowerWickRatio<=0.20",
    2: "C[1]<O[1]; BodyRatio>=0.75; UpperWickRatio<=0.10; LowerWickRatio<=0.10",
    3: "C[1]<O[1]; 0.30<=BodyRatio<=0.60; ClosePos<=0.40",
    4: "C[1]<O[1]; UpperWickRatio>=0.40; ClosePos<=0.40",
    5: "BodyRatio<=0.15; UpperWickRatio>=0.50; LowerWickRatio<=0.20; ClosePos<=0.25",
    6: "BodyRatio<=0.15; 0.40<=ClosePos<=0.60",
    7: "BodyRatio<=0.25; UpperWickRatio>=0.25; LowerWickRatio>=0.25",
    8: "Range<=10; BodyRatio<=0.30",
    9: "C[1]<O[1]; LowerWickRatio>=0.40",
    10: "BodyRatio<=0.15; LowerWickRatio>=0.50; ClosePos>=0.75",
    11: "C[1]>O[1]; BodyRatio>=0.60; ClosePos>=0.75; UpperWickRatio<=0.20",
    12: "C[1]>O[1]; UpperWickRatio>=0.40",
    13: "C[1]>O[1]; LowerWickRatio>=0.40",
    14: "Range>=30; BodyRatio>=0.60",
    15: "Range>=30; BodyRatio<=0.25; VolRatio>=1.20",
    16: "Range<=10; VolRatio<=0.80",
}


@dataclass(frozen=True)
class XSParams:
    begin_time: int = 90500
    end_time: int = 131000
    force_exit_time: int = 131200
    mirror_short_anchor: int = 1
    range_min: float = 1
    range_max: float = 999999
    body_min: float = 0
    body_max: float = 999999
    strong_body_pct: float = 60
    strong_close_pct: float = 75
    opp_tail_max_pct: float = 20
    maru_body_pct: float = 75
    maru_tail_max_pct: float = 10
    normal_body_min_pct: float = 30
    normal_body_max_pct: float = 60
    normal_close_pct: float = 60
    hammer_lower_min_pct: float = 40
    hammer_close_pct: float = 60
    dragon_body_max_pct: float = 15
    dragon_main_tail_min_pct: float = 50
    dragon_close_pct: float = 75
    dragon_opp_tail_max_pct: float = 20
    doji_body_max_pct: float = 15
    doji_close_low_pct: float = 40
    doji_close_high_pct: float = 60
    both_tail_body_max_pct: float = 25
    both_tail_min_pct: float = 25
    noise_range_max: float = 10
    noise_body_max_pct: float = 30
    long_upper_tail_min_pct: float = 40
    grave_body_max_pct: float = 15
    grave_main_tail_min_pct: float = 50
    grave_close_pct: float = 25
    big_range_min: float = 30
    big_body_pct: float = 60
    high_turn_body_max_pct: float = 25
    high_turn_vol_ratio_min: float = 120
    low_vol_range_max: float = 10
    low_vol_ratio_max: float = 80
    vol_avg_len: int = 20
    use_vol_filter: int = 0
    vol_ratio_min: float = 0
    vol_ratio_max: float = 999999
    side_mode: int = 0
    dual_signal_mode: int = 0


@dataclass
class DenseStats:
    shape: tuple[int, int, int, int]
    years: tuple[int, ...]

    def __post_init__(self) -> None:
        self.eligible = np.zeros(self.shape, dtype=np.int32)
        self.trades = np.zeros(self.shape, dtype=np.int32)
        self.long_trades = np.zeros(self.shape, dtype=np.int32)
        self.short_trades = np.zeros(self.shape, dtype=np.int32)
        self.wins = np.zeros(self.shape, dtype=np.int32)
        self.losses = np.zeros(self.shape, dtype=np.int32)
        self.flats = np.zeros(self.shape, dtype=np.int32)
        self.raw_points = np.zeros(self.shape, dtype=np.float64)
        self.net_points = np.zeros(self.shape, dtype=np.float64)
        self.gp = np.zeros(self.shape, dtype=np.float64)
        self.gl_abs = np.zeros(self.shape, dtype=np.float64)
        self.fees = np.zeros(self.shape, dtype=np.float64)
        self.taxes = np.zeros(self.shape, dtype=np.float64)
        self.slippage = np.zeros(self.shape, dtype=np.float64)
        self.equity = np.zeros(self.shape, dtype=np.float64)
        self.peak = np.zeros(self.shape, dtype=np.float64)
        self.mdd = np.zeros(self.shape, dtype=np.float64)
        self.losing_streak = np.zeros(self.shape, dtype=np.int32)
        self.max_losing_streak = np.zeros(self.shape, dtype=np.int32)
        year_shape = (len(self.years), *self.shape)
        self.year_trades = np.zeros(year_shape, dtype=np.int32)
        self.year_wins = np.zeros(year_shape, dtype=np.int32)
        self.year_net = np.zeros(year_shape, dtype=np.float64)
        self.year_gp = np.zeros(year_shape, dtype=np.float64)
        self.year_gl_abs = np.zeros(year_shape, dtype=np.float64)
        self.year_equity = np.zeros(year_shape, dtype=np.float64)
        self.year_peak = np.zeros(year_shape, dtype=np.float64)
        self.year_mdd = np.zeros(year_shape, dtype=np.float64)

    def add_eligible(self, mask: np.ndarray) -> None:
        if mask.any():
            self.eligible[mask] += 1

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
        self.year_trades[year_index][mask] += 1
        self.year_net[year_index][mask] += net_points
        self.year_equity[year_index][mask] += net_points
        self.year_peak[year_index][mask] = np.maximum(self.year_peak[year_index][mask], self.year_equity[year_index][mask])
        self.year_mdd[year_index][mask] = np.maximum(
            self.year_mdd[year_index][mask],
            self.year_peak[year_index][mask] - self.year_equity[year_index][mask],
        )
        if net_points > 0:
            self.wins[mask] += 1
            self.gp[mask] += net_points
            self.year_wins[year_index][mask] += 1
            self.year_gp[year_index][mask] += net_points
            self.losing_streak[mask] = 0
        elif net_points < 0:
            loss = -net_points
            self.losses[mask] += 1
            self.gl_abs[mask] += loss
            self.year_gl_abs[year_index][mask] += loss
            self.losing_streak[mask] += 1
            self.max_losing_streak[mask] = np.maximum(self.max_losing_streak[mask], self.losing_streak[mask])
        else:
            self.flats[mask] += 1
            self.losing_streak[mask] = 0


def combo_count() -> int:
    return len(ANCHOR_MODES) * len(PATTERN_MODES) * len(GAP_PAIRS) * len(PENETRATE_LIST)


def _prepare_samples(df: pd.DataFrame, params: XSParams) -> pd.DataFrame:
    work = df.sort_values("datetime").reset_index(drop=True).copy()
    prev = work.shift(1)
    nxt = work.shift(-1)
    in_time = (work["TimeInt"] >= params.begin_time) & (work["TimeInt"] <= params.end_time)
    in_time &= work["TimeInt"] < params.force_exit_time
    same_day = (prev["DateInt"] == work["DateInt"]) & (work["DateInt"] == nxt["DateInt"])
    mask = in_time & same_day & nxt["Open"].notna() & nxt["TimeInt"].le(params.force_exit_time)
    avg_vol = work["Volume"].shift(1).rolling(params.vol_avg_len, min_periods=1).mean()
    out = pd.DataFrame(
        {
            "EntryIndex": work.index.to_numpy()[mask.to_numpy()],
            "ExitIndex": work.index.to_numpy()[mask.to_numpy()] + 1,
            "DateTime": work.loc[mask, "datetime"].to_numpy(),
            "NextDateTime": nxt.loc[mask, "datetime"].to_numpy(),
            "Year": work.loc[mask, "Year"].to_numpy(dtype=np.int16),
            "O1": prev.loc[mask, "Open"].to_numpy(dtype=float),
            "H1": prev.loc[mask, "High"].to_numpy(dtype=float),
            "L1": prev.loc[mask, "Low"].to_numpy(dtype=float),
            "C1": prev.loc[mask, "Close"].to_numpy(dtype=float),
            "V1": prev.loc[mask, "Volume"].to_numpy(dtype=float),
            "AvgVol1": avg_vol.loc[mask].to_numpy(dtype=float),
            "O0": work.loc[mask, "Open"].to_numpy(dtype=float),
            "H0": work.loc[mask, "High"].to_numpy(dtype=float),
            "L0": work.loc[mask, "Low"].to_numpy(dtype=float),
            "NextOpen": nxt.loc[mask, "Open"].to_numpy(dtype=float),
        }
    )
    range1 = out["H1"].to_numpy() - out["L1"].to_numpy()
    body_signed = out["C1"].to_numpy() - out["O1"].to_numpy()
    body_abs = np.abs(body_signed)
    body_top = np.maximum(out["O1"].to_numpy(), out["C1"].to_numpy())
    body_bot = np.minimum(out["O1"].to_numpy(), out["C1"].to_numpy())
    upper = np.maximum(out["H1"].to_numpy() - body_top, 0)
    lower = np.maximum(body_bot - out["L1"].to_numpy(), 0)
    out["Range1"] = range1
    out["BodySigned1"] = body_signed
    out["BodyAbs1"] = body_abs
    out["BodyTop1"] = body_top
    out["BodyBot1"] = body_bot
    out["UpperWick1"] = upper
    out["LowerWick1"] = lower
    out["Mid1"] = (out["H1"].to_numpy() + out["L1"].to_numpy()) / 2
    out["BodyMid1"] = (out["O1"].to_numpy() + out["C1"].to_numpy()) / 2
    out["BodyPct1"] = np.divide(body_abs * 100, range1, out=np.zeros_like(range1), where=range1 > 0)
    out["UpperWickPct1"] = np.divide(upper * 100, range1, out=np.zeros_like(range1), where=range1 > 0)
    out["LowerWickPct1"] = np.divide(lower * 100, range1, out=np.zeros_like(range1), where=range1 > 0)
    out["ClosePosPct1"] = np.divide((out["C1"].to_numpy() - out["L1"].to_numpy()) * 100, range1, out=np.zeros_like(range1), where=range1 > 0)
    out["VolRatioPct1"] = np.divide(out["V1"].to_numpy() * 100, out["AvgVol1"].to_numpy(), out=np.zeros(len(out)), where=out["AvgVol1"].to_numpy() > 0)
    return out.reset_index(drop=True)


def _anchor_values(row: object) -> tuple[np.ndarray, np.ndarray]:
    long_values = np.asarray(
        [
            row.C1,
            row.Mid1,
            row.BodyMid1,
            row.H1,
            row.O1,
            row.BodyTop1,
            row.BodyBot1,
            row.L1 + row.Range1 * 0.75,
        ],
        dtype=float,
    )
    short_values = np.asarray(
        [
            row.C1,
            row.Mid1,
            row.BodyMid1,
            row.L1,
            row.O1,
            row.BodyBot1,
            row.BodyTop1,
            row.L1 + row.Range1 * 0.25,
        ],
        dtype=float,
    )
    return long_values, short_values


def _pattern_flags(row: object, params: XSParams) -> tuple[np.ndarray, np.ndarray]:
    long = np.zeros(len(PATTERN_MODES), dtype=bool)
    short = np.zeros(len(PATTERN_MODES), dtype=bool)
    geometry_ok = (
        row.Range1 >= params.range_min
        and row.Range1 <= params.range_max
        and row.BodyAbs1 >= params.body_min
        and row.BodyAbs1 <= params.body_max
        and row.Range1 > 0
    )
    if params.use_vol_filter:
        geometry_ok = geometry_ok and params.vol_ratio_min <= row.VolRatioPct1 <= params.vol_ratio_max
    if not geometry_ok:
        return long, short

    # P01
    long[0] = row.C1 > row.O1 and row.BodyPct1 >= params.strong_body_pct and row.ClosePosPct1 >= params.strong_close_pct and row.UpperWickPct1 <= params.opp_tail_max_pct
    short[0] = row.C1 < row.O1 and row.BodyPct1 >= params.strong_body_pct and row.ClosePosPct1 <= 100 - params.strong_close_pct and row.LowerWickPct1 <= params.opp_tail_max_pct
    # P02
    long[1] = row.C1 > row.O1 and row.BodyPct1 >= params.maru_body_pct and row.UpperWickPct1 <= params.maru_tail_max_pct and row.LowerWickPct1 <= params.maru_tail_max_pct
    short[1] = row.C1 < row.O1 and row.BodyPct1 >= params.maru_body_pct and row.UpperWickPct1 <= params.maru_tail_max_pct and row.LowerWickPct1 <= params.maru_tail_max_pct
    # P03
    long[2] = row.C1 > row.O1 and params.normal_body_min_pct <= row.BodyPct1 <= params.normal_body_max_pct and row.ClosePosPct1 >= params.normal_close_pct
    short[2] = row.C1 < row.O1 and params.normal_body_min_pct <= row.BodyPct1 <= params.normal_body_max_pct and row.ClosePosPct1 <= 100 - params.normal_close_pct
    # P04
    long[3] = row.C1 > row.O1 and row.LowerWickPct1 >= params.hammer_lower_min_pct and row.ClosePosPct1 >= params.hammer_close_pct
    short[3] = row.C1 < row.O1 and row.UpperWickPct1 >= params.hammer_lower_min_pct and row.ClosePosPct1 <= 100 - params.hammer_close_pct
    # P05
    long[4] = row.BodyPct1 <= params.dragon_body_max_pct and row.LowerWickPct1 >= params.dragon_main_tail_min_pct and row.UpperWickPct1 <= params.dragon_opp_tail_max_pct and row.ClosePosPct1 >= params.dragon_close_pct
    short[4] = row.BodyPct1 <= params.dragon_body_max_pct and row.UpperWickPct1 >= params.dragon_main_tail_min_pct and row.LowerWickPct1 <= params.dragon_opp_tail_max_pct and row.ClosePosPct1 <= 100 - params.dragon_close_pct
    # P06
    both = row.BodyPct1 <= params.doji_body_max_pct and params.doji_close_low_pct <= row.ClosePosPct1 <= params.doji_close_high_pct
    long[5] = both
    short[5] = both
    # P07
    both = row.BodyPct1 <= params.both_tail_body_max_pct and row.UpperWickPct1 >= params.both_tail_min_pct and row.LowerWickPct1 >= params.both_tail_min_pct
    long[6] = both
    short[6] = both
    # P08
    both = row.Range1 <= params.noise_range_max and row.BodyPct1 <= params.noise_body_max_pct
    long[7] = both
    short[7] = both
    # P09
    long[8] = row.C1 > row.O1 and row.UpperWickPct1 >= params.long_upper_tail_min_pct
    short[8] = row.C1 < row.O1 and row.LowerWickPct1 >= params.long_upper_tail_min_pct
    # P10
    long[9] = row.BodyPct1 <= params.grave_body_max_pct and row.UpperWickPct1 >= params.grave_main_tail_min_pct and row.ClosePosPct1 <= params.grave_close_pct
    short[9] = row.BodyPct1 <= params.grave_body_max_pct and row.LowerWickPct1 >= params.grave_main_tail_min_pct and row.ClosePosPct1 >= 100 - params.grave_close_pct
    # P11
    long[10] = row.C1 < row.O1 and row.BodyPct1 >= params.strong_body_pct and row.ClosePosPct1 <= 100 - params.strong_close_pct and row.LowerWickPct1 <= params.opp_tail_max_pct
    short[10] = row.C1 > row.O1 and row.BodyPct1 >= params.strong_body_pct and row.ClosePosPct1 >= params.strong_close_pct and row.UpperWickPct1 <= params.opp_tail_max_pct
    # P12
    long[11] = row.C1 < row.O1 and row.LowerWickPct1 >= params.hammer_lower_min_pct
    short[11] = row.C1 > row.O1 and row.UpperWickPct1 >= params.hammer_lower_min_pct
    # P13
    long[12] = row.C1 < row.O1 and row.UpperWickPct1 >= params.long_upper_tail_min_pct
    short[12] = row.C1 > row.O1 and row.LowerWickPct1 >= params.long_upper_tail_min_pct
    # P14
    long[13] = row.C1 > row.O1 and row.Range1 >= params.big_range_min and row.BodyPct1 >= params.big_body_pct
    short[13] = row.C1 < row.O1 and row.Range1 >= params.big_range_min and row.BodyPct1 >= params.big_body_pct
    # P15
    both = row.Range1 >= params.big_range_min and row.BodyPct1 <= params.high_turn_body_max_pct and row.VolRatioPct1 >= params.high_turn_vol_ratio_min
    long[14] = both
    short[14] = both
    # P16
    both = row.Range1 <= params.low_vol_range_max and row.VolRatioPct1 <= params.low_vol_ratio_max
    long[15] = both
    short[15] = both
    return long, short


def _gap_mask(distance: np.ndarray) -> np.ndarray:
    mins = np.asarray([x[0] for x in GAP_PAIRS], dtype=float)
    maxs = np.asarray([x[1] for x in GAP_PAIRS], dtype=float)
    return (distance[:, None] >= mins[None, :]) & (distance[:, None] <= maxs[None, :])


def scan(data_path: Path, outdir: Path, *, params: XSParams | None = None, cost: CostConfig | None = None, progress_every: int = 50_000) -> dict[str, Path]:
    params = params or XSParams()
    cost = cost or CostConfig()
    if combo_count() != EXPECTED_COMBOS:
        raise RuntimeError(f"combo count {combo_count():,} != {EXPECTED_COMBOS:,}")
    outdir.mkdir(parents=True, exist_ok=True)
    df, _ = load_ohlcv(data_path)
    samples = _prepare_samples(df, params)
    years = tuple(int(y) for y in sorted(samples["Year"].unique()))
    year_to_index = {year: i for i, year in enumerate(years)}
    shape = (len(ANCHOR_MODES), len(PATTERN_MODES), len(GAP_PAIRS), len(PENETRATE_LIST))
    stats = DenseStats(shape, years)
    last_entry_order = np.full(shape, -10_000_000, dtype=np.int32)

    for order, row in enumerate(samples.itertuples(index=False)):
        long_patterns, short_patterns = _pattern_flags(row, params)
        long_anchors, short_anchors = _anchor_values(row)
        pen_values = np.asarray(PENETRATE_LIST, dtype=float)
        year_index = year_to_index[int(row.Year)]

        long_gap = _gap_mask(float(row.O0) - long_anchors)
        short_gap = _gap_mask(short_anchors - float(row.O0))
        long_pen = (long_anchors - float(row.L0))[:, None] >= pen_values[None, :]
        short_pen = (float(row.H0) - short_anchors)[:, None] >= pen_values[None, :]
        long_signal = (
            long_patterns[None, :, None, None]
            & long_gap[:, None, :, None]
            & long_pen[:, None, None, :]
        )
        short_signal = (
            short_patterns[None, :, None, None]
            & short_gap[:, None, :, None]
            & short_pen[:, None, None, :]
        )
        if params.side_mode == 1:
            short_signal[:] = False
        elif params.side_mode == -1:
            long_signal[:] = False
        both = long_signal & short_signal
        if both.any():
            if params.dual_signal_mode == 0:
                long_signal[both] = False
                short_signal[both] = False
            elif params.dual_signal_mode == 1:
                short_signal[both] = False
            elif params.dual_signal_mode == 2:
                long_signal[both] = False

        active_long = long_signal & (last_entry_order < order - 1)
        active_short = short_signal & (last_entry_order < order - 1)
        stats.add_eligible(active_long | active_short)

        for anchor_i in range(len(ANCHOR_MODES)):
            mask = active_long[anchor_i]
            if mask.any():
                trade = costed_points(1, float(long_anchors[anchor_i]), float(row.NextOpen), cost)
                full = np.zeros(shape, dtype=bool)
                full[anchor_i] = mask
                stats.add_trade(full, side=1, raw_points=trade.raw_points, net_points=trade.net_points, fee_twd=trade.fee_twd, tax_twd=trade.tax_twd, slippage_twd=trade.slippage_twd, year_index=year_index)
                last_entry_order[full] = order
            mask = active_short[anchor_i]
            if mask.any():
                trade = costed_points(-1, float(short_anchors[anchor_i]), float(row.NextOpen), cost)
                full = np.zeros(shape, dtype=bool)
                full[anchor_i] = mask
                stats.add_trade(full, side=-1, raw_points=trade.raw_points, net_points=trade.net_points, fee_twd=trade.fee_twd, tax_twd=trade.tax_twd, slippage_twd=trade.slippage_twd, year_index=year_index)
                last_entry_order[full] = order

        if progress_every and (order + 1) % progress_every == 0:
            print(f"XS Anchor ROD progress: {order + 1:,}/{len(samples):,}")
    print(f"XS Anchor ROD progress: {len(samples):,}/{len(samples):,}")

    summary, by_year = _flatten(stats, years, cost)
    summary_path = outdir / "summary_xs_anchor_rod.csv"
    by_year_path = outdir / "by_year_xs_anchor_rod.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_year.to_csv(by_year_path, index=False, encoding="utf-8-sig")
    summary.sort_values("NetProfitTWD", ascending=False).head(500).to_csv(outdir / "top_xs_net.csv", index=False, encoding="utf-8-sig")
    tradable = summary[summary["TotalTrades"] >= 300]
    tradable.sort_values("PFNet", ascending=False, na_position="last").head(500).to_csv(outdir / "top_xs_pf.csv", index=False, encoding="utf-8-sig")
    tradable.sort_values("AvgNetPoints", ascending=False, na_position="last").head(500).to_csv(outdir / "top_xs_avg.csv", index=False, encoding="utf-8-sig")
    robust = tradable[(tradable["NetProfitTWD"] > 0) & (tradable["PFNet"] > 1.05) & (tradable["AvgNetPoints"] > 0) & (tradable["PositiveYears"] >= 4)]
    robust.sort_values(["PFNet", "AvgNetPoints", "NetProfitTWD"], ascending=[False, False, False]).head(500).to_csv(outdir / "top_xs_robust.csv", index=False, encoding="utf-8-sig")
    write_html(summary, by_year, outdir / "xs_anchor_rod_report.html")
    return {"summary": summary_path, "by_year": by_year_path, "html": outdir / "xs_anchor_rod_report.html"}


def _flatten(stats: DenseStats, years: tuple[int, ...], cost: CostConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = np.where(np.ones(stats.shape, dtype=bool))
    anchor_idx, pattern_idx, gap_idx, pen_idx = coords
    rows = pd.DataFrame(
        {
            "RuleID": np.arange(1, len(anchor_idx) + 1),
            "RunID": [f"R{x:05d}" for x in range(1, len(anchor_idx) + 1)],
            "StrategyID": STRATEGY_ID,
            "AnchorMode": np.asarray(ANCHOR_MODES)[anchor_idx],
            "AnchorID": [f"A{int(np.asarray(ANCHOR_MODES)[i]):02d}" for i in anchor_idx],
            "AnchorLabel": [ANCHOR_LABELS[int(np.asarray(ANCHOR_MODES)[i])] for i in anchor_idx],
            "PatternMode": np.asarray(PATTERN_MODES)[pattern_idx],
            "PatternID": [f"P{int(np.asarray(PATTERN_MODES)[i]):02d}" for i in pattern_idx],
            "PatternLabel": [PATTERN_LABELS[int(np.asarray(PATTERN_MODES)[i])] for i in pattern_idx],
            "StrategyFamily": [PATTERN_FAMILIES[int(np.asarray(PATTERN_MODES)[i])] for i in pattern_idx],
            "GapMin": [GAP_PAIRS[i][0] for i in gap_idx],
            "GapMax": [GAP_PAIRS[i][1] for i in gap_idx],
            "Penetrate": np.asarray(PENETRATE_LIST)[pen_idx],
        }
    )
    rows["BaseID"] = rows["AnchorID"] + "×" + rows["PatternID"]
    idx = coords
    count = stats.trades[idx]
    net = stats.net_points[idx]
    rows["EligibleTriggerCount"] = stats.eligible[idx]
    rows["FillCount"] = count
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
    rows["PFNet"] = profit_factor(stats.gp[idx], stats.gl_abs[idx])
    rows["AvgNetPoints"] = safe_divide(net, count)
    rows["MaxDrawdownNetPoints"] = stats.mdd[idx]
    rows["MaxDrawdownRate"] = rows["MaxDrawdownNetPoints"] * cost.point_value_twd / cost.capital_twd
    rows["MaxLosingStreak"] = stats.max_losing_streak[idx]
    rows["TotalFeeTWD"] = stats.fees[idx]
    rows["TotalTaxTWD"] = stats.taxes[idx]
    rows["TotalSlippageTWD"] = stats.slippage[idx]
    year_points = np.vstack([stats.year_net[i][idx] for i in range(len(years))])
    year_counts = np.vstack([stats.year_trades[i][idx] for i in range(len(years))])
    active_year = year_counts > 0
    rows["YearCount"] = active_year.sum(axis=0)
    rows["PositiveYears"] = ((year_points > 0) & active_year).sum(axis=0)
    rows["NegativeYears"] = ((year_points < 0) & active_year).sum(axis=0)
    by_parts = []
    key_cols = ["RuleID", "StrategyID", "AnchorMode", "PatternMode", "GapMin", "GapMax", "Penetrate"]
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


def write_html(summary: pd.DataFrame, by_year: pd.DataFrame, output: Path, limit: int = 1000) -> None:
    top = summary.sort_values("NetProfitTWD", ascending=False).head(limit).copy()
    years = sorted(by_year["Year"].unique())
    year_map = {(int(r.RuleID), int(r.Year)): r for r in by_year[by_year["RuleID"].isin(top["RuleID"])].itertuples(index=False)}

    def fmt_num(v: object, d: int = 1) -> str:
        if v is None or pd.isna(v):
            return ""
        if v == np.inf:
            return "∞"
        if abs(float(v) - round(float(v))) < 1e-9:
            return f"{int(round(float(v))):,}"
        return f"{float(v):,.{d}f}"

    def fmt_pct(v: object, d: int = 1) -> str:
        if v is None or pd.isna(v):
            return ""
        return f"{float(v) * 100:.{d}f}%"

    body = []
    for rank, r in enumerate(top.itertuples(index=False), start=1):
        year_cells = []
        for y in years:
            yr = year_map.get((int(r.RuleID), int(y)))
            if yr is None or int(yr.Trades) == 0:
                year_cells.append('<td class="empty">無</td>')
            else:
                cls = "pos" if yr.NetPoints > 0 else "neg" if yr.NetPoints < 0 else ""
                year_cells.append(f'<td class="{cls}">次 {fmt_num(yr.Trades,0)}<br>勝 {fmt_pct(yr.WinRate)}<br>報 {fmt_pct(yr.TotalReturnRate,2)}</td>')
        cls = "pos" if r.NetProfitTWD > 0 else "neg" if r.NetProfitTWD < 0 else ""
        body.append(
            f"<tr><td>{rank}</td><td>{r.RuleID}</td><td>{r.AnchorMode}<br>{r.AnchorLabel}</td>"
            f"<td>P{int(r.PatternMode):02d}<br>{r.PatternLabel}</td><td>G={r.GapMin}~{r.GapMax}<br>P={r.Penetrate}</td>"
            + "".join(year_cells)
            + f'<td>{fmt_num(r.TotalTrades,0)}</td><td>{fmt_pct(r.WinRate)}</td><td class="{cls}">{fmt_num(r.NetPoints,1)}</td>'
            f'<td class="{cls}">{fmt_num(r.NetProfitTWD,0)}</td><td class="{cls}">{fmt_pct(r.TotalReturnRate,2)}</td><td>{fmt_num(r.PFNet,2)}</td><td>{fmt_num(r.MaxDrawdownNetPoints,1)}</td></tr>'
        )
    headers = "".join(f"<th>{y}</th>" for y in years)
    output.write_text(
        f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>XS Anchor ROD 18,816 組報表</title>
<style>
body{{font-family:"Microsoft JhengHei",Arial,sans-serif;margin:0;background:#f7faf8;color:#1d2823;font-size:18px}}
header{{padding:24px 10px;background:white;border-bottom:1px solid #dce7e1}}h1{{font-size:34px;margin:0 0 8px}}
main{{padding:18px 8px}}.wrap{{overflow:auto;max-height:80vh;border:1px solid #dce7e1;background:white}}
table{{border-collapse:separate;border-spacing:0;min-width:1900px;font-size:15px}}th,td{{border-right:1px solid #e0e8e4;border-bottom:1px solid #e8efeb;padding:7px 8px;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:#dfece6;z-index:2}}tbody tr:nth-child(even) td{{background:#f5faf7}}
.pos{{color:#bd3e31;font-weight:800}}.neg{{color:#2f8b58;font-weight:800}}.empty{{color:#9aa7a0;text-align:center}}
</style><header><h1>XS Anchor ROD Pullback 18,816 組報表</h1>
<div>8 Anchor × 16 Pattern × 49 GapMin/GapMax × 3 Penetrate。績效已扣出場 2 點滑點、來回手續費 36 元與期交稅，本金 250,000 元。</div></header>
<main><div class="wrap"><table><thead><tr><th>編號</th><th>RuleID</th><th>Anchor</th><th>Pattern</th><th>參數</th>{headers}<th>總次數</th><th>勝率</th><th>淨點</th><th>淨損益</th><th>總報酬率</th><th>PF</th><th>MDD</th></tr></thead><tbody>{''.join(body)}</tbody></table></div></main></html>""",
        encoding="utf-8",
    )


def write_html(summary: pd.DataFrame, by_year: pd.DataFrame, output: Path, limit: int | None = None) -> None:
    rows_df = summary.sort_values("RuleID", ascending=True, kind="mergesort").copy()
    if limit is not None:
        rows_df = rows_df.head(limit)
    years = sorted(int(y) for y in by_year["Year"].unique())
    year_map = {(int(r.RuleID), int(r.Year)): r for r in by_year[by_year["RuleID"].isin(rows_df["RuleID"])].itertuples(index=False)}

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

    def metric_cell(value: object, label: str = "", d: int = 1) -> str:
        css = cls(value)
        prefix = f"{escape(label)} " if label else ""
        return f'<span class="{css}">{prefix}{fmt_num(value, d)}</span>'

    def pct_cell(value: object, label: str = "", d: int = 1) -> str:
        css = cls(value)
        prefix = f"{escape(label)} " if label else ""
        return f'<span class="{css}">{prefix}{fmt_pct(value, d)}</span>'

    body = []
    for display_no, r in enumerate(rows_df.itertuples(index=False), start=1):
        anchor_no = int(r.AnchorMode)
        pattern_no = int(r.PatternMode)
        gap_min = int(r.GapMin)
        gap_max = int(r.GapMax)
        penetrate = int(r.Penetrate)
        long_anchor = ANCHOR_LONG_FORMULAS[anchor_no]
        short_anchor = ANCHOR_SHORT_FORMULAS[anchor_no]
        long_formula = (
            f"{PATTERN_LONG_TEXT[pattern_no]}\n"
            f"O >= {long_anchor} + {gap_min}; O <= {long_anchor} + {gap_max}\n"
            f"L <= {long_anchor} - {penetrate}; 進 C1/Anchor；下根 O 出"
        )
        short_formula = (
            f"{PATTERN_SHORT_TEXT[pattern_no]}\n"
            f"O <= {short_anchor} - {gap_min}; O >= {short_anchor} - {gap_max}\n"
            f"H >= {short_anchor} + {penetrate}; 進 C1/Anchor；下根 O 出"
        )
        year_cells = []
        for y in years:
            yr = year_map.get((int(r.RuleID), y))
            if yr is None or int(yr.Trades) == 0:
                year_cells.append('<td class="empty">無</td>')
            else:
                ycls = cls(yr.NetProfitTWD)
                year_cells.append(
                    f'<td class="year {ycls}" data-sort="{float(yr.NetProfitTWD):.6f}">'
                    f'次 {fmt_num(yr.Trades, 0)}<br>'
                    f'勝 {fmt_pct(yr.WinRate, 1)}<br>'
                    f'點 {fmt_num(yr.NetPoints, 1)}<br>'
                    f'報 {fmt_pct(yr.TotalReturnRate, 2)}</td>'
                )
        pf_sort = 999999.0 if r.PFNet == np.inf else float(r.PFNet) if not pd.isna(r.PFNet) else -999999.0
        row_text = " ".join(
            [
                str(r.RunID),
                str(r.BaseID),
                str(r.AnchorID),
                str(r.AnchorLabel),
                str(r.PatternID),
                str(r.PatternLabel),
                str(r.StrategyFamily),
                f"G={gap_min}~{gap_max}",
                f"P={penetrate}",
            ]
        )
        body.append(
            f'<tr data-search="{escape(row_text, quote=True)}" data-anchor="{escape(str(r.AnchorID))}" '
            f'data-pattern="{escape(str(r.PatternID))}" data-trades="{int(r.TotalTrades)}" '
            f'data-return="{float(r.TotalReturnRate) * 100:.6f}" data-pf="{pf_sort:.6f}" '
            f'data-positive-years="{int(r.PositiveYears)}">'
            f'<td data-sort="{display_no}">{display_no:,}</td>'
            f'<td data-sort="{int(r.RuleID)}">{escape(str(r.RunID))}</td>'
            f'<td>{escape(str(r.BaseID))}</td>'
            f'<td>{escape(str(r.AnchorID))}</td>'
            f'<td class="left">{escape(str(r.AnchorLabel))}<br><span class="muted">多 {escape(long_anchor)}｜空 {escape(short_anchor)}</span></td>'
            f'<td>{escape(str(r.PatternID))}</td>'
            f'<td class="left">{escape(str(r.PatternLabel))}<br><span class="muted">{escape(str(r.StrategyFamily))}</span></td>'
            f'<td data-sort="{gap_min}">{gap_min}</td>'
            f'<td data-sort="{gap_max}">{gap_max}</td>'
            f'<td data-sort="{penetrate}">{penetrate}</td>'
            f'<td class="formula long">{escape(long_formula).replace(chr(10), "<br>")}</td>'
            f'<td class="formula short">{escape(short_formula).replace(chr(10), "<br>")}</td>'
            + "".join(year_cells)
            + f'<td data-sort="{int(r.TotalTrades)}">{fmt_num(r.TotalTrades, 0)}</td>'
            f'<td data-sort="{float(r.NetPoints):.6f}" class="{cls(r.NetPoints)}">{fmt_num(r.NetPoints, 1)}</td>'
            f'<td data-sort="{float(r.AvgNetPoints) if not pd.isna(r.AvgNetPoints) else -999999:.6f}" class="{cls(r.AvgNetPoints)}">{fmt_num(r.AvgNetPoints, 2)}</td>'
            f'<td data-sort="{float(r.WinRate):.6f}">{fmt_pct(r.WinRate, 1)}</td>'
            f'<td data-sort="{pf_sort:.6f}">{fmt_num(r.PFNet, 2)}</td>'
            f'<td data-sort="{float(r.MaxDrawdownNetPoints):.6f}">{fmt_num(r.MaxDrawdownNetPoints, 1)}</td>'
            f'<td data-sort="{float(r.MaxDrawdownRate):.6f}">{fmt_pct(r.MaxDrawdownRate, 2)}</td>'
            f'<td data-sort="{float(r.NetProfitTWD):.6f}" class="{cls(r.NetProfitTWD)}">{fmt_num(r.NetProfitTWD, 0)}</td>'
            f'<td data-sort="{float(r.TotalReturnRate):.6f}" class="{cls(r.TotalReturnRate)}">{fmt_pct(r.TotalReturnRate, 2)}</td>'
            f'<td data-sort="{int(r.PositiveYears)}">{int(r.PositiveYears)}</td>'
            f'<td></td><td></td></tr>'
        )

    layer_body = []
    layer_no = 1
    for anchor_no in ANCHOR_MODES:
        anchor_rows = rows_df[rows_df["AnchorMode"] == anchor_no]
        for pattern_idx, pattern_no in enumerate(PATTERN_MODES):
            row_cells = []
            for p_no in PENETRATE_LIST:
                block = anchor_rows[
                    (anchor_rows["PatternMode"] == pattern_no)
                    & (anchor_rows["Penetrate"] == p_no)
                ]
                if block.empty:
                    row_cells.append(f'<td colspan="5" class="empty p{p_no}">無</td>')
                    continue
                best = block.sort_values(
                    ["TotalReturnRate", "TotalTrades", "PFNet"],
                    ascending=[False, False, False],
                    na_position="last",
                    kind="mergesort",
                ).iloc[0]
                best_cls = cls(best.NetProfitTWD)
                row_cells.append(
                    f'<td class="p{p_no}">G={int(best.GapMin)}~{int(best.GapMax)}<br>'
                    f'<span class="muted">Run {escape(str(best.RunID))}</span></td>'
                    f'<td class="p{p_no} {best_cls}">{fmt_pct(best.TotalReturnRate, 2)}</td>'
                    f'<td class="p{p_no}">{fmt_num(best.TotalTrades, 0)}</td>'
                    f'<td class="p{p_no}">{fmt_num(best.MaxDrawdownNetPoints, 1)}</td>'
                    f'<td class="p{p_no}">{fmt_pct(best.WinRate, 1)}</td>'
                )
            anchor_cell = ""
            if pattern_idx == 0:
                anchor_cell = (
                    f'<td class="anchor-layer" rowspan="{len(PATTERN_MODES)}">'
                    f'A{anchor_no:02d}<br>{escape(ANCHOR_LABELS[anchor_no])}<br>'
                    f'<span class="muted">多 {escape(ANCHOR_LONG_FORMULAS[anchor_no])}<br>'
                    f'空 {escape(ANCHOR_SHORT_FORMULAS[anchor_no])}</span></td>'
                )
            layer_body.append(
                f"<tr><td>{layer_no}</td>{anchor_cell}"
                f"<td>P{pattern_no:02d}</td>"
                f'<td class="left">{escape(PATTERN_LABELS[pattern_no])}<br>'
                f'<span class="muted">{escape(PATTERN_FAMILIES[pattern_no])}</span></td>'
                + "".join(row_cells)
                + "</tr>"
            )
            layer_no += 1

    year_headers = "".join(f"<th>{y}</th>" for y in years)
    anchor_options = "".join(f'<option value="A{k:02d}">A{k:02d}</option>' for k in ANCHOR_MODES)
    pattern_options = "".join(f'<option value="P{k:02d}">P{k:02d}</option>' for k in PATTERN_MODES)
    style = """
body{font-family:"Microsoft JhengHei",Arial,sans-serif;margin:0;background:#f7faf8;color:#1d2823;font-size:17px}
header{padding:18px 8px;background:white;border-bottom:1px solid #dce7e1}
h1{font-size:30px;margin:0 0 6px}.sub{color:#64736d;line-height:1.55}
main{padding:10px 6px}.controls{display:grid;grid-template-columns:repeat(9,minmax(120px,1fr));gap:8px;margin:8px 0 10px}
label{font-size:13px;color:#60716a;font-weight:700}input,select,button{width:100%;box-sizing:border-box;padding:8px;border:1px solid #d8e4df;border-radius:4px;background:#fff;font-size:15px}
button{font-weight:800;color:#285f86;background:#f5faf8}.count{font-weight:800;margin:6px 0 8px;color:#40544b}
.wrap{overflow:auto;max-height:78vh;border:1px solid #dce7e1;background:white}
.layer-wrap{overflow:auto;border:1px solid #dce7e1;background:white;margin:10px 0 22px}
.section-title{font-size:24px;font-weight:900;margin:18px 0 6px}.section-note{color:#64736d;margin-bottom:8px}
table{border-collapse:separate;border-spacing:0;min-width:3300px;font-size:14px}
th,td{border-right:1px solid #e0e8e4;border-bottom:1px solid #e8efeb;padding:6px 7px;text-align:right;vertical-align:middle;white-space:nowrap}
th{position:sticky;top:0;background:#dfece6;z-index:2;cursor:pointer}
tbody tr:nth-child(even) td{background:#f6faf8}td.left{text-align:left}.formula{text-align:left;white-space:normal;min-width:220px;font-size:12px;line-height:1.35}
.long{background:#fff3ef}.short{background:#eef9f2}.pos{color:#bd3e31;font-weight:800}.neg{color:#2f8b58;font-weight:800}.empty{color:#9aa7a0;text-align:center}.muted{color:#71817a;font-size:12px}
.layer-table{min-width:1800px;font-size:14px}.layer-table th{cursor:default}.layer-table td{height:42px}
.anchor-layer{text-align:left;font-weight:900;background:#d6d431;min-width:190px;line-height:1.45}
.pgrp1,.p1{background:#d6d431}.pgrp2,.p2{background:#438fba;color:#10232c}.pgrp3,.p3{background:#62a958;color:#102817}
.layer-table tbody tr:nth-child(even) td.p1{background:#dede43}.layer-table tbody tr:nth-child(even) td.p2{background:#4a99c4}.layer-table tbody tr:nth-child(even) td.p3{background:#6ab260}
"""
    script = """
const rows = Array.from(document.querySelectorAll('#mainRows tr'));
const visibleCount = document.getElementById('visibleCount');
function numValue(id, fallback){
  const raw = document.getElementById(id).value.trim();
  return raw === '' ? fallback : Number(raw);
}
function applyFilters(){
  const q = document.getElementById('q').value.trim().toLowerCase();
  const anchor = document.getElementById('anchor').value;
  const pattern = document.getElementById('pattern').value;
  const minTrades = numValue('minTrades', -Infinity);
  const minReturn = numValue('minReturn', -Infinity);
  const minPf = numValue('minPf', -Infinity);
  const minYears = numValue('minYears', -Infinity);
  let shown = 0;
  for (const tr of rows){
    const ok =
      (!q || tr.dataset.search.toLowerCase().includes(q)) &&
      (!anchor || tr.dataset.anchor === anchor) &&
      (!pattern || tr.dataset.pattern === pattern) &&
      Number(tr.dataset.trades) >= minTrades &&
      Number(tr.dataset.return) >= minReturn &&
      Number(tr.dataset.pf) >= minPf &&
      Number(tr.dataset.positiveYears) >= minYears;
    tr.style.display = ok ? '' : 'none';
    if (ok) shown++;
  }
  visibleCount.textContent = shown.toLocaleString();
}
function resetFilters(){
  for (const id of ['q','anchor','pattern','minTrades','minReturn','minPf','minYears']) document.getElementById(id).value = '';
  applyFilters();
}
let sortState = {index: 1, dir: 1};
document.querySelectorAll('#detailTable th').forEach((th, index) => {
  th.addEventListener('click', () => {
    const dir = sortState.index === index ? -sortState.dir : 1;
    sortState = {index, dir};
    const tbody = document.getElementById('mainRows');
    rows.sort((a,b) => {
      const av = a.children[index]?.dataset.sort ?? a.children[index]?.innerText ?? '';
      const bv = b.children[index]?.dataset.sort ?? b.children[index]?.innerText ?? '';
      const an = Number(String(av).replaceAll(',', ''));
      const bn = Number(String(bv).replaceAll(',', ''));
      if (!Number.isNaN(an) && !Number.isNaN(bn)) return (an - bn) * dir;
      return String(av).localeCompare(String(bv), 'zh-Hant') * dir;
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
for (const id of ['q','anchor','pattern','minTrades','minReturn','minPf','minYears']) document.getElementById(id).addEventListener('input', applyFilters);
applyFilters();
"""
    output.write_text(
        f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>XS Anchor ROD Excel 總表</title>
<style>{style}</style>
<header><h1>XS Anchor ROD Pullback 18,816 組 Excel 總表</h1>
<div class="sub">預設照 Excel RunID 原始順序排列，不用績效排序。8 Anchor × 16 Pattern × 49 GapMin/GapMax × 3 Penetrate；績效已扣出場滑點 2 點、來回手續費 36 元、期交稅，本金 250,000 元。</div></header>
<main>
<section>
<div class="section-title">分層總表：Anchor × Pattern × P</div>
<div class="section-note">每一列是一個 Anchor + Pattern；P=1、P=2、P=3 三個色塊各自挑出 49 組 Gap 中總報酬率最高的組合，方便先從上層挑方向。</div>
<div class="layer-wrap"><table class="layer-table"><thead>
<tr><th rowspan="2">編號</th><th rowspan="2">Anchor</th><th rowspan="2">Pattern</th><th rowspan="2">K線態樣</th>
<th colspan="5" class="pgrp1">P=1 回踩/回抽</th><th colspan="5" class="pgrp2">P=2 回踩/回抽</th><th colspan="5" class="pgrp3">P=3 回踩/回抽</th></tr>
<tr><th class="pgrp1">最佳Gap</th><th class="pgrp1">總報酬率</th><th class="pgrp1">次數</th><th class="pgrp1">MDD</th><th class="pgrp1">勝率</th>
<th class="pgrp2">最佳Gap</th><th class="pgrp2">總報酬率</th><th class="pgrp2">次數</th><th class="pgrp2">MDD</th><th class="pgrp2">勝率</th>
<th class="pgrp3">最佳Gap</th><th class="pgrp3">總報酬率</th><th class="pgrp3">次數</th><th class="pgrp3">MDD</th><th class="pgrp3">勝率</th></tr>
</thead><tbody>{''.join(layer_body)}</tbody></table></div>
</section>
<div class="section-title">明細總表：全部 18,816 組</div>
<div class="controls">
<div><label>搜尋</label><input id="q" placeholder="RunID / Anchor / Pattern / 公式"></div>
<div><label>Anchor</label><select id="anchor"><option value="">全部</option>{anchor_options}</select></div>
<div><label>Pattern</label><select id="pattern"><option value="">全部</option>{pattern_options}</select></div>
<div><label>最小交易次數</label><input id="minTrades" type="number" placeholder="不限"></div>
<div><label>最小總報酬率 %</label><input id="minReturn" type="number" step="0.01" placeholder="不限"></div>
<div><label>最小 PF</label><input id="minPf" type="number" step="0.01" placeholder="不限"></div>
<div><label>最小正報酬年度</label><input id="minYears" type="number" placeholder="不限"></div>
<div><label>清除</label><button onclick="resetFilters()">清除篩選</button></div>
<div><label>顯示筆數</label><div class="count"><span id="visibleCount">0</span> / {len(rows_df):,}</div></div>
</div>
<div class="wrap"><table id="detailTable"><thead><tr>
<th>編號</th><th>RunID</th><th>BaseID</th><th>AnchorID</th><th>Anchor名稱</th><th>PatternID</th><th>K線態樣</th>
<th>GapMin</th><th>GapMax</th><th>P</th><th>多方條件</th><th>空方條件</th>{year_headers}
<th>交易次數</th><th>總點數</th><th>平均每筆</th><th>勝率</th><th>PF</th><th>MDD點數</th><th>MDD%</th>
<th>淨損益</th><th>總報酬率</th><th>正報酬年度數</th><th>是否進下一輪</th><th>備註</th>
</tr></thead><tbody id="mainRows">{''.join(body)}</tbody></table></div>
</main><script>{script}</script></html>""",
        encoding="utf-8",
    )
