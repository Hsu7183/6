from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ResearchConfig


@dataclass
class SampleBuildReport:
    sample_count: int
    cross_day_removed: int
    missing_next_removed: int
    next_after_force_removed: int
    date_min: str
    date_max: str


def _safe_pct(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator * 100.0, denominator, out=out, where=denominator > 0)
    return out


def time_segment(time_int: int) -> str:
    if time_int < 93000:
        return "T1"
    if time_int < 103000:
        return "T2"
    if time_int < 123000:
        return "T3"
    return "T4"


def build_r2a_samples(df: pd.DataFrame, config: ResearchConfig) -> tuple[pd.DataFrame, SampleBuildReport]:
    work = df.sort_values("datetime").reset_index(drop=True).copy()
    prev = work.shift(1)
    nxt = work.shift(-1)

    in_time = (work["TimeInt"] >= config.time.begin_time) & (work["TimeInt"] <= config.time.end_time)
    in_time &= work["TimeInt"] < config.time.force_exit_time
    has_next = nxt["Open"].notna()
    same_day = (prev["DateInt"] == work["DateInt"]) & (work["DateInt"] == nxt["DateInt"])
    next_after_force = has_next & nxt["TimeInt"].gt(config.time.force_exit_time)
    mask = in_time & has_next & same_day & ~next_after_force

    missing_next_removed = int((in_time & ~has_next).sum())
    cross_day_removed = int((in_time & has_next & ~same_day).sum())
    next_after_force_removed = int((in_time & has_next & same_day & next_after_force).sum())

    samples = pd.DataFrame(
        {
            "BarIndex": work.index.to_numpy()[mask.to_numpy()],
            "EntryIndex": work.index.to_numpy()[mask.to_numpy()],
            "ExitIndex": work.index.to_numpy()[mask.to_numpy()] + 1,
            "DateTime": work.loc[mask, "datetime"].to_numpy(),
            "NextDateTime": nxt.loc[mask, "datetime"].to_numpy(),
            "DateInt": work.loc[mask, "DateInt"].to_numpy(dtype=np.int32),
            "TimeInt": work.loc[mask, "TimeInt"].to_numpy(dtype=np.int32),
            "Year": work.loc[mask, "Year"].to_numpy(dtype=np.int16),
            "O1": prev.loc[mask, "Open"].to_numpy(dtype=float),
            "H1": prev.loc[mask, "High"].to_numpy(dtype=float),
            "L1": prev.loc[mask, "Low"].to_numpy(dtype=float),
            "C1": prev.loc[mask, "Close"].to_numpy(dtype=float),
            "O0": work.loc[mask, "Open"].to_numpy(dtype=float),
            "H0": work.loc[mask, "High"].to_numpy(dtype=float),
            "L0": work.loc[mask, "Low"].to_numpy(dtype=float),
            "NextOpen": nxt.loc[mask, "Open"].to_numpy(dtype=float),
        }
    ).reset_index(drop=True)

    range1 = samples["H1"].to_numpy(dtype=float) - samples["L1"].to_numpy(dtype=float)
    c1 = samples["C1"].to_numpy(dtype=float)
    o1 = samples["O1"].to_numpy(dtype=float)
    h1 = samples["H1"].to_numpy(dtype=float)
    l1 = samples["L1"].to_numpy(dtype=float)
    body_long = c1 - o1
    body_short = o1 - c1
    abs_body = np.abs(c1 - o1)
    body_high = np.maximum(o1, c1)
    body_low = np.minimum(o1, c1)
    upper_tail = h1 - body_high
    lower_tail = body_low - l1

    samples["Range1"] = range1
    samples["BodyLong"] = body_long
    samples["BodyShort"] = body_short
    samples["AbsBody"] = abs_body
    samples["UpperTail"] = upper_tail
    samples["LowerTail"] = lower_tail
    samples["BodyPct"] = _safe_pct(abs_body, range1)
    samples["UpperTailPct"] = _safe_pct(upper_tail, range1)
    samples["LowerTailPct"] = _safe_pct(lower_tail, range1)
    samples["ClosePosPct"] = _safe_pct(c1 - l1, range1)
    samples["M1"] = (h1 + l1) / 2.0
    samples["BM1"] = (o1 + c1) / 2.0
    samples["LongDirection"] = c1 > o1
    samples["ShortDirection"] = c1 < o1
    samples["TimeSegment"] = samples["TimeInt"].map(time_segment)

    report = SampleBuildReport(
        sample_count=len(samples),
        cross_day_removed=cross_day_removed,
        missing_next_removed=missing_next_removed,
        next_after_force_removed=next_after_force_removed,
        date_min=str(samples["DateInt"].min()) if not samples.empty else "",
        date_max=str(samples["DateInt"].max()) if not samples.empty else "",
    )
    return samples, report
