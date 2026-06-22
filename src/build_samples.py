from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SampleReport:
    sample_count: int
    cross_day_removed: int
    missing_next_removed: int
    date_min: str
    date_max: str
    time_min: str
    time_max: str


def _time_to_minutes(text: str) -> int:
    hour, minute = [int(part) for part in text.split(":")]
    return hour * 60 + minute


def build_samples(
    df: pd.DataFrame,
    *,
    begin_time: str,
    end_time: str,
    exit_time_limit: str,
) -> tuple[pd.DataFrame, SampleReport]:
    work = df.sort_values("datetime").reset_index(drop=True).copy()
    work["date"] = work["datetime"].dt.date.astype(str)
    work["time_minutes"] = work["datetime"].dt.hour * 60 + work["datetime"].dt.minute

    prev = work.shift(1)
    nxt = work.shift(-1)
    same_day = (prev["date"] == work["date"]) & (work["date"] == nxt["date"])
    has_next = nxt["open"].notna()
    begin = _time_to_minutes(begin_time)
    end = _time_to_minutes(end_time)
    exit_limit = _time_to_minutes(exit_time_limit)
    in_time = (work["time_minutes"] >= begin) & (work["time_minutes"] <= end)
    next_ok = nxt["time_minutes"].le(exit_limit)

    candidate = in_time
    missing_next_removed = int((candidate & ~has_next).sum())
    cross_day_removed = int((candidate & has_next & ~same_day).sum())
    mask = candidate & has_next & same_day & next_ok

    samples = pd.DataFrame(
        {
            "datetime": work.loc[mask, "datetime"].to_numpy(),
            "NEXT_DATETIME": nxt.loc[mask, "datetime"].to_numpy(),
            "date": work.loc[mask, "date"].to_numpy(),
            "year": work.loc[mask, "datetime"].dt.year.to_numpy(),
            "O1": prev.loc[mask, "open"].to_numpy(dtype=float),
            "H1": prev.loc[mask, "high"].to_numpy(dtype=float),
            "L1": prev.loc[mask, "low"].to_numpy(dtype=float),
            "C1": prev.loc[mask, "close"].to_numpy(dtype=float),
            "O0": work.loc[mask, "open"].to_numpy(dtype=float),
            "H0": work.loc[mask, "high"].to_numpy(dtype=float),
            "L0": work.loc[mask, "low"].to_numpy(dtype=float),
            "C0": work.loc[mask, "close"].to_numpy(dtype=float),
            "O_NEXT": nxt.loc[mask, "open"].to_numpy(dtype=float),
        }
    ).reset_index(drop=True)

    unique_dates = np.array(sorted(samples["date"].unique()))
    n_dates = len(unique_dates)
    train_end = int(np.floor(n_dates * 0.70))
    valid_end = int(np.floor(n_dates * 0.85))
    split_by_date: dict[str, str] = {}
    for idx, date in enumerate(unique_dates):
        if idx < train_end:
            split_by_date[date] = "train"
        elif idx < valid_end:
            split_by_date[date] = "valid"
        else:
            split_by_date[date] = "test"
    samples["split"] = samples["date"].map(split_by_date)

    report = SampleReport(
        sample_count=len(samples),
        cross_day_removed=cross_day_removed,
        missing_next_removed=missing_next_removed,
        date_min=str(samples["date"].min()) if not samples.empty else "",
        date_max=str(samples["date"].max()) if not samples.empty else "",
        time_min=begin_time,
        time_max=end_time,
    )
    return samples, report
