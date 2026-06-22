from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


LogFn = Callable[[str], None]


@dataclass
class DataLoadReport:
    raw_rows: int = 0
    cleaned_rows: int = 0
    duplicate_datetime_rows: int = 0
    invalid_ohlc_rows: int = 0
    missing_ohlc_rows: int = 0
    missing_volume_rows: int = 0
    datetime_min: str = ""
    datetime_max: str = ""


def _looks_headerless(first_line: str) -> bool:
    parts = first_line.strip().replace(",", " ").split()
    return bool(parts and parts[0].replace(".", "", 1).isdigit())


def _normalize_column_name(name: object) -> str:
    text = str(name).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "datetime": "datetime",
        "timestamp": "datetime",
        "ts": "datetime",
        "date": "date",
        "日期": "date",
        "time": "time",
        "時間": "time",
        "open": "open",
        "o": "open",
        "開盤": "open",
        "開盤價": "open",
        "high": "high",
        "h": "high",
        "最高": "high",
        "最高價": "high",
        "low": "low",
        "l": "low",
        "最低": "low",
        "最低價": "low",
        "close": "close",
        "c": "close",
        "收盤": "close",
        "收盤價": "close",
        "volume": "volume",
        "vol": "volume",
        "v": "volume",
        "成交量": "volume",
    }
    return aliases.get(text, text)


def normalize_time_to_hhmmss(value: object) -> str:
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if not digits:
        return ""
    if len(digits) <= 4:
        return digits.zfill(4) + "00"
    return digits.zfill(6)[:6]


def normalize_date_to_yyyymmdd(value: object) -> str:
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    return digits.zfill(8)[:8]


def _combine_date_time(date: pd.Series, time: pd.Series) -> pd.Series:
    return date.map(normalize_date_to_yyyymmdd) + time.map(normalize_time_to_hhmmss)


def _read_raw(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        first_line = handle.readline()

    if _looks_headerless(first_line):
        df = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None, encoding="utf-8-sig")
        if df.shape[1] < 6:
            raise ValueError("headerless data must contain datetime open high low close volume")
        df = df.iloc[:, :6].copy()
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        return df

    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [_normalize_column_name(col) for col in df.columns]
    if "datetime" not in df.columns:
        if "date" not in df.columns or "time" not in df.columns:
            raise ValueError("data must contain datetime, or date + time columns")
        df["datetime"] = _combine_date_time(df["date"], df["time"])
    required = ["datetime", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]].copy()


def _parse_datetime(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, errors="coerce")
    digits = raw.str.replace(r"\D", "", regex=True)
    compact = parsed.isna() & digits.str.len().isin([12, 14])
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(digits.loc[compact].str[:12], format="%Y%m%d%H%M", errors="coerce")
    return parsed


def load_ohlcv(path: Path, *, log: LogFn | None = None) -> tuple[pd.DataFrame, DataLoadReport]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = _read_raw(path)
    report = DataLoadReport(raw_rows=len(df))
    df["datetime"] = _parse_datetime(df["datetime"])
    df = df.dropna(subset=["datetime"]).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report.missing_volume_rows = int(df["volume"].isna().sum())
    df["volume"] = df["volume"].fillna(0)

    report.missing_ohlc_rows = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    before_dup = len(df)
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="first").reset_index(drop=True)
    report.duplicate_datetime_rows = before_dup - len(df)

    valid = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
        & (df["high"] >= df["low"])
    )
    report.invalid_ohlc_rows = int((~valid).sum())
    if report.invalid_ohlc_rows and log:
        log(f"invalid OHLC rows removed: {report.invalid_ohlc_rows:,}")
    df = df.loc[valid].reset_index(drop=True)

    df["DateInt"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)
    df["TimeInt"] = df["datetime"].dt.strftime("%H%M%S").astype(int)
    df["Year"] = df["datetime"].dt.year.astype(int)
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df = df[["datetime", "DateInt", "TimeInt", "Year", "Open", "High", "Low", "Close", "Volume"]]

    report.cleaned_rows = len(df)
    if not df.empty:
        report.datetime_min = str(df["datetime"].min())
        report.datetime_max = str(df["datetime"].max())
    return df, report
