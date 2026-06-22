from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataCheck:
    raw_rows: int = 0
    cleaned_rows: int = 0
    duplicate_datetime_rows: int = 0
    invalid_ohlc_rows: int = 0
    missing_ohlc_rows: int = 0
    missing_volume_rows: int = 0
    datetime_min: str = ""
    datetime_max: str = ""


def _looks_headerless(first_line: str) -> bool:
    first = first_line.strip().split()[0] if first_line.strip() else ""
    return first[:8].isdigit()


def _normalize_column_name(name: object) -> str:
    text = str(name).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "datetime": "datetime",
        "timestamp": "datetime",
        "ts": "datetime",
        "date": "date",
        "日期": "date",
        "交易日": "date",
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
        "量": "volume",
    }
    return aliases.get(text, text)


def _combine_date_time(date: pd.Series, time: pd.Series) -> pd.Series:
    date_digits = date.astype(str).str.strip().str.replace(r"\D", "", regex=True).str.zfill(8)
    time_digits = time.astype(str).str.strip().str.replace(r"\D", "", regex=True)

    def normalize_time(value: str) -> str:
        if len(value) <= 4:
            return value.zfill(4) + "00"
        return value.zfill(6)[:6]

    normalized_time = time_digits.map(normalize_time)
    return date_digits + normalized_time


def _read_raw(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        first_line = handle.readline()

    if _looks_headerless(first_line):
        df = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None, encoding="utf-8-sig")
        if df.shape[1] < 6:
            raise ValueError("headerless data must have at least 6 columns: datetime open high low close volume")
        df = df.iloc[:, :6]
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        return df

    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [_normalize_column_name(col) for col in df.columns]
    if "datetime" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = _combine_date_time(df["date"], df["time"])
        else:
            raise ValueError("data must contain datetime, or date + time columns")
    required = ["datetime", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]].copy()


def load_price_data(path: Path) -> tuple[pd.DataFrame, DataCheck]:
    df = _read_raw(path)
    report = DataCheck(raw_rows=len(df))

    raw_dt = df["datetime"].astype(str).str.strip()
    digits = raw_dt.str.replace(r"\D", "", regex=True)
    parsed = pd.to_datetime(raw_dt, errors="coerce")
    compact_mask = parsed.isna() & digits.str.len().isin([12, 14])
    if compact_mask.any():
        parsed.loc[compact_mask] = pd.to_datetime(
            digits.loc[compact_mask].str[:12], format="%Y%m%d%H%M", errors="coerce"
        )
    df["datetime"] = parsed
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

    valid_ohlc = (
        (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df["low"])
    )
    report.invalid_ohlc_rows = int((~valid_ohlc).sum())
    df = df.loc[valid_ohlc].reset_index(drop=True)

    report.cleaned_rows = len(df)
    if not df.empty:
        report.datetime_min = str(df["datetime"].min())
        report.datetime_max = str(df["datetime"].max())
    return df, report
