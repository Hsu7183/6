from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataSource:
    symbol: str
    label: str
    path: Path


MTX_FULL_DATA = DataSource(
    symbol="mtx",
    label="MTX full-session 6y",
    path=Path(
        r"C:\XQ\data\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343"
        r"\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt"
    ),
)

TX_FULL_DATA = DataSource(
    symbol="tx",
    label="TX full-session 6y",
    path=Path(
        r"C:\XQ\data\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343"
        r"\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt"
    ),
)

DATA_SOURCES = {
    MTX_FULL_DATA.symbol: MTX_FULL_DATA,
    TX_FULL_DATA.symbol: TX_FULL_DATA,
}

DEFAULT_INSTRUMENT = "mtx"


def resolve_data_path(instrument: str = DEFAULT_INSTRUMENT, data: Path | None = None) -> Path:
    if data is not None:
        return data
    try:
        return DATA_SOURCES[instrument].path
    except KeyError as exc:
        choices = ", ".join(sorted(DATA_SOURCES))
        raise ValueError(f"unknown instrument: {instrument!r}; choices: {choices}") from exc
