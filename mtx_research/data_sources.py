from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CostConfig


@dataclass(frozen=True)
class DataSource:
    symbol: str
    label: str
    path: Path
    cost: CostConfig


MTX_FULL_DATA = DataSource(
    symbol="mtx",
    label="MTX full-session 6y",
    path=Path(
        r"C:\XQ\data\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343"
        r"\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt"
    ),
    cost=CostConfig(point_value_twd=50, fee_per_side_twd=18),
)

TX_FULL_DATA = DataSource(
    symbol="tx",
    label="TX full-session 6y",
    path=Path(
        r"C:\XQ\data\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343"
        r"\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt"
    ),
    cost=CostConfig(point_value_twd=200, fee_per_side_twd=35),
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


def cost_for_instrument(instrument: str = DEFAULT_INSTRUMENT) -> CostConfig:
    try:
        return DATA_SOURCES[instrument].cost
    except KeyError as exc:
        choices = ", ".join(sorted(DATA_SOURCES))
        raise ValueError(f"unknown instrument: {instrument!r}; choices: {choices}") from exc
