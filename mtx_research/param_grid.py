from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from . import config as cfg


ALL_FAMILY_PARAM_COLUMNS = [
    "BodyMin",
    "RangeMax",
    "BodyPctMin",
    "BodyPctFloor",
    "ClosePosMin",
    "EffOppTailMax",
    "MaruBodyPct",
    "TailMax",
    "MainTailMin",
    "CenterOffset",
    "RangeMinLarge",
    "RangeMaxLarge",
    "BodyMinLarge",
]


@dataclass(frozen=True)
class FamilySpec:
    name: str
    param_names: tuple[str, ...]
    values: tuple[tuple[int, ...], ...]

    @property
    def dense_shape(self) -> tuple[int, ...]:
        return tuple(len(v) for v in self.values)


def family_specs() -> list[FamilySpec]:
    return [
        FamilySpec(
            "F01_ATTACK",
            ("BodyMin", "RangeMax", "BodyPctMin", "EffOppTailMax"),
            (
                tuple(cfg.BODY_MIN_LIST),
                tuple(cfg.RANGE_MAX_LIST),
                tuple(cfg.BODY_PCT_MIN_LIST),
                tuple(cfg.EFF_OPP_TAIL_MAX_LIST),
            ),
        ),
        FamilySpec(
            "F02_STRONG_CLOSE",
            ("BodyMin", "RangeMax", "BodyPctFloor", "ClosePosMin"),
            (
                tuple(cfg.BODY_MIN_LIST),
                tuple(cfg.RANGE_MAX_LIST),
                tuple(cfg.BODY_PCT_FLOOR_LIST),
                tuple(cfg.CLOSE_POS_STRONG_LIST),
            ),
        ),
        FamilySpec(
            "F03_MARUBOZU",
            ("BodyMin", "RangeMax", "MaruBodyPct", "TailMax"),
            (
                tuple(cfg.BODY_MIN_LIST),
                tuple(cfg.RANGE_MAX_LIST),
                tuple(cfg.MARU_BODY_PCT_LIST),
                tuple(cfg.TAIL_MAX_LIST),
            ),
        ),
        FamilySpec(
            "F04_BODY_EFFICIENCY",
            ("BodyMin", "RangeMax", "BodyPctMin", "ClosePosMin"),
            (
                tuple(cfg.BODY_MIN_LIST),
                tuple(cfg.RANGE_MAX_LIST),
                tuple(cfg.BODY_PCT_MIN_LIST),
                tuple(cfg.CLOSE_POS_EFF_LIST),
            ),
        ),
        FamilySpec(
            "F05_TAIL_SUPPORT_CONTINUATION",
            ("BodyMin", "RangeMax", "MainTailMin", "EffOppTailMax"),
            (
                tuple(cfg.BODY_MIN_LIST),
                tuple(cfg.RANGE_MAX_LIST),
                tuple(cfg.MAIN_TAIL_MIN_LIST),
                tuple(cfg.EFF_OPP_TAIL_MAX_LIST),
            ),
        ),
        FamilySpec(
            "F06_BODY_CENTER",
            ("BodyMin", "RangeMax", "BodyPctMin", "ClosePosMin", "CenterOffset"),
            (
                tuple(cfg.BODY_MIN_LIST),
                tuple(cfg.RANGE_MAX_LIST),
                tuple(cfg.BODY_PCT_CENTER_LIST),
                tuple(cfg.CLOSE_POS_CENTER_LIST),
                tuple(cfg.CENTER_OFFSET_LIST),
            ),
        ),
        FamilySpec(
            "F07_LARGE_RANGE_ATTACK",
            ("RangePair", "BodyMinLarge", "BodyPctMin", "EffOppTailMax"),
            (
                tuple(range(len(cfg.LARGE_RANGE_PAIR_LIST))),
                tuple(cfg.BODY_MIN_LARGE_LIST),
                tuple(cfg.BODY_PCT_MIN_LIST),
                tuple(cfg.EFF_OPP_TAIL_MAX_LIST),
            ),
        ),
    ]


def body_range_legal(shape: tuple[int, ...]) -> np.ndarray:
    body = np.asarray(cfg.BODY_MIN_LIST)[:, None]
    rng = np.asarray(cfg.RANGE_MAX_LIST)[None, :]
    legal_2d = body <= rng
    extra = (1,) * (len(shape) - 2)
    return np.broadcast_to(legal_2d.reshape(legal_2d.shape + extra), shape).copy()


def family_legal_mask(spec: FamilySpec) -> np.ndarray:
    if spec.name in {"F01_ATTACK", "F02_STRONG_CLOSE", "F03_MARUBOZU", "F04_BODY_EFFICIENCY", "F05_TAIL_SUPPORT_CONTINUATION", "F06_BODY_CENTER"}:
        return body_range_legal(spec.dense_shape)
    return np.ones(spec.dense_shape, dtype=bool)


def open_gap_legal_mask() -> np.ndarray:
    mins = np.asarray(cfg.OPEN_GAP_MIN_LIST)[:, None]
    maxs = np.asarray(cfg.OPEN_GAP_MAX_LIST)[None, :]
    return maxs >= mins


def legal_open_gap_pairs() -> list[tuple[int, int]]:
    return [
        (mn, mx)
        for mn, mx in product(cfg.OPEN_GAP_MIN_LIST, cfg.OPEN_GAP_MAX_LIST)
        if mx >= mn
    ]


def family_combo_count(spec: FamilySpec) -> int:
    return int(family_legal_mask(spec).sum())


def r2a_combo_count() -> int:
    family_total = sum(family_combo_count(spec) for spec in family_specs())
    exec_total = len(legal_open_gap_pairs()) * len(cfg.PENETRATE_LIST) * len(cfg.EXEC_MODE_LIST)
    return family_total * exec_total


def assert_expected_r2a_count(expected: int) -> None:
    actual = r2a_combo_count()
    if actual != expected:
        raise RuntimeError(f"R2A combo count {actual:,} != expected {expected:,}")


def family_param_columns(spec: FamilySpec, coords: tuple[np.ndarray, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(len(coords[0])) if coords else [])
    for col in ALL_FAMILY_PARAM_COLUMNS:
        frame[col] = np.nan
    for dim, name in enumerate(spec.param_names):
        values = np.asarray(spec.values[dim])
        if name == "RangePair":
            pair_index = values[coords[dim]]
            pairs = np.asarray(cfg.LARGE_RANGE_PAIR_LIST, dtype=int)[pair_index]
            frame["RangeMinLarge"] = pairs[:, 0]
            frame["RangeMaxLarge"] = pairs[:, 1]
        else:
            frame[name] = values[coords[dim]]
    return frame


def exec_param_columns(coords: tuple[np.ndarray, ...]) -> pd.DataFrame:
    gap_min_idx, gap_max_idx, pen_idx, exec_idx = coords
    return pd.DataFrame(
        {
            "OpenGapMin": np.asarray(cfg.OPEN_GAP_MIN_LIST, dtype=int)[gap_min_idx],
            "OpenGapMax": np.asarray(cfg.OPEN_GAP_MAX_LIST, dtype=int)[gap_max_idx],
            "Penetrate": np.asarray(cfg.PENETRATE_LIST, dtype=int)[pen_idx],
            "BreakSmallMax": 8,
            "ExecMode": np.asarray(cfg.EXEC_MODE_LIST, dtype=object)[exec_idx],
        }
    )
