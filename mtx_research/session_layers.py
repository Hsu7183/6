from __future__ import annotations

from dataclasses import dataclass

from .xs_anchor_rod import XSParams


@dataclass(frozen=True)
class SessionSpec:
    key: str
    label: str
    params: XSParams


DAY_SESSION = SessionSpec(
    key="day",
    label="日盤",
    params=XSParams(
        use_time1=1,
        time1_begin=90500,
        time1_end=131000,
        time1_force=131200,
        use_time2=0,
        use_time3=0,
    ),
)

FULL_SESSION = SessionSpec(
    key="all",
    label="全日",
    params=XSParams(
        use_time1=1,
        time1_begin=90500,
        time1_end=131000,
        time1_force=131200,
        use_time2=1,
        time2_begin=150300,
        time2_end=235500,
        time2_force=235700,
        use_time3=1,
        time3_begin=300,
        time3_end=45500,
        time3_force=45700,
    ),
)

SESSION_SPECS = {
    DAY_SESSION.key: DAY_SESSION,
    FULL_SESSION.key: FULL_SESSION,
}

DEFAULT_SESSION = FULL_SESSION.key


def resolve_session(key: str = DEFAULT_SESSION) -> SessionSpec:
    try:
        return SESSION_SPECS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(SESSION_SPECS))
        raise ValueError(f"unknown session: {key!r}; choices: {choices}") from exc
