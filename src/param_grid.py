from __future__ import annotations

import pandas as pd

from . import config


def build_param_grid() -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    rule_id = 1
    gap_pair_index = 0
    for open_gap_min in config.OPEN_GAP_MIN_LIST:
        for open_gap_max in config.OPEN_GAP_MAX_LIST:
            if open_gap_max < open_gap_min:
                continue
            for pullback_depth in config.PULLBACK_DEPTH_LIST:
                pullback_index = config.PULLBACK_DEPTH_LIST.index(pullback_depth)
                for body_min in config.BODY_MIN_LIST:
                    body_index = config.BODY_MIN_LIST.index(body_min)
                    rows.append(
                        {
                            "rule_id": rule_id,
                            "gap_pair_index": gap_pair_index,
                            "open_gap_min": open_gap_min,
                            "open_gap_max": open_gap_max,
                            "pullback_depth": pullback_depth,
                            "pullback_index": pullback_index,
                            "body_min": body_min,
                            "body_index": body_index,
                        }
                    )
                    rule_id += 1
            gap_pair_index += 1

    grid = pd.DataFrame(rows)
    actual = len(grid)
    if actual != config.EXPECTED_PARAM_COUNT:
        raise RuntimeError(
            f"actual_param_count={actual:,} != expected_param_count={config.EXPECTED_PARAM_COUNT:,}"
        )
    return grid


def build_gap_pairs() -> pd.DataFrame:
    pairs: list[dict[str, int]] = []
    gap_pair_index = 0
    for open_gap_min in config.OPEN_GAP_MIN_LIST:
        for open_gap_max in config.OPEN_GAP_MAX_LIST:
            if open_gap_max < open_gap_min:
                continue
            pairs.append(
                {
                    "gap_pair_index": gap_pair_index,
                    "open_gap_min": open_gap_min,
                    "open_gap_max": open_gap_max,
                }
            )
            gap_pair_index += 1
    return pd.DataFrame(pairs)

