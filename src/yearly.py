from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .metrics import metric_dict
from .scanner_l1 import materialize_rule_trades


def select_top_rule_ids(main_df: pd.DataFrame) -> list[int]:
    ids: list[int] = []
    for col in ("combined_robust_score", "long_robust_score", "short_robust_score"):
        top_ids = main_df.sort_values(col, ascending=False, na_position="last")["rule_id"].head(
            config.YEARLY_TOP_N
        )
        ids.extend(int(rule_id) for rule_id in top_ids)
    return list(dict.fromkeys(ids))


def build_yearly_stats(samples: pd.DataFrame, params: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    param_by_rule = params.set_index("rule_id")
    rows: list[dict[str, object]] = []
    for rule_id in select_top_rule_ids(main_df):
        if rule_id not in param_by_rule.index:
            continue
        rule = param_by_rule.loc[rule_id].copy()
        rule["rule_id"] = rule_id
        trades = materialize_rule_trades(samples, rule)
        if trades.empty:
            continue
        for side_type, side_trades in (
            ("LONG", trades[trades["side"] == "LONG"]),
            ("SHORT", trades[trades["side"] == "SHORT"]),
            ("COMBINED", trades),
        ):
            if side_trades.empty:
                continue
            for year, group in side_trades.groupby("year", sort=True):
                stats = metric_dict(group["points"].to_numpy(), group["datetime"].to_numpy())
                rows.append(
                    {
                        "rule_id": rule_id,
                        "side_type": side_type,
                        "year": int(year),
                        "trade_count": stats["trade_count"],
                        "win_rate": stats["win_rate"],
                        "avg_points": stats["avg_points"],
                        "gross_points": stats["gross_points"],
                        "net_profit_twd": stats["gross_points"] * config.POINT_VALUE_TWD,
                        "return_rate": (stats["gross_points"] * config.POINT_VALUE_TWD) / config.CAPITAL_TWD,
                        "profit_factor": stats["profit_factor"],
                        "max_drawdown_points": stats["max_drawdown_points"],
                        "max_drawdown_twd": stats["max_drawdown_points"] * config.POINT_VALUE_TWD,
                    }
                )
    return pd.DataFrame(rows)


def write_yearly_stats(samples: pd.DataFrame, params: pd.DataFrame, main_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    yearly = build_yearly_stats(samples, params, main_df)
    path = output_dir / "L1_yearly_stats_top_rules.csv"
    yearly.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_top_trade_logs(
    samples: pd.DataFrame,
    params: pd.DataFrame,
    main_df: pd.DataFrame,
    output_dir: Path,
    *,
    min_trades: int,
) -> Path:
    log_dir = output_dir / "L1_top20_trade_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    param_by_rule = params.set_index("rule_id")
    top = (
        main_df[main_df["combined_trade_count"] >= min_trades]
        .sort_values("combined_robust_score", ascending=False, na_position="last")
        .head(config.TOP_TRADE_LOG_N)
    )
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        rule_id = int(row.rule_id)
        rule = param_by_rule.loc[rule_id].copy()
        rule["rule_id"] = rule_id
        trades = materialize_rule_trades(samples, rule)
        path = log_dir / f"rule_{rank:06d}_trades.csv"
        trades.drop(columns=["year", "rule_id"], errors="ignore").to_csv(path, index=False, encoding="utf-8-sig")
    return log_dir
