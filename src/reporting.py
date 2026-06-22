from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def write_data_check_report(
    path: Path,
    *,
    data_path: Path,
    data_report,
    sample_report,
    expected_param_count: int,
    actual_param_count: int,
) -> None:
    lines = [
        "L0 資料檢查報告",
        "",
        f"資料檔 = {data_path}",
        f"原始資料列數 = {data_report.raw_rows:,}",
        f"清理後K棒數 = {data_report.cleaned_rows:,}",
        f"移除重複 datetime 列數 = {data_report.duplicate_datetime_rows:,}",
        f"OHLC缺值列數 = {data_report.missing_ohlc_rows:,}",
        f"OHLC不合理列數 = {data_report.invalid_ohlc_rows:,}",
        f"成交量缺值補0列數 = {data_report.missing_volume_rows:,}",
        f"資料起點 = {data_report.datetime_min}",
        f"資料終點 = {data_report.datetime_max}",
        "",
        f"可研究樣本數 = {sample_report.sample_count:,}",
        f"跨日移除樣本數 = {sample_report.cross_day_removed:,}",
        f"缺下一根Open移除樣本數 = {sample_report.missing_next_removed:,}",
        f"樣本日期起點 = {sample_report.date_min}",
        f"樣本日期終點 = {sample_report.date_max}",
        f"樣本時間範圍 = {sample_report.time_min} ~ {sample_report.time_max}",
        "",
        f"報酬率本金 = {config.CAPITAL_TWD:,.0f} 元",
        f"小台每點價值 = {config.POINT_VALUE_TWD:,.0f} 元",
        f"進場滑點 = {config.ENTRY_SLIPPAGE_POINTS:g} 點",
        f"出場滑點 = {config.EXIT_SLIPPAGE_POINTS:g} 點",
        f"單邊手續費 = {config.FEE_PER_SIDE_TWD:,.0f} 元",
        f"一次進出手續費 = {config.FEE_PER_SIDE_TWD * 2:,.0f} 元",
        f"期交稅率 = {config.TRANSACTION_TAX_RATE:.5f} / 每邊",
        "期交稅取整 = 單邊四捨五入到元",
        "",
        f"預期參數組合數 = {expected_param_count:,}",
        f"實際參數組合數 = {actual_param_count:,}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "L1_c1_rod_bodymin_all_params.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_top(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    prefix: str,
    sort_col: str,
    filename: str,
    min_trades: int,
) -> Path:
    filtered = df[df[f"{prefix}_trade_count"] >= min_trades].copy()
    filtered = filtered.sort_values(sort_col, ascending=False, na_position="last").head(config.TOP_N)
    path = output_dir / filename
    filtered.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_top_csvs(df: pd.DataFrame, output_dir: Path, *, min_trades: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _write_top(
            df,
            output_dir,
            prefix="long",
            sort_col="long_robust_score",
            filename="L1_top_long_by_robust_score.csv",
            min_trades=min_trades,
        ),
        _write_top(
            df,
            output_dir,
            prefix="short",
            sort_col="short_robust_score",
            filename="L1_top_short_by_robust_score.csv",
            min_trades=min_trades,
        ),
        _write_top(
            df,
            output_dir,
            prefix="combined",
            sort_col="combined_robust_score",
            filename="L1_top_combined_by_robust_score.csv",
            min_trades=min_trades,
        ),
        _write_top(
            df,
            output_dir,
            prefix="long",
            sort_col="long_win_rate",
            filename="L1_top_long_by_win_rate.csv",
            min_trades=min_trades,
        ),
        _write_top(
            df,
            output_dir,
            prefix="short",
            sort_col="short_win_rate",
            filename="L1_top_short_by_win_rate.csv",
            min_trades=min_trades,
        ),
        _write_top(
            df,
            output_dir,
            prefix="combined",
            sort_col="combined_win_rate",
            filename="L1_top_combined_by_win_rate.csv",
            min_trades=min_trades,
        ),
    ]
    return outputs


def write_progress(output_dir: Path, *, last_completed_rule_id: int, status: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "{\n"
        f'  "last_completed_rule_id": {last_completed_rule_id},\n'
        f'  "status": "{status}"\n'
        "}\n"
    )
    (output_dir / "progress.json").write_text(text, encoding="utf-8")
