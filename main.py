from __future__ import annotations

import argparse
from pathlib import Path

from mtx_research.data_sources import MTX_FULL_DATA
from src import config
from src.build_samples import build_samples
from src.load_data import load_price_data
from src.param_grid import build_param_grid
from src.html_report import write_html_report
from src.reporting import write_data_check_report, write_main_csv, write_progress, write_top_csvs
from src.scanner_l1 import scan_l1
from src.yearly import write_top_trade_logs, write_yearly_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小台指 1分K C1 ROD L1 BodyMin 掃描")
    parser.add_argument("--data", type=Path, default=None, help="OHLCV CSV/TXT 檔案路徑")
    parser.add_argument("--begin-time", default=config.BEGIN_TIME)
    parser.add_argument("--end-time", default=config.END_TIME)
    parser.add_argument("--exit-time-limit", default=config.EXIT_TIME_LIMIT)
    parser.add_argument("--min-trades", type=int, default=config.MIN_TRADES)
    parser.add_argument("--capital", type=float, default=config.CAPITAL_TWD, help="報酬率本金，預設 250000")
    parser.add_argument("--point-value", type=float, default=config.POINT_VALUE_TWD, help="小台每點價值，預設 50")
    parser.add_argument("--entry-slippage", type=float, default=config.ENTRY_SLIPPAGE_POINTS, help="進場滑點，預設 0")
    parser.add_argument("--exit-slippage", type=float, default=config.EXIT_SLIPPAGE_POINTS, help="出場滑點，預設 2")
    parser.add_argument("--fee-per-side", type=float, default=config.FEE_PER_SIDE_TWD, help="單邊手續費，預設 18")
    parser.add_argument("--tax-rate", type=float, default=config.TRANSACTION_TAX_RATE, help="期交稅率，預設 0.00002")
    parser.add_argument("--resume", action="store_true", help="若主輸出已完成，跳過重新掃描並重建附屬輸出")
    parser.add_argument("--chunk-size", type=int, default=5000, help="每幾筆樣本印出一次進度")
    return parser.parse_args()


def resolve_data_path(data_arg: Path | None) -> Path:
    if data_arg is not None:
        return data_arg.resolve()
    if MTX_FULL_DATA.path.exists():
        return MTX_FULL_DATA.path.resolve()
    files = sorted([*config.DATA_DIR.glob("*.csv"), *config.DATA_DIR.glob("*.txt")])
    if len(files) == 1:
        return files[0].resolve()
    root_files = sorted([*config.PROJECT_ROOT.glob("*.csv"), *config.PROJECT_ROOT.glob("*.txt")])
    if len(files) == 0 and len(root_files) == 1:
        return root_files[0].resolve()
    raise SystemExit("請用 --data 指定資料檔，或在 data/ 或專案根目錄只放一個 CSV/TXT。")


def main() -> None:
    args = parse_args()
    config.CAPITAL_TWD = args.capital
    config.POINT_VALUE_TWD = args.point_value
    config.ENTRY_SLIPPAGE_POINTS = args.entry_slippage
    config.EXIT_SLIPPAGE_POINTS = args.exit_slippage
    config.FEE_PER_SIDE_TWD = args.fee_per_side
    config.TRANSACTION_TAX_RATE = args.tax_rate
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = resolve_data_path(args.data)

    print("building parameter grid...")
    params = build_param_grid()
    actual_param_count = len(params)
    print(f"expected_param_count={config.EXPECTED_PARAM_COUNT:,}")
    print(f"actual_param_count={actual_param_count:,}")

    print(f"loading data: {data_path}")
    df, data_report = load_price_data(data_path)
    samples, sample_report = build_samples(
        df,
        begin_time=args.begin_time,
        end_time=args.end_time,
        exit_time_limit=args.exit_time_limit,
    )
    write_data_check_report(
        config.OUTPUT_DIR / "L0_data_check_report.txt",
        data_path=data_path,
        data_report=data_report,
        sample_report=sample_report,
        expected_param_count=config.EXPECTED_PARAM_COUNT,
        actual_param_count=actual_param_count,
    )

    main_csv = config.OUTPUT_DIR / "L1_c1_rod_bodymin_all_params.csv"
    if args.resume and main_csv.exists():
        print(f"resume mode: reading existing {main_csv}")
        import pandas as pd

        main_df = pd.read_csv(main_csv, encoding="utf-8-sig", low_memory=False)
        if len(main_df) != config.EXPECTED_PARAM_COUNT:
            raise RuntimeError(f"existing main csv row count {len(main_df):,} is not 328,000")
    else:
        print("scanning L1 parameter grid...")
        main_df = scan_l1(samples, params, min_trades=args.min_trades, chunk_size=args.chunk_size)
        write_main_csv(main_df, config.OUTPUT_DIR)
        write_progress(config.OUTPUT_DIR, last_completed_rule_id=int(main_df["rule_id"].max()), status="main_complete")

    print("writing top csvs...")
    write_top_csvs(main_df, config.OUTPUT_DIR, min_trades=args.min_trades)
    print("writing yearly stats for top rules...")
    write_yearly_stats(samples, params, main_df, config.OUTPUT_DIR)
    print("writing top 20 trade logs...")
    write_top_trade_logs(samples, params, main_df, config.OUTPUT_DIR, min_trades=args.min_trades)
    print("writing html report...")
    write_html_report(
        main_csv=config.OUTPUT_DIR / "L1_c1_rod_bodymin_all_params.csv",
        report_txt=config.OUTPUT_DIR / "L0_data_check_report.txt",
        output_html=config.PROJECT_ROOT / "rod_yearly_report.html",
    )
    write_progress(config.OUTPUT_DIR, last_completed_rule_id=int(main_df["rule_id"].max()), status="complete")
    print("done.")


if __name__ == "__main__":
    main()
