from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.html_report import write_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MTX R2A HTML report.")
    parser.add_argument(
        "--r2a-dir",
        type=Path,
        default=Path("report_outputs") / "r2a_1k_trend_pullback_all_families",
        help="Folder containing summary_r2a_all.csv and top_r2a_*.csv.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    output = args.output or (args.r2a_dir / "mtx_r2a_report.html")
    path = write_html_report(args.r2a_dir, output, limit=args.limit)
    print(path)


if __name__ == "__main__":
    main()
