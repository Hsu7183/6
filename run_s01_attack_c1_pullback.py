from __future__ import annotations

import argparse
from pathlib import Path

from mtx_research.data_sources import DEFAULT_INSTRUMENT, DATA_SOURCES, resolve_data_path
from src.s01_attack_c1_pullback import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S01_ATTACK_C1_PULLBACK scan.")
    parser.add_argument("--data", type=Path, default=None, help="Override OHLCV data path.")
    parser.add_argument(
        "--instrument",
        choices=sorted(DATA_SOURCES),
        default=DEFAULT_INSTRUMENT,
        help="Use configured full-session data source when --data is omitted.",
    )
    parser.add_argument(
        "--outdir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("report_outputs") / "s01_attack_c1_pullback_v1",
    )
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    data_path = resolve_data_path(args.instrument, args.data)
    run(data_path, args.output_dir, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
