from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.data_sources import DEFAULT_INSTRUMENT, DATA_SOURCES, resolve_data_path
from mtx_research.xs_anchor_rod import EXPECTED_COMBOS, combo_count, scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XS Anchor ROD Pullback 18,816-combo scan.")
    parser.add_argument("--data", type=Path, default=None, help="Override OHLCV data path.")
    parser.add_argument(
        "--instrument",
        choices=sorted(DATA_SOURCES),
        default=DEFAULT_INSTRUMENT,
        help="Use configured full-session data source when --data is omitted.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("report_outputs") / "xs_anchor_rod_18816",
    )
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    total = combo_count()
    if total != EXPECTED_COMBOS:
        raise RuntimeError(f"combo count {total:,} != {EXPECTED_COMBOS:,}")
    data_path = resolve_data_path(args.instrument, args.data)
    paths = scan(data_path, args.outdir, progress_every=args.progress_every)
    print(f"Done. combos={total:,}")
    print(f"data={data_path}")
    print(f"summary={paths['summary']}")
    print(f"html={paths['html']}")


if __name__ == "__main__":
    main()
