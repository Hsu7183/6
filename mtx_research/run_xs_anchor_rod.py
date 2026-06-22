from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.xs_anchor_rod import EXPECTED_COMBOS, combo_count, scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XS Anchor ROD Pullback 18,816-combo scan.")
    parser.add_argument("--data", type=Path, default=Path("FIMTX_M1_202001020845.txt"))
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
    paths = scan(args.data, args.outdir, progress_every=args.progress_every)
    print(f"Done. combos={total:,}")
    print(f"summary={paths['summary']}")
    print(f"html={paths['html']}")


if __name__ == "__main__":
    main()
