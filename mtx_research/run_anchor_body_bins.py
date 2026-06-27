from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.anchor_body_bins import EXPECTED_COMBOS, scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A01-A08 x body-bin x open-gap-bin ROD scan.")
    parser.add_argument("--data", type=Path, default=Path("FIMTX_M1_202001020845.txt"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("report_outputs") / "anchor_body_gap_bins_11152",
    )
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    paths = scan(args.data, args.outdir, progress_every=args.progress_every)
    print(f"Done. combos={EXPECTED_COMBOS:,}")
    print(f"summary={paths['summary']}")
    print(f"html={paths['html']}")


if __name__ == "__main__":
    main()
