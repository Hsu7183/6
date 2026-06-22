from __future__ import annotations

import argparse
from pathlib import Path

from src.s01_attack_c1_pullback import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S01_ATTACK_C1_PULLBACK scan.")
    parser.add_argument("--data", type=Path, default=Path("FIMTX_M1_202001020845.txt"))
    parser.add_argument(
        "--outdir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("report_outputs") / "s01_attack_c1_pullback_v1",
    )
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    run(args.data, args.output_dir, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
