from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.backtest_engine import r2a_total_combo_count, scan_family_r2a
from mtx_research.checkpoint import (
    append_log,
    ensure_dirs,
    file_hash,
    load_status,
    new_status,
    reset_stage,
    save_status,
    stable_config_hash,
    stage_checkpoint_dir,
)
from mtx_research.config import ResearchConfig
from mtx_research.data_loader import load_ohlcv
from mtx_research.feature_engine import build_r2a_samples
from mtx_research.html_report import write_html_report
from mtx_research.param_grid import family_combo_count, family_specs, legal_open_gap_pairs
from mtx_research.r2b_engine import run_r2b
from mtx_research.report_writer import write_r2a_final_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MTX 1K trend pullback research runner")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("report_outputs"))
    parser.add_argument("--stages", type=str, default="r2a")
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--force", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--limit-families", type=int, default=0, help="Smoke-test only. 0 means all families.")
    return parser.parse_args()


def _write_part(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_r2a(
    *,
    data_path: Path,
    outdir: Path,
    samples: pd.DataFrame,
    config: ResearchConfig,
    resume: bool,
    force: bool,
    progress_every: int,
    limit_families: int,
    input_hash: str,
    config_hash: str,
) -> None:
    total_combos = r2a_total_combo_count(config)
    ckpt = stage_checkpoint_dir(outdir, "r2a")
    if force:
        reset_stage(outdir, "r2a")
        ckpt = stage_checkpoint_dir(outdir, "r2a")

    status = load_status(outdir, "r2a") if resume else None
    if status is None:
        status = new_status(
            base_outdir=outdir,
            stage="r2a",
            total_combos=total_combos,
            input_data_hash=input_hash,
            config_hash=config_hash,
        )

    specs = family_specs()
    if limit_families > 0:
        specs = specs[:limit_families]
        append_log(outdir, f"SMOKE MODE: limit_families={limit_families}")

    exec_combo_count = len(legal_open_gap_pairs()) * 3 * 5
    rule_id_start = 1
    completed = 0
    for family_index, spec in enumerate(family_specs(), start=1):
        family_rows = family_combo_count(spec) * exec_combo_count
        if spec not in specs:
            rule_id_start += family_rows
            continue

        summary_path = ckpt / f"summary_part_{family_index:02d}_{spec.name}.csv"
        by_year_path = ckpt / f"by_year_part_{family_index:02d}_{spec.name}.csv"
        by_side_path = ckpt / f"by_side_part_{family_index:02d}_{spec.name}.csv"
        by_class_path = ckpt / f"by_open_class_part_{family_index:02d}_{spec.name}.csv"
        by_segment_path = ckpt / f"by_time_segment_part_{family_index:02d}_{spec.name}.csv"

        if resume and summary_path.exists():
            append_log(outdir, f"R2A skip existing {spec.name}")
            completed += family_rows
            rule_id_start += family_rows
            continue

        append_log(outdir, f"R2A start {spec.name}, rows={family_rows:,}, rule_id_start={rule_id_start:,}")
        summary, by_year, by_side, by_class, by_segment = scan_family_r2a(
            samples,
            spec,
            config=config,
            rule_id_start=rule_id_start,
            progress_every=progress_every,
        )
        _write_part(summary, summary_path)
        _write_part(by_year, by_year_path)
        _write_part(by_side, by_side_path)
        _write_part(by_class, by_class_path)
        _write_part(by_segment, by_segment_path)

        completed += len(summary)
        rule_id_start += family_rows
        status.completed_combos = completed if limit_families else min(status.completed_combos + len(summary), total_combos)
        status.next_batch_start = status.completed_combos
        save_status(outdir, status)
        append_log(outdir, f"R2A done {spec.name}, summary_rows={len(summary):,}")

    paths = ensure_dirs(outdir)
    stats = write_r2a_final_outputs(
        outdir=paths["r2a"],
        checkpoint_dir=ckpt,
        samples=samples,
        config=config,
    )
    html_path = write_html_report(paths["r2a"], paths["r2a"] / "mtx_r2a_report.html")
    append_log(outdir, f"R2A html_report={html_path}")
    for key, value in stats.items():
        append_log(outdir, f"R2A {key}={value:,}")


def main() -> None:
    args = parse_args()
    config = ResearchConfig()
    paths = ensure_dirs(args.outdir)
    input_hash = file_hash(args.data)
    config_hash = stable_config_hash(asdict(config))

    append_log(args.outdir, f"run start data={args.data}")
    append_log(args.outdir, f"input_hash={input_hash} config_hash={config_hash}")
    append_log(args.outdir, f"R2A expected combos={config.r2a_expected_combo_count:,}")

    df, data_report = load_ohlcv(args.data, log=lambda msg: append_log(args.outdir, msg))
    samples, sample_report = build_r2a_samples(df, config)
    append_log(args.outdir, f"data datetime min={data_report.datetime_min} max={data_report.datetime_max}")
    append_log(args.outdir, f"data rows raw={data_report.raw_rows:,} cleaned={data_report.cleaned_rows:,}")
    append_log(args.outdir, f"samples={sample_report.sample_count:,} date={sample_report.date_min}~{sample_report.date_max}")

    stages = {stage.strip().lower() for stage in args.stages.split(",") if stage.strip()}
    if "r2a" in stages:
        run_r2a(
            data_path=args.data,
            outdir=args.outdir,
            samples=samples,
            config=config,
            resume=bool(args.resume),
            force=bool(args.force),
            progress_every=args.progress_every,
            limit_families=args.limit_families,
            input_hash=input_hash,
            config_hash=config_hash,
        )
    if "r2b" in stages:
        run_r2b(df=df, samples=samples, outdir=args.outdir, config=config)
    append_log(args.outdir, "run finished")
    print(f"Done. Outputs: {paths['r2a']}")


if __name__ == "__main__":
    main()
