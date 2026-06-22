from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StageStatus:
    stage: str
    total_combos: int
    completed_combos: int
    next_batch_start: int
    started_at: str
    updated_at: str
    input_data_hash: str
    config_hash: str


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs(base_outdir: Path) -> dict[str, Path]:
    paths = {
        "r2a": base_outdir / "r2a_1k_trend_pullback_all_families",
        "r2b": base_outdir / "r2b_exit_universe",
        "logs": base_outdir / "logs",
        "checkpoints": base_outdir / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def stage_checkpoint_dir(base_outdir: Path, stage: str) -> Path:
    path = base_outdir / "checkpoints" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def status_path(base_outdir: Path, stage: str) -> Path:
    return stage_checkpoint_dir(base_outdir, stage) / "status.json"


def load_status(base_outdir: Path, stage: str) -> StageStatus | None:
    path = status_path(base_outdir, stage)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return StageStatus(**data)


def save_status(base_outdir: Path, status: StageStatus) -> None:
    path = status_path(base_outdir, status.stage)
    status.updated_at = now_text()
    path.write_text(json.dumps(asdict(status), ensure_ascii=False, indent=2), encoding="utf-8")


def new_status(
    *,
    base_outdir: Path,
    stage: str,
    total_combos: int,
    input_data_hash: str,
    config_hash: str,
) -> StageStatus:
    started = now_text()
    status = StageStatus(
        stage=stage,
        total_combos=total_combos,
        completed_combos=0,
        next_batch_start=0,
        started_at=started,
        updated_at=started,
        input_data_hash=input_data_hash,
        config_hash=config_hash,
    )
    save_status(base_outdir, status)
    return status


def reset_stage(base_outdir: Path, stage: str) -> None:
    ckpt = stage_checkpoint_dir(base_outdir, stage)
    for file in ckpt.glob("*.csv"):
        file.unlink()
    path = status_path(base_outdir, stage)
    if path.exists():
        path.unlink()


def append_log(base_outdir: Path, message: str) -> None:
    log_dir = base_outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_text()}] {message}\n")


def stable_config_hash(payload: dict[str, Any]) -> str:
    import hashlib

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()[:16]
