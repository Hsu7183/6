from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.anchor_body_bins import EXPECTED_COMBOS, scan
from mtx_research.data_sources import resolve_data_path
from mtx_research.session_layers import resolve_session
from mtx_research.winrate_threshold_report import build_report, write_html


@dataclass(frozen=True)
class Layer:
    key: str
    label: str
    instrument: str
    session: str


LAYERS = (
    Layer("mtx_day", "小台日盤", "mtx", "day"),
    Layer("mtx_all", "小台全日", "mtx", "all"),
    Layer("tx_day", "大台日盤", "tx", "day"),
    Layer("tx_all", "大台全日", "tx", "all"),
)


def _required_outputs(outdir: Path) -> tuple[Path, Path, Path]:
    return (
        outdir / "summary_anchor_body_gap_bins.csv",
        outdir / "by_year_anchor_body_gap_bins.csv",
        outdir / "anchor_body_gap_bins_report.html",
    )


def _scan_or_reuse(layer: Layer, outdir: Path, progress_every: int, skip_existing: bool) -> dict[str, Path]:
    summary, by_year, html = _required_outputs(outdir)
    if skip_existing and summary.exists() and by_year.exists() and html.exists():
        print(f"[{layer.label}] reuse existing matrix: {outdir}")
        return {"summary": summary, "by_year": by_year, "html": html}

    data_path = resolve_data_path(layer.instrument)
    session = resolve_session(layer.session)
    print("=" * 72)
    print(f"[{layer.label}] start matrix")
    print(f"data={data_path}")
    print(f"session={session.key} ({session.label})")
    print(f"outdir={outdir}")
    return scan(data_path, outdir, params=session.params, progress_every=progress_every)


def _build_layer_threshold(layer: Layer, paths: dict[str, Path], outdir: Path) -> dict[str, Path]:
    print(f"[{layer.label}] build threshold report")
    return build_report(
        paths["summary"],
        paths["by_year"],
        outdir,
        layer_label=layer.label,
        title=f"{layer.label} 勝率門檻總表",
        description=(
            f"{layer.label}：{EXPECTED_COMBOS:,} 組參數矩陣，依總年勝率篩選 >50%、>60%、>70%、>80%、>90%、=100% 的策略。"
        ),
        matrix_href="anchor_body_gap_bins_report.html",
    )


def _write_index(root: Path, layer_outputs: list[tuple[Layer, Path, Path]]) -> Path:
    rows = []
    for layer, matrix_html, threshold_html in layer_outputs:
        rows.append(
            "<tr>"
            f"<td>{layer.label}</td>"
            f"<td>{layer.instrument}</td>"
            f"<td>{layer.session}</td>"
            f"<td><a href=\"{matrix_html.relative_to(root).as_posix()}\">參數矩陣</a></td>"
            f"<td><a href=\"{threshold_html.relative_to(root).as_posix()}\">勝率門檻總表</a></td>"
            "</tr>"
        )
    html = root / "four_layer_matrix_index.html"
    html.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>四層策略參數矩陣入口</title>
<style>
body{{font-family:"Microsoft JhengHei",Arial,sans-serif;margin:0;background:#f6faf8;color:#1d2823;font-size:18px}}
header{{padding:22px 10px;background:#fff;border-bottom:1px solid #dbe8e2}}
main{{padding:20px 8px 40px}}
h1{{font-size:32px;margin:0 0 10px}}
.sub{{color:#5e6f68;line-height:1.65}}
table{{border-collapse:collapse;min-width:920px;background:#fff}}
th,td{{border:1px solid #dfe8e3;padding:10px 12px;text-align:left}}
th{{background:#dfebe5}}
a{{color:#255d87;font-weight:700;text-decoration:none}}
</style>
</head>
<body>
<header>
<h1>四層策略參數矩陣入口</h1>
<div class="sub">小台日盤、小台全日、大台日盤、大台全日。每層都有參數矩陣與勝率門檻總表。</div>
</header>
<main>
<p><a href="four_layer_threshold_report.html">開啟四層合併勝率門檻總表</a></p>
<table>
<thead><tr><th>層次</th><th>商品</th><th>時段</th><th>參數矩陣</th><th>勝率門檻總表</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html


def run_all(
    outdir: Path,
    *,
    only: set[str] | None = None,
    progress_every: int = 100_000,
    skip_existing: bool = False,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    selected_layers = [layer for layer in LAYERS if only is None or layer.key in only]
    if not selected_layers:
        raise ValueError("no layers selected")

    total_parts: list[pd.DataFrame] = []
    yearly_parts: list[pd.DataFrame] = []
    layer_links: list[tuple[Layer, Path, Path]] = []

    for layer in selected_layers:
        layer_outdir = outdir / layer.key
        matrix_paths = _scan_or_reuse(layer, layer_outdir, progress_every, skip_existing)
        threshold_paths = _build_layer_threshold(layer, matrix_paths, layer_outdir)
        total_parts.append(pd.read_csv(threshold_paths["summary"]))
        yearly_parts.append(pd.read_csv(threshold_paths["by_year"]))
        layer_links.append((layer, matrix_paths["html"], threshold_paths["html"]))

    total = pd.concat(total_parts, ignore_index=True)
    yearly = pd.concat(yearly_parts, ignore_index=True)
    total_csv = outdir / "winrate_threshold_summary.csv"
    yearly_csv = outdir / "winrate_threshold_by_year.csv"
    combined_html = outdir / "four_layer_threshold_report.html"
    total.to_csv(total_csv, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_csv, index=False, encoding="utf-8-sig")
    write_html(
        total,
        yearly,
        combined_html,
        title="四層勝率門檻總表",
        description=(
            "合併小台日盤、小台全日、大台日盤、大台全日。"
            "每層先跑同一套 11,152 組參數矩陣，再依總年勝率門檻彙總總年與分年度績效。"
        ),
        matrix_href=None,
    )
    index_html = _write_index(outdir, layer_links)
    return {
        "index": index_html,
        "combined_html": combined_html,
        "summary": total_csv,
        "by_year": yearly_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four-layer strategy matrix and win-rate threshold reports.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("report_outputs") / "four_layer_matrix",
    )
    parser.add_argument("--only", nargs="*", choices=[layer.key for layer in LAYERS])
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    paths = run_all(
        args.outdir,
        only=set(args.only) if args.only else None,
        progress_every=args.progress_every,
        skip_existing=args.skip_existing,
    )
    print("=" * 72)
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
