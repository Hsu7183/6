from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.config import CostConfig
from mtx_research.data_sources import DEFAULT_INSTRUMENT, DATA_SOURCES, cost_for_instrument


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def _threshold_label(threshold: float) -> str:
    if threshold >= 1.0:
        return "勝率 = 100%"
    return f"勝率 > {int(threshold * 100)}%"


def _select_by_threshold(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    valid = summary["TotalTrades"] > 0
    if threshold >= 1.0:
        return summary[valid & (summary["WinRate"] >= 1.0 - 1e-12)].copy()
    return summary[valid & (summary["WinRate"] > threshold)].copy()


def _pf(gross_profit: float, gross_loss_abs: float) -> float:
    if gross_loss_abs <= 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss_abs


def _fmt_int(value: float | int) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def _fmt_num(value: float | int, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    if value == np.inf:
        return "∞"
    value = float(value)
    text = f"{value:,.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fmt_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:,.{digits}f}%"


def _cls(value: float) -> str:
    if pd.isna(value):
        return ""
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "zero"


def aggregate_total(summary: pd.DataFrame, cost: CostConfig, layer_label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        selected = _select_by_threshold(summary, threshold)
        trades = float(selected["TotalTrades"].sum())
        wins = float(selected["WinTrades"].sum())
        net_points = float(selected["NetPoints"].sum())
        net_profit = float(selected["NetProfitTWD"].sum())
        gross_profit = float(selected["GrossProfitNetPoints"].sum())
        gross_loss_abs = float((-selected["GrossLossNetPoints"]).sum())
        mdd_points_sum = float(selected["MaxDrawdownNetPoints"].sum())
        mdd_points_max = float(selected["MaxDrawdownNetPoints"].max()) if len(selected) else 0.0
        rows.append(
            {
                "Layer": layer_label,
                "Threshold": threshold,
                "ThresholdLabel": _threshold_label(threshold),
                "ParamCount": int(len(selected)),
                "TotalTrades": int(trades),
                "WinTrades": int(wins),
                "WinRate": wins / trades if trades else np.nan,
                "NetPoints": net_points,
                "NetProfitTWD": net_profit,
                "TotalReturnRate": net_profit / cost.capital_twd,
                "MDDNetPointsSum": mdd_points_sum,
                "MDDRateSum": mdd_points_sum * cost.point_value_twd / cost.capital_twd,
                "MDDNetPointsMax": mdd_points_max,
                "MDDRateMax": mdd_points_max * cost.point_value_twd / cost.capital_twd,
                "PFNet": _pf(gross_profit, gross_loss_abs),
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_year(
    summary: pd.DataFrame,
    by_year: pd.DataFrame,
    cost: CostConfig,
    layer_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    years = sorted(int(y) for y in by_year["Year"].unique())
    for threshold in THRESHOLDS:
        selected = _select_by_threshold(summary, threshold)
        selected_ids = set(int(x) for x in selected["RuleID"])
        selected_years = by_year[by_year["RuleID"].isin(selected_ids)]
        for year in years:
            part = selected_years[selected_years["Year"] == year]
            trades = float(part["Trades"].sum())
            wins = float((part["Trades"] * part["WinRate"].fillna(0)).sum())
            net_points = float(part["NetPoints"].sum())
            net_profit = float(part["NetProfitTWD"].sum())
            mdd_points_sum = float(part["MaxDrawdownNetPoints"].sum())
            mdd_points_max = float(part["MaxDrawdownNetPoints"].max()) if len(part) else 0.0
            rows.append(
                {
                    "Layer": layer_label,
                    "Threshold": threshold,
                    "ThresholdLabel": _threshold_label(threshold),
                    "Year": int(year),
                    "ParamCount": int(len(selected)),
                    "ActiveParamCount": int((part["Trades"] > 0).sum()),
                    "TotalTrades": int(trades),
                    "WinTrades": int(round(wins)),
                    "WinRate": wins / trades if trades else np.nan,
                    "NetPoints": net_points,
                    "NetProfitTWD": net_profit,
                    "TotalReturnRate": net_profit / cost.capital_twd,
                    "MDDNetPointsSum": mdd_points_sum,
                    "MDDRateSum": mdd_points_sum * cost.point_value_twd / cost.capital_twd,
                    "MDDNetPointsMax": mdd_points_max,
                    "MDDRateMax": mdd_points_max * cost.point_value_twd / cost.capital_twd,
                }
            )
    return pd.DataFrame(rows)


def _date_label(value: int | float | str) -> str:
    text = str(int(float(value)))
    if len(text) != 8:
        return text
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _scale_y(value: float, ymin: float, ymax: float, top: float, height: float) -> float:
    if ymax <= ymin:
        return top + height / 2
    return top + (ymax - value) / (ymax - ymin) * height


def _chart_domain(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    ymin = min(values + [0.0])
    ymax = max(values + [0.0])
    if ymin == ymax:
        pad = max(abs(ymin) * 0.1, 1.0)
        return ymin - pad, ymax + pad
    pad = (ymax - ymin) * 0.08
    return ymin - pad, ymax + pad


def _line_points(values: list[float], ymin: float, ymax: float, width: int, height: int) -> str:
    left, top, chart_w, chart_h = 62, 18, width - 82, height - 58
    if len(values) <= 1:
        x_step = 0.0
    else:
        x_step = chart_w / (len(values) - 1)
    points = []
    for idx, value in enumerate(values):
        x = left + idx * x_step
        y = _scale_y(value, ymin, ymax, top, chart_h)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _equity_svg(part: pd.DataFrame, title: str) -> str:
    width, height = 980, 300
    cum_net = [float(x) for x in part["CumNetTWD"]]
    cum_long = [float(x) for x in part["CumLongTWD"]]
    cum_short = [float(x) for x in part["CumShortTWD"]]
    ymin, ymax = _chart_domain(cum_net + cum_long + cum_short)
    zero_y = _scale_y(0, ymin, ymax, 18, height - 58)
    first_date = _date_label(part["Date"].iloc[0])
    last_date = _date_label(part["Date"].iloc[-1])
    final_net = cum_net[-1] if cum_net else 0.0
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)} 累積獲利折線圖">
  <rect x="0" y="0" width="{width}" height="{height}" rx="6" fill="#ffffff"/>
  <text x="16" y="26" class="chart-title">{escape(title)}｜累積獲利</text>
  <text x="{width - 16}" y="26" text-anchor="end" class="chart-note">{escape(first_date)} ~ {escape(last_date)}｜總累積 {_fmt_int(final_net)} 元</text>
  <line x1="62" x2="{width - 20}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" class="axis-zero"/>
  <text x="12" y="54" class="axis-label">{escape(_fmt_int(ymax))}</text>
  <text x="12" y="{height - 48}" class="axis-label">{escape(_fmt_int(ymin))}</text>
  <polyline points="{_line_points(cum_net, ymin, ymax, width, height)}" class="line-net"/>
  <polyline points="{_line_points(cum_long, ymin, ymax, width, height)}" class="line-long"/>
  <polyline points="{_line_points(cum_short, ymin, ymax, width, height)}" class="line-short"/>
  <g transform="translate(62 {height - 26})">
    <text class="legend net">總累積</text>
    <text x="96" class="legend long">做多累積</text>
    <text x="214" class="legend short">做空累積</text>
  </g>
</svg>
"""


def _daily_bar_svg(part: pd.DataFrame, title: str) -> str:
    width, height = 980, 260
    left, top, chart_w, chart_h = 62, 18, width - 82, height - 58
    values = [float(x) for x in part["DailyNetTWD"]]
    max_abs = max([abs(x) for x in values] + [1.0])
    ymin, ymax = -max_abs * 1.08, max_abs * 1.08
    zero_y = _scale_y(0, ymin, ymax, top, chart_h)
    x_step = chart_w / max(len(values), 1)
    stroke_w = max(0.6, min(7.0, x_step * 0.72))
    lines = []
    for idx, value in enumerate(values):
        x = left + idx * x_step + x_step / 2
        y = _scale_y(value, ymin, ymax, top, chart_h)
        cls = "bar-pos" if value >= 0 else "bar-neg"
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{zero_y:.1f}" y2="{y:.1f}" class="{cls}" stroke-width="{stroke_w:.2f}"/>')
    best = max(values) if values else 0.0
    worst = min(values) if values else 0.0
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)} 每日損益柱狀圖">
  <rect x="0" y="0" width="{width}" height="{height}" rx="6" fill="#ffffff"/>
  <text x="16" y="26" class="chart-title">{escape(title)}｜每日損益</text>
  <text x="{width - 16}" y="26" text-anchor="end" class="chart-note">最好 {_fmt_int(best)} 元｜最差 {_fmt_int(worst)} 元</text>
  <line x1="{left}" x2="{left + chart_w}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" class="axis-zero"/>
  {"".join(lines)}
</svg>
"""


def _charts_html(daily: pd.DataFrame | None) -> str:
    if daily is None or daily.empty:
        return """
<h2>每日圖表</h2>
<div class="note">目前沒有每日圖表資料；請重新執行矩陣掃描以產生 winrate_threshold_daily.csv。</div>
"""

    work = daily.copy()
    if "Layer" not in work.columns:
        work["Layer"] = "策略層"
    work["Date"] = work["Date"].astype(int)
    work = work.sort_values(["Layer", "Threshold", "Date"])
    cards: list[str] = []
    for (layer, label), part in work.groupby(["Layer", "ThresholdLabel"], sort=False):
        part = part.sort_values("Date")
        title = f"{layer}｜{label}"
        cards.append(
            "<section class=\"chart-card\">"
            f"<h3>{escape(title)}</h3>"
            f"{_equity_svg(part, title)}"
            f"{_daily_bar_svg(part, title)}"
            "</section>"
        )
    return f"""
<h2>每日圖表</h2>
<div class="sub">折線圖：總累積獲利、做多累積獲利、做空累積獲利。柱狀圖：每日損益，紅色為獲利、綠色為虧損。</div>
<div class="chart-grid">
{''.join(cards)}
</div>
"""


def _total_rows_html(df: pd.DataFrame) -> str:
    rows: list[str] = []
    for row in df.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.Layer))}</td>"
            f"<td>{escape(row.ThresholdLabel)}</td>"
            f"<td>{_fmt_int(row.ParamCount)}</td>"
            f"<td>{_fmt_int(row.TotalTrades)}</td>"
            f"<td>{_fmt_pct(row.WinRate, 1)}</td>"
            f"<td class=\"{_cls(row.NetPoints)}\">{_fmt_num(row.NetPoints, 1)}</td>"
            f"<td class=\"{_cls(row.NetProfitTWD)}\">{_fmt_int(row.NetProfitTWD)}</td>"
            f"<td class=\"{_cls(row.TotalReturnRate)}\">{_fmt_pct(row.TotalReturnRate, 2)}</td>"
            f"<td>{_fmt_num(row.MDDNetPointsSum, 1)}</td>"
            f"<td>{_fmt_pct(row.MDDRateSum, 2)}</td>"
            f"<td>{_fmt_num(row.MDDNetPointsMax, 1)}</td>"
            f"<td>{_fmt_pct(row.MDDRateMax, 2)}</td>"
            f"<td>{_fmt_num(row.PFNet, 2)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _year_rows_html(df: pd.DataFrame) -> str:
    rows: list[str] = []
    for row in df.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.Layer))}</td>"
            f"<td>{escape(row.ThresholdLabel)}</td>"
            f"<td>{row.Year}</td>"
            f"<td>{_fmt_int(row.ParamCount)}</td>"
            f"<td>{_fmt_int(row.ActiveParamCount)}</td>"
            f"<td>{_fmt_int(row.TotalTrades)}</td>"
            f"<td>{_fmt_pct(row.WinRate, 1)}</td>"
            f"<td class=\"{_cls(row.NetPoints)}\">{_fmt_num(row.NetPoints, 1)}</td>"
            f"<td class=\"{_cls(row.NetProfitTWD)}\">{_fmt_int(row.NetProfitTWD)}</td>"
            f"<td class=\"{_cls(row.TotalReturnRate)}\">{_fmt_pct(row.TotalReturnRate, 2)}</td>"
            f"<td>{_fmt_num(row.MDDNetPointsSum, 1)}</td>"
            f"<td>{_fmt_pct(row.MDDRateSum, 2)}</td>"
            f"<td>{_fmt_num(row.MDDNetPointsMax, 1)}</td>"
            f"<td>{_fmt_pct(row.MDDRateMax, 2)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_html(
    total: pd.DataFrame,
    yearly: pd.DataFrame,
    output: Path,
    *,
    title: str,
    description: str,
    matrix_href: str | None = None,
    daily: pd.DataFrame | None = None,
) -> None:
    links = []
    if matrix_href:
        links.append(f'<a href="{escape(matrix_href)}">參數矩陣</a>')
    links.extend(
        [
            '<a href="winrate_threshold_summary.csv">總年度 CSV</a>',
            '<a href="winrate_threshold_by_year.csv">分年度 CSV</a>',
            '<a href="winrate_threshold_daily.csv">每日圖表資料 CSV</a>',
        ]
    )
    output.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body{{font-family:"Microsoft JhengHei",Arial,sans-serif;margin:0;background:#f6faf8;color:#1d2823;font-size:18px}}
header{{padding:22px 10px;background:#fff;border-bottom:1px solid #dbe8e2}}
main{{padding:0 8px 36px}}
h1{{font-size:32px;margin:0 0 10px}} h2{{font-size:25px;margin:34px 0 10px}} h3{{font-size:21px;margin:0 0 10px}}
.sub{{color:#5e6f68;line-height:1.65;max-width:1400px}}
.links a{{display:inline-block;margin:10px 8px 0 0;padding:8px 12px;border:1px solid #c9d9d1;border-radius:5px;text-decoration:none;color:#255d87;background:#fff;font-weight:700}}
.note{{background:#fff9e8;border:1px solid #ead9a3;border-radius:6px;padding:10px 12px;margin:14px 0;line-height:1.6}}
.table-wrap{{overflow:auto;border:1px solid #dce7e1;background:#fff;box-shadow:0 10px 22px rgba(32,48,40,.06)}}
table{{border-collapse:collapse;width:max-content;min-width:100%;font-variant-numeric:tabular-nums}}
th,td{{border:1px solid #dfe8e3;padding:8px 10px;text-align:right;white-space:nowrap}}
th{{background:#dfebe5;color:#1d2a25;position:sticky;top:0;z-index:1}}
td:first-child,th:first-child{{text-align:left;position:sticky;left:0;background:inherit;z-index:2}}
tr:nth-child(even){{background:#f3f8f5}} tr:nth-child(odd){{background:#fff}}
.pos{{color:#bf4e3e;font-weight:700}} .neg{{color:#2f7d56;font-weight:700}} .zero{{color:#56645f}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(680px,1fr));gap:14px;margin-top:12px}}
.chart-card{{background:#fff;border:1px solid #dce7e1;border-radius:6px;padding:12px;box-shadow:0 10px 22px rgba(32,48,40,.05)}}
.chart{{width:100%;height:auto;display:block;margin-top:8px;border:1px solid #edf3ef;border-radius:6px}}
.chart-title{{font-size:18px;font-weight:700;fill:#1d2823}}
.chart-note,.axis-label{{font-size:13px;fill:#6b7b74}}
.axis-zero{{stroke:#87948f;stroke-width:1;stroke-dasharray:4 4}}
.line-net{{fill:none;stroke:#2d5f8f;stroke-width:2.3}}
.line-long{{fill:none;stroke:#bf4e3e;stroke-width:1.7}}
.line-short{{fill:none;stroke:#2f7d56;stroke-width:1.7}}
.bar-pos{{stroke:#bf4e3e}} .bar-neg{{stroke:#2f7d56}}
.legend{{font-size:14px;font-weight:700}} .legend.net{{fill:#2d5f8f}} .legend.long{{fill:#bf4e3e}} .legend.short{{fill:#2f7d56}}
</style>
</head>
<body>
<header>
<h1>{escape(title)}</h1>
<div class="sub">{escape(description)}</div>
<div class="links">{"".join(links)}</div>
</header>
<main>
<div class="note">門檻統計使用扣成本後勝率：小台與大台分別套用自己的本金、點值、手續費、出場 2 點滑點與期交稅。MDD 表格同時保留所有入選策略加總 MDD，以及單一策略最大 MDD。</div>

{_charts_html(daily)}

<h2>總年度門檻總表</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>層次</th><th>勝率門檻</th><th>參數組數</th><th>總次數</th><th>總勝率</th><th>淨點數</th><th>淨損益</th><th>總報酬率</th><th>MDD 加總</th><th>MDD 加總率</th><th>單組最大 MDD</th><th>單組最大 MDD 率</th><th>PF</th>
</tr></thead>
<tbody>
{_total_rows_html(total)}
</tbody>
</table>
</div>

<h2>分年度門檻總表</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>層次</th><th>勝率門檻</th><th>年度</th><th>參數組數</th><th>有交易組數</th><th>總次數</th><th>勝率</th><th>淨點數</th><th>淨損益</th><th>報酬率</th><th>MDD 加總</th><th>MDD 加總率</th><th>單組最大 MDD</th><th>單組最大 MDD 率</th>
</tr></thead>
<tbody>
{_year_rows_html(yearly)}
</tbody>
</table>
</div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_report(
    summary_csv: Path,
    by_year_csv: Path,
    output_dir: Path,
    *,
    cost: CostConfig | None = None,
    layer_label: str = "策略層",
    title: str | None = None,
    description: str | None = None,
    matrix_href: str | None = "anchor_body_gap_bins_report.html",
    daily_csv: Path | None = None,
) -> dict[str, Path]:
    cost = cost or CostConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    by_year = pd.read_csv(by_year_csv)

    total = aggregate_total(summary, cost, layer_label)
    yearly = aggregate_by_year(summary, by_year, cost, layer_label)
    daily: pd.DataFrame | None = None
    daily_out = output_dir / "winrate_threshold_daily.csv"
    if daily_csv is not None and daily_csv.exists():
        daily = pd.read_csv(daily_csv)
        if "Layer" not in daily.columns:
            daily.insert(0, "Layer", layer_label)
        daily.to_csv(daily_out, index=False, encoding="utf-8-sig")
    elif not daily_out.exists():
        pd.DataFrame(
            columns=[
                "Layer",
                "Threshold",
                "ThresholdLabel",
                "Date",
                "DailyNetTWD",
                "DailyLongTWD",
                "DailyShortTWD",
                "CumNetTWD",
                "CumLongTWD",
                "CumShortTWD",
            ]
        ).to_csv(daily_out, index=False, encoding="utf-8-sig")

    total_csv = output_dir / "winrate_threshold_summary.csv"
    yearly_csv = output_dir / "winrate_threshold_by_year.csv"
    html = output_dir / "winrate_threshold_report.html"
    total.to_csv(total_csv, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_csv, index=False, encoding="utf-8-sig")
    write_html(
        total,
        yearly,
        html,
        title=title or f"{layer_label} 勝率門檻總表",
        description=description or "依照參數矩陣結果，把勝率門檻策略彙整成總年度、分年度與每日圖表。",
        matrix_href=matrix_href,
        daily=daily,
    )
    return {"summary": total_csv, "by_year": yearly_csv, "daily": daily_out, "html": html}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build win-rate threshold aggregation report.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("report_outputs") / "anchor_body_gap_bins_11152" / "summary_anchor_body_gap_bins.csv",
    )
    parser.add_argument(
        "--by-year",
        type=Path,
        default=Path("report_outputs") / "anchor_body_gap_bins_11152" / "by_year_anchor_body_gap_bins.csv",
    )
    parser.add_argument("--daily", type=Path)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("report_outputs") / "anchor_body_gap_bins_11152",
    )
    parser.add_argument(
        "--instrument",
        choices=sorted(DATA_SOURCES),
        default=DEFAULT_INSTRUMENT,
        help="Use the matching point value, fee, slippage, and tax settings.",
    )
    parser.add_argument("--layer-label", default="策略層")
    args = parser.parse_args()
    paths = build_report(
        args.summary,
        args.by_year,
        args.outdir,
        cost=cost_for_instrument(args.instrument),
        layer_label=args.layer_label,
        daily_csv=args.daily,
    )
    print(f"summary={paths['summary']}")
    print(f"by_year={paths['by_year']}")
    print(f"daily={paths['daily']}")
    print(f"html={paths['html']}")


if __name__ == "__main__":
    main()
