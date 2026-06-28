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
) -> None:
    links = []
    if matrix_href:
        links.append(f'<a href="{escape(matrix_href)}">回參數矩陣</a>')
    links.extend(
        [
            '<a href="winrate_threshold_summary.csv">下載總年 CSV</a>',
            '<a href="winrate_threshold_by_year.csv">下載分年度 CSV</a>',
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
h1{{font-size:32px;margin:0 0 10px}} h2{{font-size:25px;margin:34px 0 10px}}
.sub{{color:#5e6f68;line-height:1.65;max-width:1280px}}
.links a{{display:inline-block;margin:10px 8px 0 0;padding:8px 12px;border:1px solid #c9d9d1;border-radius:5px;text-decoration:none;color:#255d87;background:#fff;font-weight:700}}
.note{{background:#fff9e8;border:1px solid #ead9a3;border-radius:6px;padding:10px 12px;margin:14px 0;line-height:1.6}}
.table-wrap{{overflow:auto;border:1px solid #dce7e1;background:#fff;box-shadow:0 10px 22px rgba(32,48,40,.06)}}
table{{border-collapse:collapse;width:max-content;min-width:100%;font-variant-numeric:tabular-nums}}
th,td{{border:1px solid #dfe8e3;padding:8px 10px;text-align:right;white-space:nowrap}}
th{{background:#dfebe5;color:#1d2a25;position:sticky;top:0;z-index:1}}
td:first-child,th:first-child{{text-align:left;position:sticky;left:0;background:inherit;z-index:2}}
tr:nth-child(even){{background:#f3f8f5}} tr:nth-child(odd){{background:#fff}}
.pos{{color:#bf4e3e;font-weight:700}} .neg{{color:#2f7d56;font-weight:700}} .zero{{color:#56645f}}
</style>
</head>
<body>
<header>
<h1>{escape(title)}</h1>
<div class="sub">{escape(description)}</div>
<div class="links">{"".join(links)}</div>
</header>
<main>
<div class="note">門檻規則：50% 到 90% 使用嚴格大於；100% 使用勝率等於 100%。MDD 合計是把入選策略各自最大回撤加總，最大單策略 MDD 則是入選策略中最大的單一策略回撤。</div>

<h2>總年門檻總表</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>層次</th><th>勝率門檻</th><th>策略數</th><th>總次數</th><th>整體勝率</th><th>淨點數</th><th>淨損益</th><th>總報酬率</th><th>MDD 合計</th><th>MDD 合計率</th><th>最大單策略 MDD</th><th>最大單策略 MDD 率</th><th>PF</th>
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
<th>層次</th><th>勝率門檻</th><th>年度</th><th>策略數</th><th>該年有交易策略數</th><th>總次數</th><th>整體勝率</th><th>淨點數</th><th>淨損益</th><th>總報酬率</th><th>MDD 合計</th><th>MDD 合計率</th><th>最大單策略 MDD</th><th>最大單策略 MDD 率</th>
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
    layer_label: str = "單一層次",
    title: str | None = None,
    description: str | None = None,
    matrix_href: str | None = "anchor_body_gap_bins_report.html",
) -> dict[str, Path]:
    cost = CostConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    by_year = pd.read_csv(by_year_csv)

    total = aggregate_total(summary, cost, layer_label)
    yearly = aggregate_by_year(summary, by_year, cost, layer_label)

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
        description=description or "依參數矩陣總年勝率篩選策略後，彙總總年與分年度績效。",
        matrix_href=matrix_href,
    )
    return {"summary": total_csv, "by_year": yearly_csv, "html": html}


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
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("report_outputs") / "anchor_body_gap_bins_11152",
    )
    parser.add_argument("--layer-label", default="單一層次")
    args = parser.parse_args()
    paths = build_report(args.summary, args.by_year, args.outdir, layer_label=args.layer_label)
    print(f"summary={paths['summary']}")
    print(f"by_year={paths['by_year']}")
    print(f"html={paths['html']}")


if __name__ == "__main__":
    main()
