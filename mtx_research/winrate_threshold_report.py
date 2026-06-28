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


def _fmt_int(value: float | int) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def _fmt_num(value: float | int, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    text = f"{value:,.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_pct(rate: float, digits: int = 1) -> str:
    if pd.isna(rate):
        return ""
    return f"{float(rate) * 100:,.{digits}f}%"


def _cls(value: float) -> str:
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "zero"


def _pf(gross_profit: float, gross_loss_abs: float) -> float:
    if gross_loss_abs <= 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss_abs


def _aggregate_total(summary: pd.DataFrame, cost: CostConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        selected = summary[(summary["TotalTrades"] > 0) & (summary["WinRate"] >= threshold)].copy()
        trades = float(selected["TotalTrades"].sum())
        wins = float(selected["WinTrades"].sum())
        net_points = float(selected["NetPoints"].sum())
        net_profit = float(selected["NetProfitTWD"].sum())
        mdd_points = float(selected["MaxDrawdownNetPoints"].sum())
        gross_profit = float(selected["GrossProfitNetPoints"].sum())
        gross_loss_abs = float((-selected["GrossLossNetPoints"]).sum())
        rows.append(
            {
                "Threshold": threshold,
                "ThresholdLabel": f">={int(threshold * 100)}%",
                "ParamCount": int(len(selected)),
                "TotalTrades": int(trades),
                "WinTrades": int(wins),
                "WinRate": wins / trades if trades else np.nan,
                "NetPoints": net_points,
                "NetProfitTWD": net_profit,
                "TotalReturnRate": net_profit / cost.capital_twd,
                "MDDNetPointsSum": mdd_points,
                "MDDTWDSum": mdd_points * cost.point_value_twd,
                "MDDRateSum": mdd_points * cost.point_value_twd / cost.capital_twd,
                "PFNet": _pf(gross_profit, gross_loss_abs),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_by_year(summary: pd.DataFrame, by_year: pd.DataFrame, cost: CostConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    years = sorted(int(y) for y in by_year["Year"].unique())
    for threshold in THRESHOLDS:
        selected = summary[(summary["TotalTrades"] > 0) & (summary["WinRate"] >= threshold)]
        selected_ids = set(int(x) for x in selected["RuleID"])
        selected_years = by_year[by_year["RuleID"].isin(selected_ids)]
        for year in years:
            part = selected_years[selected_years["Year"] == year]
            trades = float(part["Trades"].sum())
            wins = float((part["Trades"] * part["WinRate"]).sum())
            net_points = float(part["NetPoints"].sum())
            net_profit = float(part["NetProfitTWD"].sum())
            mdd_points = float(part["MaxDrawdownNetPoints"].sum())
            rows.append(
                {
                    "Threshold": threshold,
                    "ThresholdLabel": f">={int(threshold * 100)}%",
                    "Year": int(year),
                    "ParamCount": int(len(selected)),
                    "ActiveParamCount": int((part["Trades"] > 0).sum()),
                    "TotalTrades": int(trades),
                    "WinTrades": int(round(wins)),
                    "WinRate": wins / trades if trades else np.nan,
                    "NetPoints": net_points,
                    "NetProfitTWD": net_profit,
                    "TotalReturnRate": net_profit / cost.capital_twd,
                    "MDDNetPointsSum": mdd_points,
                    "MDDTWDSum": mdd_points * cost.point_value_twd,
                    "MDDRateSum": mdd_points * cost.point_value_twd / cost.capital_twd,
                }
            )
    return pd.DataFrame(rows)


def _summary_rows_html(df: pd.DataFrame) -> str:
    html: list[str] = []
    for row in df.itertuples(index=False):
        html.append(
            "<tr>"
            f"<td>{escape(row.ThresholdLabel)}</td>"
            f"<td>{_fmt_int(row.ParamCount)}</td>"
            f"<td>{_fmt_int(row.TotalTrades)}</td>"
            f"<td>{_fmt_pct(row.WinRate, 1)}</td>"
            f"<td class=\"{_cls(row.NetPoints)}\">{_fmt_num(row.NetPoints, 1)}</td>"
            f"<td class=\"{_cls(row.NetProfitTWD)}\">{_fmt_int(row.NetProfitTWD)}</td>"
            f"<td class=\"{_cls(row.TotalReturnRate)}\">{_fmt_pct(row.TotalReturnRate, 2)}</td>"
            f"<td>{_fmt_num(row.MDDNetPointsSum, 1)}</td>"
            f"<td>{_fmt_int(row.MDDTWDSum)}</td>"
            f"<td>{_fmt_pct(row.MDDRateSum, 2)}</td>"
            f"<td>{_fmt_num(row.PFNet, 2)}</td>"
            "</tr>"
        )
    return "\n".join(html)


def _year_rows_html(df: pd.DataFrame) -> str:
    html: list[str] = []
    for row in df.itertuples(index=False):
        html.append(
            "<tr>"
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
            f"<td>{_fmt_int(row.MDDTWDSum)}</td>"
            f"<td>{_fmt_pct(row.MDDRateSum, 2)}</td>"
            "</tr>"
        )
    return "\n".join(html)


def _write_html(total: pd.DataFrame, yearly: pd.DataFrame, output: Path, cost: CostConfig) -> None:
    output.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>勝率門檻彙總報表</title>
<style>
body{{font-family:"Microsoft JhengHei",Arial,sans-serif;margin:0;background:#f7faf8;color:#1d2823;font-size:16px}}
header{{padding:22px 10px;background:white;border-bottom:1px solid #dce7e1}}
h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:24px;margin:34px 0 10px}}
.sub{{color:#64736d;line-height:1.65;max-width:1180px}}
main{{padding:0 8px 36px}}
.note{{background:#fff9e8;border:1px solid #ead9a3;border-radius:6px;padding:10px 12px;margin:14px 0;line-height:1.55}}
.table-wrap{{overflow:auto;border:1px solid #dce7e1;background:white;box-shadow:0 10px 22px rgba(32,48,40,.06)}}
table{{border-collapse:collapse;width:max-content;min-width:100%;font-variant-numeric:tabular-nums}}
th,td{{border:1px solid #dfe8e3;padding:8px 10px;text-align:right;white-space:nowrap}}
th{{background:#dfebe5;color:#1d2a25;position:sticky;top:0;z-index:1}}
td:first-child,th:first-child{{text-align:left;position:sticky;left:0;background:inherit;z-index:2}}
tr:nth-child(even){{background:#f4f8f5}} tr:nth-child(odd){{background:white}}
.pos{{color:#bf4e3e;font-weight:700}} .neg{{color:#3b855b;font-weight:700}} .zero{{color:#55635e}}
.links a{{display:inline-block;margin:6px 8px 0 0;padding:8px 12px;border:1px solid #c9d9d1;border-radius:5px;text-decoration:none;color:#255d87;background:white;font-weight:700}}
</style>
</head>
<body>
<header>
<h1>勝率門檻彙總報表</h1>
<div class="sub">
資料來源：anchor_body_gap_bins_11152 的 summary / by_year CSV。<br>
門檻採用「總勝率 >= 50%、60%、70%、80%、90%、100%」篩選參數；成本已扣出場滑點 {cost.exit_slippage_points:g} 點、來回手續費 {cost.round_trip_fee_twd} 元與期交稅。
</div>
<div class="links">
<a href="anchor_body_gap_bins_report.html">回 11,152 組總表</a>
<a href="winrate_threshold_summary.csv">下載總年度 CSV</a>
<a href="winrate_threshold_by_year.csv">下載分年度 CSV</a>
</div>
</header>
<main>
<div class="note">
MDD 欄位是「符合該勝率門檻的每一組參數 MaxDrawdown 加總」，用來回答全部參數合計的壓力大小；它不是把所有逐筆交易重新疊成一條權益曲線後的真實組合 MDD。
</div>

<h2>總年度彙總</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>勝率門檻</th><th>符合參數</th><th>總次數</th><th>合計勝率</th><th>淨點數</th><th>淨損益</th><th>總報酬率</th><th>MDD 點數合計</th><th>MDD 金額合計</th><th>MDD 率合計</th><th>PF</th>
</tr></thead>
<tbody>
{_summary_rows_html(total)}
</tbody>
</table>
</div>

<h2>分年度彙總</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>勝率門檻</th><th>年度</th><th>符合參數</th><th>該年有交易參數</th><th>總次數</th><th>合計勝率</th><th>淨點數</th><th>淨損益</th><th>總報酬率</th><th>MDD 點數合計</th><th>MDD 金額合計</th><th>MDD 率合計</th>
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


def build_report(summary_csv: Path, by_year_csv: Path, output_dir: Path) -> dict[str, Path]:
    cost = CostConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    by_year = pd.read_csv(by_year_csv)

    total = _aggregate_total(summary, cost)
    yearly = _aggregate_by_year(summary, by_year, cost)

    total_csv = output_dir / "winrate_threshold_summary.csv"
    yearly_csv = output_dir / "winrate_threshold_by_year.csv"
    html = output_dir / "winrate_threshold_report.html"
    total.to_csv(total_csv, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_csv, index=False, encoding="utf-8-sig")
    _write_html(total, yearly, html, cost)
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
    args = parser.parse_args()
    paths = build_report(args.summary, args.by_year, args.outdir)
    print(f"summary={paths['summary']}")
    print(f"by_year={paths['by_year']}")
    print(f"html={paths['html']}")


if __name__ == "__main__":
    main()
