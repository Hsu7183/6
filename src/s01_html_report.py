from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = list(range(2020, 2027))
TOP_FILES = {
    "top_robust.csv": "穩健排行",
    "top_net.csv": "淨點排行",
    "top_pf.csv": "PF排行",
    "top_avg.csv": "平均點排行",
    "top_pullback_advantage.csv": "回踩優勢排行",
}
CAPITAL_TWD = 250_000
POINT_VALUE_TWD = 50


def _fmt_num(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    if value == np.inf:
        return "inf"
    if value == -np.inf:
        return "-inf"
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def _fmt_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%".rstrip("0").rstrip(".").replace(".%", "%")


def _read_top_rows(outdir: Path) -> pd.DataFrame:
    parts = []
    for filename, label in TOP_FILES.items():
        path = outdir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["Source"] = label
        df["SourceOrder"] = len(parts)
        df["RankInSource"] = np.arange(1, len(df) + 1)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no top csv files found in {outdir}")
    combined = pd.concat(parts, ignore_index=True)
    combined["SourceList"] = combined.groupby("RuleID")["Source"].transform(lambda values: "、".join(dict.fromkeys(values)))
    combined["BestSourceOrder"] = combined.groupby("RuleID")["SourceOrder"].transform("min")
    combined["BestRankInSource"] = combined.groupby("RuleID")["RankInSource"].transform("min")
    combined = combined.sort_values(["BestSourceOrder", "BestRankInSource", "RuleID"])
    combined = combined.drop_duplicates("RuleID", keep="first").reset_index(drop=True)
    return combined


def _read_year_rows(outdir: Path, rule_ids: set[int]) -> pd.DataFrame:
    by_year_path = outdir / "by_year.csv"
    if not by_year_path.exists():
        raise FileNotFoundError(by_year_path)
    parts = []
    for chunk in pd.read_csv(by_year_path, encoding="utf-8-sig", chunksize=200_000):
        part = chunk[chunk["RuleID"].isin(rule_ids)]
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _read_run_log(outdir: Path) -> dict[str, str]:
    path = outdir / "run_log.txt"
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _year_map(by_year: pd.DataFrame) -> dict[tuple[int, int], dict[str, object]]:
    mapping: dict[tuple[int, int], dict[str, object]] = {}
    if by_year.empty:
        return mapping
    for row in by_year.itertuples(index=False):
        mapping[(int(row.RuleID), int(row.Year))] = {
            "trades": int(row.Trades),
            "winRate": float(row.WinRate) if not pd.isna(row.WinRate) else None,
            "net": float(row.NetPoints),
            "avg": float(row.AvgPoints) if not pd.isna(row.AvgPoints) else None,
            "pf": float(row.PF) if not pd.isna(row.PF) else None,
            "mdd": float(row.MaxDrawdownPoints) if not pd.isna(row.MaxDrawdownPoints) else None,
        }
    return mapping


def _formula_text(row: pd.Series, side: str) -> str:
    if side == "long":
        return (
            f"C1>O1; Range<= {row.RangeMax}; Body>= {row.BodyMin}; "
            f"Body%>= {row.BodyPctMin}; ClosePos%>= {row.ClosePosMin}; "
            f"OppTail%<= {row.UpperTailMax}; "
            f"C1+{row.OpenGapMin} <= O0 <= C1+{row.OpenGapMax}; "
            f"L0 <= C1-{row.Penetrate}"
        )
    return (
        f"C1<O1; Range<= {row.RangeMax}; Body>= {row.BodyMin}; "
        f"Body%>= {row.BodyPctMin}; ClosePos%<= {100 - row.ClosePosMin}; "
        f"OppTail%<= {row.UpperTailMax}; "
        f"C1-{row.OpenGapMax} <= O0 <= C1-{row.OpenGapMin}; "
        f"H0 >= C1+{row.Penetrate}"
    )


def _build_rows(summary: pd.DataFrame, by_year: pd.DataFrame) -> list[dict[str, object]]:
    years = _year_map(by_year)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(summary.itertuples(index=False), start=1):
        series = pd.Series(row._asdict())
        yearly = {}
        for year in YEARS:
            yearly[str(year)] = years.get((int(row.RuleID), year))
        rows.append(
            {
                "no": index,
                "ruleId": int(row.RuleID),
                "source": row.SourceList,
                "formulaLong": _formula_text(series, "long"),
                "formulaShort": _formula_text(series, "short"),
                "params": (
                    f"BM={_fmt_num(row.BodyMin)} / RM={_fmt_num(row.RangeMax)} / "
                    f"BP={_fmt_num(row.BodyPctMin)} / CP={_fmt_num(row.ClosePosMin)} / "
                    f"UT={_fmt_num(row.UpperTailMax)} / Eff={_fmt_num(row.EffOppTailMax)} / "
                    f"OG={_fmt_num(row.OpenGapMin)}~{_fmt_num(row.OpenGapMax)} / P={_fmt_num(row.Penetrate)}"
                ),
                "bodyMin": float(row.BodyMin),
                "rangeMax": float(row.RangeMax),
                "bodyPctMin": float(row.BodyPctMin),
                "closePosMin": float(row.ClosePosMin),
                "upperTailMax": float(row.UpperTailMax),
                "openGapMin": float(row.OpenGapMin),
                "openGapMax": float(row.OpenGapMax),
                "penetrate": float(row.Penetrate),
                "rawTriggers": int(row.RawTriggerCount),
                "eligibleTriggers": int(row.EligibleTriggerCount),
                "fillCount": int(row.FillCount),
                "fillRate": float(row.FillRate) if not pd.isna(row.FillRate) else None,
                "totalTrades": int(row.TotalTrades),
                "winRate": float(row.WinRate) if not pd.isna(row.WinRate) else None,
                "netPoints": float(row.NetPoints),
                "netProfitTwd": (
                    float(row.NetProfitTWD)
                    if hasattr(row, "NetProfitTWD") and not pd.isna(row.NetProfitTWD)
                    else float(row.NetPoints) * POINT_VALUE_TWD
                ),
                "totalReturnRate": (
                    float(row.TotalReturnRate)
                    if hasattr(row, "TotalReturnRate") and not pd.isna(row.TotalReturnRate)
                    else float(row.NetPoints) * POINT_VALUE_TWD / CAPITAL_TWD
                ),
                "avgCostPoints": (
                    float(row.CostPoints)
                    if hasattr(row, "CostPoints") and not pd.isna(row.CostPoints)
                    else None
                ),
                "avgPoints": float(row.AvgPoints) if not pd.isna(row.AvgPoints) else None,
                "pf": float(row.PF) if not pd.isna(row.PF) else None,
                "mdd": float(row.MaxDrawdownPoints) if not pd.isna(row.MaxDrawdownPoints) else None,
                "pullbackAdv": float(row.PullbackAdvantage) if not pd.isna(row.PullbackAdvantage) else None,
                "positiveYears": int(row.PositiveYears),
                "worstYear": str(row.WorstYear) if not pd.isna(row.WorstYear) else "",
                "worstYearPoints": float(row.WorstYearPoints) if not pd.isna(row.WorstYearPoints) else None,
                "bestYear": str(row.BestYear) if not pd.isna(row.BestYear) else "",
                "bestYearPoints": float(row.BestYearPoints) if not pd.isna(row.BestYearPoints) else None,
                "yearly": yearly,
            }
        )
    return rows


def write_s01_html_report(*, outdir: Path, output_html: Path, root_copy: Path | None = None) -> Path:
    outdir = outdir.resolve()
    summary = _read_top_rows(outdir)
    by_year = _read_year_rows(outdir, set(int(value) for value in summary["RuleID"]))
    run_log = _read_run_log(outdir)
    rows = _build_rows(summary, by_year)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

    cards = [
        ("策略", "S01_ATTACK_C1_PULLBACK"),
        ("合法組合", run_log.get("actual_param_count", "318,240")),
        ("有效樣本", run_log.get("research_samples", "")),
        ("資料期間", f"{run_log.get('sample_date_min', '')} ~ {run_log.get('sample_date_max', '')}"),
        ("成本", "已扣：出場滑點2點、來回手續費36元、期交稅雙邊"),
        ("總報酬率", f"成本後淨損益 ÷ {CAPITAL_TWD:,}"),
        ("輸出資料夾", str(outdir)),
        ("HTML列數", f"{len(rows):,} 組排行榜交集"),
        ("產生時間", generated_at),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>S01 318,240 組條列報表</title>
  <style>
    :root {{
      --line:#dfe8e3; --head:#e4eee9; --text:#1f2a24; --muted:#65726c;
      --red:#ba3f32; --green:#287247; --soft-red:#fff2ef; --soft-green:#f0faf4;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; font-family:"Microsoft JhengHei","Noto Sans TC",Arial,sans-serif;
      color:var(--text); font-size:17px; line-height:1.45; background:#fff;
    }}
    header {{ padding:28px 10px 12px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:34px; letter-spacing:0; }}
    .note {{ max-width:1280px; color:var(--muted); font-weight:600; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(220px,1fr)); gap:10px; padding:16px 10px 28px; }}
    .card {{ border:1px solid var(--line); background:#f8fbf9; border-radius:8px; padding:12px 14px; min-height:78px; }}
    .label {{ color:var(--muted); font-size:14px; margin-bottom:3px; }}
    .value {{ font-size:19px; font-weight:800; word-break:break-word; }}
    main {{ padding:16px 8px 40px; }}
    h2 {{ margin:8px 0 8px; font-size:28px; }}
    .filters {{ display:grid; grid-template-columns:repeat(10,minmax(110px,1fr)); gap:8px; align-items:end; margin:10px 0; }}
    .field label {{ display:block; color:var(--muted); font-size:14px; font-weight:800; margin-bottom:3px; }}
    input,select,button {{ width:100%; height:38px; border:1px solid var(--line); border-radius:6px; background:#fff; font:inherit; font-size:16px; padding:4px 8px; }}
    button {{ cursor:pointer; background:#f8fbf9; color:#245d86; font-weight:800; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; color:var(--muted); font-weight:800; margin:8px 0 10px; }}
    .table-wrap {{ width:100%; max-height:calc(100vh - 250px); overflow:auto; border:1px solid var(--line); box-shadow:0 8px 22px rgba(17,36,28,.08); }}
    table {{ width:100%; min-width:2520px; border-collapse:collapse; table-layout:fixed; font-size:14px; }}
    th,td {{ border-bottom:1px solid var(--line); border-right:1px solid var(--line); padding:6px 7px; vertical-align:middle; }}
    th {{ position:sticky; top:0; z-index:2; background:var(--head); font-size:14px; cursor:pointer; white-space:nowrap; }}
    tbody tr:nth-child(even) {{ background:#f7faf8; }}
    .num {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:800; }}
    .center {{ text-align:center; }}
    .formula {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; }}
    .box {{ border-radius:6px; padding:7px; font-size:12px; line-height:1.35; font-family:Consolas,"Microsoft JhengHei",monospace; white-space:normal; }}
    .long {{ color:var(--red); background:var(--soft-red); border:1px solid #f2c8c1; }}
    .short {{ color:var(--green); background:var(--soft-green); border:1px solid #cce5d5; }}
    .yearcell {{ font-size:12px; line-height:1.25; }}
    .yearcell b {{ font-size:13px; }}
    .pos {{ color:var(--red); font-weight:900; }}
    .neg {{ color:var(--green); font-weight:900; }}
    .muted {{ color:var(--muted); }}
    .source {{ font-size:12px; color:#245d86; font-weight:900; }}
    col.no {{ width:58px; }} col.formula {{ width:330px; }} col.params {{ width:210px; }}
    col.year {{ width:128px; }} col.small {{ width:102px; }} col.medium {{ width:124px; }}
    @media (max-width: 1100px) {{
      .cards {{ grid-template-columns:repeat(2,minmax(180px,1fr)); }}
      .filters {{ grid-template-columns:repeat(2,minmax(150px,1fr)); }}
      h1 {{ font-size:28px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>S01_ATTACK_C1_PULLBACK 318,240 組條列報表</h1>
    <div class="note">
      這份才是最新 S01 報表。訊號只用前一根 O/H/L/C 與本根 Open；本根 High/Low 只驗證是否回踩成交；
      成交價 C1，下一根 Open 出場。已扣出場滑點2點、來回手續費36元、期交稅雙邊；完整資料在 CSV。
    </div>
  </header>
  <section class="cards">{card_html}</section>
  <main>
    <h2>條列總表</h2>
    <div class="filters">
      <div class="field"><label>排行榜來源</label><select id="source"><option value="">全部</option></select></div>
      <div class="field"><label>最小總次數</label><input id="minTrades" type="number" placeholder="不限"></div>
      <div class="field"><label>最小 PF</label><input id="minPf" type="number" step="0.01" placeholder="不限"></div>
      <div class="field"><label>最小平均點</label><input id="minAvg" type="number" step="0.01" placeholder="不限"></div>
      <div class="field"><label>最小總報酬率 %</label><input id="minReturn" type="number" step="0.1" placeholder="不限"></div>
      <div class="field"><label>最小回踩優勢</label><input id="minAdv" type="number" step="0.01" placeholder="不限"></div>
      <div class="field"><label>最小正報酬年</label><input id="minYears" type="number" placeholder="不限"></div>
      <div class="field"><label>搜尋參數</label><input id="query" placeholder="OG=3~8 / BM=18"></div>
      <div class="field"><label>每頁列數</label><select id="pageSize"><option>100</option><option selected>200</option><option>500</option><option>1000</option></select></div>
      <button id="apply">套用</button>
      <button id="clear">清除</button>
    </div>
    <div class="toolbar">
      <span id="count"></span>
      <button id="prev" style="width:90px">上一頁</button>
      <span id="page"></span>
      <button id="next" style="width:90px">下一頁</button>
    </div>
    <div class="table-wrap">
      <table id="tbl">
        <colgroup>
          <col class="no"><col class="small"><col class="formula"><col class="params">
          <col class="year"><col class="year"><col class="year"><col class="year"><col class="year"><col class="year"><col class="year">
          <col class="small"><col class="small"><col class="small"><col class="small"><col class="small"><col class="small"><col class="small"><col class="small"><col class="medium"><col class="medium">
        </colgroup>
        <thead></thead>
        <tbody></tbody>
      </table>
    </div>
  </main>
<script>
const ROWS = {rows_json};
const YEARS = {json.dumps([str(year) for year in YEARS])};
let sortKey = "pf";
let sortDir = -1;
let pageIndex = 0;
let filtered = [];

const columns = [
  ["no","編號"],["source","來源"],["formula","公式簡碼"],["params","參數"],
  ...YEARS.map(y => ["year:"+y, y]),
  ["eligibleTriggers","可交易Trigger"],["fillCount","成交數"],["fillRate","成交率"],
  ["totalTrades","總次數"],["winRate","總勝率"],["totalReturnRate","總報酬率"],["netProfitTwd","淨損益"],
  ["netPoints","淨點數"],["avgPoints","平均點"],["avgCostPoints","平均成本點"],
  ["pf","PF"],["mdd","最大回撤"],["pullbackAdv","回踩優勢"]
];

function fmtNum(v, d=2) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  if (v === Infinity) return "inf";
  if (v === -Infinity) return "-inf";
  const n = Number(v);
  if (Math.abs(n - Math.round(n)) < 1e-9) return Math.round(n).toLocaleString("en-US");
  return n.toLocaleString("en-US", {{maximumFractionDigits:d, minimumFractionDigits:0}});
}}
function fmtPct(v, d=1) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  return (Number(v) * 100).toFixed(d).replace(/\\.0$/, "") + "%";
}}
function cls(v) {{ return Number(v) >= 0 ? "pos" : "neg"; }}
function getValue(row, key) {{
  if (key.startsWith("year:")) {{
    const y = key.slice(5);
    return row.yearly[y] ? row.yearly[y].net : null;
  }}
  return row[key];
}}
function yearHtml(row, y) {{
  const item = row.yearly[y];
  if (!item) return '<span class="muted">無資料</span>';
  return `<div class="yearcell">
    <div>次 <b>${{fmtNum(item.trades,0)}}</b></div>
    <div>勝 <b>${{fmtPct(item.winRate,1)}}</b></div>
    <div>均 <b class="${{cls(item.avg)}}">${{fmtNum(item.avg,2)}}</b></div>
    <div>淨 <b class="${{cls(item.net)}}">${{fmtNum(item.net,0)}}</b></div>
  </div>`;
}}
function cellHtml(row, key) {{
  if (key === "no") return `<td class="center">${{row.no}}</td>`;
  if (key === "source") return `<td><span class="source">${{row.source}}</span><br><span class="muted">#${{row.ruleId}}</span></td>`;
  if (key === "formula") return `<td><div class="formula"><div class="box long">做多<br>${{row.formulaLong}}</div><div class="box short">做空<br>${{row.formulaShort}}</div></div></td>`;
  if (key === "params") return `<td><b>${{row.params}}</b></td>`;
  if (key.startsWith("year:")) return `<td class="num">${{yearHtml(row, key.slice(5))}}</td>`;
  if (["fillRate","winRate"].includes(key)) return `<td class="num">${{fmtPct(row[key],1)}}</td>`;
  if (["totalReturnRate"].includes(key)) return `<td class="num ${{cls(row[key])}}">${{fmtPct(row[key],1)}}</td>`;
  if (["netProfitTwd"].includes(key)) return `<td class="num ${{cls(row[key])}}">${{fmtNum(row[key],0)}}</td>`;
  if (["netPoints","avgPoints","pullbackAdv"].includes(key)) return `<td class="num ${{cls(row[key])}}">${{fmtNum(row[key],2)}}</td>`;
  if (["avgCostPoints"].includes(key)) return `<td class="num">${{fmtNum(row[key],2)}}</td>`;
  if (["pf"].includes(key)) return `<td class="num">${{fmtNum(row[key],2)}}</td>`;
  return `<td class="num">${{fmtNum(row[key],0)}}</td>`;
}}
function applyFilters() {{
  const source = document.getElementById("source").value;
  const minTrades = Number(document.getElementById("minTrades").value || -Infinity);
  const minPf = Number(document.getElementById("minPf").value || -Infinity);
  const minAvg = Number(document.getElementById("minAvg").value || -Infinity);
  const minReturn = Number(document.getElementById("minReturn").value || -Infinity) / 100;
  const minAdv = Number(document.getElementById("minAdv").value || -Infinity);
  const minYears = Number(document.getElementById("minYears").value || -Infinity);
  const q = document.getElementById("query").value.trim().toLowerCase();
  filtered = ROWS.filter(r =>
    (!source || r.source.includes(source)) &&
    r.totalTrades >= minTrades &&
    (r.pf ?? -Infinity) >= minPf &&
    (r.avgPoints ?? -Infinity) >= minAvg &&
    (r.totalReturnRate ?? -Infinity) >= minReturn &&
    (r.pullbackAdv ?? -Infinity) >= minAdv &&
    r.positiveYears >= minYears &&
    (!q || (r.params + " " + r.formulaLong + " " + r.formulaShort + " " + r.ruleId).toLowerCase().includes(q))
  );
  filtered.sort((a,b) => {{
    const av = getValue(a, sortKey);
    const bv = getValue(b, sortKey);
    if (av === bv) return a.no - b.no;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return av > bv ? sortDir : -sortDir;
  }});
  pageIndex = 0;
  render();
}}
function renderHeader() {{
  const thead = document.querySelector("thead");
  thead.innerHTML = "<tr>" + columns.map(([key,label]) => `<th data-key="${{key}}">${{label}}${{key===sortKey ? (sortDir<0 ? " ▼" : " ▲") : ""}}</th>`).join("") + "</tr>";
  thead.querySelectorAll("th").forEach(th => {{
    th.onclick = () => {{
      const key = th.dataset.key;
      if (sortKey === key) sortDir *= -1; else {{ sortKey = key; sortDir = -1; }}
      applyFilters();
    }};
  }});
}}
function render() {{
  renderHeader();
  const pageSize = Number(document.getElementById("pageSize").value);
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  pageIndex = Math.min(pageIndex, pageCount - 1);
  const rows = filtered.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize);
  document.querySelector("tbody").innerHTML = rows.map(r => "<tr>" + columns.map(([key]) => cellHtml(r,key)).join("") + "</tr>").join("");
  document.getElementById("count").textContent = `顯示 ${{filtered.length.toLocaleString("en-US")}} / ${{ROWS.length.toLocaleString("en-US")}} 組`;
  document.getElementById("page").textContent = `第 ${{pageIndex + 1}} / ${{pageCount}} 頁`;
}}
function init() {{
  const sourceSelect = document.getElementById("source");
  [...new Set(ROWS.flatMap(r => r.source.split("、")))].forEach(s => {{
    const opt = document.createElement("option"); opt.value = s; opt.textContent = s; sourceSelect.appendChild(opt);
  }});
  document.getElementById("apply").onclick = applyFilters;
  document.getElementById("clear").onclick = () => {{
    ["source","minTrades","minPf","minAvg","minReturn","minAdv","minYears","query"].forEach(id => document.getElementById(id).value = "");
    applyFilters();
  }};
  document.getElementById("prev").onclick = () => {{ pageIndex = Math.max(0, pageIndex - 1); render(); }};
  document.getElementById("next").onclick = () => {{ pageIndex += 1; render(); }};
  document.getElementById("pageSize").onchange = applyFilters;
  applyFilters();
}}
init();
</script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    if root_copy is not None:
        root_copy.write_text(html_text, encoding="utf-8")
    return output_html
