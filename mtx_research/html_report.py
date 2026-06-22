from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = list(range(2020, 2027))
TOP_FILES = [
    ("top_r2a_robust.csv", "穩健候選"),
    ("top_r2a_net.csv", "淨損益排名"),
    ("top_r2a_pf.csv", "PF排名"),
    ("top_r2a_avg.csv", "平均點排名"),
    ("top_r2a_pullback_advantage.csv", "回踩優勢排名"),
    ("top_r2a_candidates_for_r2b.csv", "R2B候選"),
]


def _fmt_num(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    if value == np.inf:
        return "∞"
    if value == -np.inf:
        return "-∞"
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def _fmt_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def _read_top_rows(r2a_dir: Path, limit: int) -> pd.DataFrame:
    parts = []
    for order, (filename, label) in enumerate(TOP_FILES):
        path = r2a_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path).copy()
        if df.empty:
            continue
        df["SourceLabel"] = label
        df["SourceOrder"] = order
        df["SourceRank"] = np.arange(1, len(df) + 1)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"找不到 top_r2a_*.csv：{r2a_dir}")
    combined = pd.concat(parts, ignore_index=True).copy()
    combined["SourceList"] = combined.groupby("StrategyID")["SourceLabel"].transform(lambda values: " / ".join(dict.fromkeys(values)))
    combined["BestSourceOrder"] = combined.groupby("StrategyID")["SourceOrder"].transform("min")
    combined["BestSourceRank"] = combined.groupby("StrategyID")["SourceRank"].transform("min")
    combined = combined.sort_values(["BestSourceOrder", "BestSourceRank", "RuleID"], kind="mergesort")
    combined = combined.drop_duplicates("StrategyID", keep="first").head(limit).reset_index(drop=True)
    return combined


def _read_by_year(r2a_dir: Path, strategy_ids: set[str]) -> pd.DataFrame:
    path = r2a_dir / "by_year_r2a.csv"
    if not path.exists() or not strategy_ids:
        return pd.DataFrame()
    parts = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        part = chunk[chunk["StrategyID"].isin(strategy_ids)]
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _family_formula(row: pd.Series) -> str:
    family = str(row.get("Family", ""))
    bits = [family]
    for col, label in [
        ("BodyMin", "Body>="),
        ("RangeMax", "Range<="),
        ("BodyPctMin", "Body%>="),
        ("BodyPctFloor", "Body%>="),
        ("ClosePosMin", "ClosePos"),
        ("EffOppTailMax", "OppTail%<="),
        ("MaruBodyPct", "Maru%>="),
        ("TailMax", "Tail%<="),
        ("MainTailMin", "MainTail%>="),
        ("CenterOffset", "Center>="),
        ("RangeMinLarge", "Range>="),
        ("RangeMaxLarge", "Range<="),
        ("BodyMinLarge", "Body>="),
    ]:
        value = row.get(col)
        if pd.notna(value):
            bits.append(f"{label}{_fmt_num(value, 0)}")
    return "；".join(bits)


def _entry_formula(row: pd.Series) -> str:
    og_min = _fmt_num(row.get("OpenGapMin"), 0)
    og_max = _fmt_num(row.get("OpenGapMax"), 0)
    p = _fmt_num(row.get("Penetrate"), 0)
    mode = html.escape(str(row.get("ExecMode", "")))
    return (
        f"做多：O0 順勢表態，C1+{og_min} <= O0 <= C1+{og_max}，"
        f"回踩 L0 <= Anchor-{p}<br>"
        f"做空：鏡像，C1-{og_max} <= O0 <= C1-{og_min}，"
        f"回抽 H0 >= Anchor+{p}<br>"
        f"ExecMode：{mode}"
    )


def _year_map(by_year: pd.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    result: dict[tuple[str, int], dict[str, object]] = {}
    if by_year.empty:
        return result
    for row in by_year.itertuples(index=False):
        result[(str(row.StrategyID), int(row.Year))] = {
            "trades": int(row.Trades),
            "win": None if pd.isna(row.WinRate) else float(row.WinRate),
            "net": float(row.NetPoints),
            "ret": None if pd.isna(row.TotalReturnRate) else float(row.TotalReturnRate),
            "mdd": None if pd.isna(row.MaxDrawdownNetPoints) else float(row.MaxDrawdownNetPoints),
        }
    return result


def _table_rows(top: pd.DataFrame, by_year: pd.DataFrame) -> str:
    years = _year_map(by_year)
    rows = []
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        s = pd.Series(row._asdict())
        year_cells = []
        for year in YEARS:
            value = years.get((str(s.StrategyID), year))
            if not value or value["trades"] == 0:
                year_cells.append('<td class="year-empty">無</td>')
                continue
            cls = "pos" if value["net"] > 0 else "neg" if value["net"] < 0 else ""
            year_cells.append(
                f'<td class="year-cell {cls}">'
                f'<b>{_fmt_num(value["trades"], 0)}</b><br>'
                f'勝 {_fmt_pct(value["win"])}<br>'
                f'淨 {_fmt_num(value["net"], 1)}<br>'
                f'報 {_fmt_pct(value["ret"])}'
                f"</td>"
            )
        total_ret = float(s.TotalReturnRate) if pd.notna(s.TotalReturnRate) else np.nan
        net_profit = float(s.NetProfitTWD) if pd.notna(s.NetProfitTWD) else np.nan
        net_cls = "pos" if net_profit > 0 else "neg" if net_profit < 0 else ""
        pf_value = 0 if pd.isna(s.PFNet) or s.PFNet == np.inf else float(s.PFNet)
        search_text = html.escape(f"{s.Family} {s.ExecMode} {s.StrategyID}")
        source_list = html.escape(str(s.get("SourceList", "")))
        strategy_id = html.escape(str(s.StrategyID))
        formula_text = html.escape(_family_formula(s))
        entry_text = _entry_formula(s)
        year_html = "".join(year_cells)
        rows.append(
            f'<tr data-trades="{int(s.TotalTrades)}" '
            f'data-return="{0 if pd.isna(total_ret) else total_ret}" '
            f'data-pf="{pf_value}" data-search="{search_text}">'
            f'<td class="sticky c0">{idx}</td>'
            f'<td class="sticky c1"><b>{strategy_id}</b><br>{source_list}</td>'
            f'<td class="sticky c2 formula">{formula_text}<hr>{entry_text}</td>'
            f'<td><b>OG={_fmt_num(s.OpenGapMin,0)}~{_fmt_num(s.OpenGapMax,0)}</b><br>P={_fmt_num(s.Penetrate,0)}</td>'
            f'{year_html}'
            f'<td>{_fmt_num(s.TotalTrades,0)}</td>'
            f'<td>{_fmt_pct(s.WinRate)}</td>'
            f'<td class="{net_cls}">{_fmt_num(s.NetPoints,1)}</td>'
            f'<td class="{net_cls}">{_fmt_num(s.NetProfitTWD,0)}</td>'
            f'<td class="{net_cls}">{_fmt_pct(s.TotalReturnRate,2)}</td>'
            f'<td>{_fmt_num(s.PFNet,2)}</td>'
            f'<td>{_fmt_num(s.AvgNetPoints,2)}</td>'
            f'<td>{_fmt_num(s.MaxDrawdownNetPoints,1)}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def write_html_report(r2a_dir: Path, output_path: Path, *, limit: int = 1000) -> Path:
    top = _read_top_rows(r2a_dir, limit)
    by_year = _read_by_year(r2a_dir, set(top["StrategyID"].astype(str)))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = _table_rows(top, by_year)
    data_info = {
        "顯示組數": len(top),
        "來源資料夾": str(r2a_dir),
        "產生時間": generated,
    }
    cards = "\n".join(
        f'<div class="card"><span>{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>'
        for k, v in data_info.items()
    )
    header_years = "".join(f"<th>{year}</th>" for year in YEARS)
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>MTX R2A 條列報表</title>
<style>
body{{margin:0;font-family:"Microsoft JhengHei",Arial,sans-serif;color:#1d2823;background:#f7faf8;font-size:18px;}}
header{{padding:24px 12px 12px;background:#fff;border-bottom:1px solid #dbe5df;}}
h1{{font-size:34px;margin:0 0 10px;}}
.note{{color:#62706a;line-height:1.7;max-width:1300px;}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:12px;margin-top:18px;}}
.card{{background:#f4f8f6;border:1px solid #dbe5df;border-radius:8px;padding:14px 16px;}}
.card span{{display:block;color:#6b7772;font-size:15px;margin-bottom:6px;}}
.card b{{font-size:18px;word-break:break-all;}}
main{{padding:20px 8px 40px;}}
.filters{{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px;margin:0 0 12px;align-items:end;}}
label{{display:block;color:#5f6d67;font-weight:700;font-size:14px;margin-bottom:5px;}}
input{{width:100%;box-sizing:border-box;font-size:17px;padding:9px 10px;border:1px solid #d8e3dd;border-radius:6px;background:#fff;}}
button{{font-size:17px;padding:10px;border:1px solid #cbd9d2;border-radius:6px;background:#eef5f1;color:#29577a;font-weight:700;cursor:pointer;}}
.count{{font-weight:700;color:#53635d;margin:8px 0 12px;}}
.table-wrap{{overflow:auto;border:1px solid #dbe5df;background:#fff;box-shadow:0 8px 22px rgba(30,50,40,.08);max-height:78vh;}}
table{{border-collapse:separate;border-spacing:0;min-width:2500px;width:max-content;font-size:15px;}}
th,td{{border-right:1px solid #e1e8e4;border-bottom:1px solid #e6ece9;padding:7px 8px;text-align:right;vertical-align:middle;white-space:nowrap;}}
th{{position:sticky;top:0;background:#dfece6;z-index:5;color:#22312b;font-weight:800;cursor:pointer;}}
td{{background:#fff;}}
tbody tr:nth-child(even) td{{background:#f6faf8;}}
.sticky{{position:sticky;z-index:4;text-align:left;}}
.c0{{left:0;min-width:44px;text-align:center;font-weight:800;}}
.c1{{left:61px;min-width:190px;}}
.c2{{left:268px;min-width:340px;max-width:340px;white-space:normal;line-height:1.45;}}
th.sticky{{z-index:7;}}
.formula{{font-family:Consolas,"Microsoft JhengHei",monospace;font-size:13px;}}
.year-cell{{font-size:14px;line-height:1.35;min-width:92px;}}
.year-empty{{color:#97a39d;text-align:center;min-width:92px;}}
.pos{{color:#bd3e31;font-weight:800;}}
.neg{{color:#2f8b58;font-weight:800;}}
hr{{border:none;border-top:1px solid #e4ddd8;margin:6px 0;}}
@media(max-width:900px){{body{{font-size:16px;}}.cards,.filters{{grid-template-columns:1fr;}}h1{{font-size:28px;}}}}
</style>
</head>
<body>
<header>
<h1>MTX R2A 條列報表</h1>
<div class="note">本表顯示 R2A top 檔彙整結果，績效已扣出場 2 點滑點、來回手續費 36 元與期交稅；總報酬率以本金 250,000 元計算。點表頭可排序。</div>
<div class="cards">{cards}</div>
</header>
<main>
<div class="filters">
<div><label>搜尋</label><input id="q" placeholder="Family / ExecMode / StrategyID"></div>
<div><label>最小總次數</label><input id="minTrades" type="number" placeholder="不限"></div>
<div><label>最小總報酬率 %</label><input id="minReturn" type="number" step="0.1" placeholder="不限"></div>
<div><label>最小 PF</label><input id="minPf" type="number" step="0.01" placeholder="不限"></div>
<button onclick="clearFilters()">清除</button>
</div>
<div id="count" class="count"></div>
<div class="table-wrap">
<table id="report">
<thead><tr>
<th class="sticky c0">編號</th><th class="sticky c1">策略ID / 來源</th><th class="sticky c2">公式簡碼</th><th>參數</th>
{header_years}
<th>總次數</th><th>總勝率</th><th>淨點數</th><th>淨損益</th><th>總報酬率</th><th>PF</th><th>平均點</th><th>MDD</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
</main>
<script>
const table = document.getElementById('report');
const rows = Array.from(table.tBodies[0].rows);
let sortState = {{}};
function applyFilters(){{
  const q = document.getElementById('q').value.trim().toLowerCase();
  const minTrades = parseFloat(document.getElementById('minTrades').value || '-Infinity');
  const minReturn = parseFloat(document.getElementById('minReturn').value || '-Infinity') / 100;
  const minPf = parseFloat(document.getElementById('minPf').value || '-Infinity');
  let shown = 0;
  rows.forEach(row => {{
    const ok = (!q || row.dataset.search.toLowerCase().includes(q))
      && Number(row.dataset.trades) >= minTrades
      && Number(row.dataset.return) >= minReturn
      && Number(row.dataset.pf) >= minPf;
    row.style.display = ok ? '' : 'none';
    if (ok) shown++;
  }});
  document.getElementById('count').textContent = `顯示 ${{shown.toLocaleString()}} / ${{rows.length.toLocaleString()}} 組`;
}}
function clearFilters(){{
  ['q','minTrades','minReturn','minPf'].forEach(id => document.getElementById(id).value='');
  applyFilters();
}}
function cellNumber(row, idx){{
  const text = row.cells[idx].innerText.replace(/[%,$,]/g,'').replace('∞','Infinity').trim();
  const n = parseFloat(text);
  return Number.isNaN(n) ? text : n;
}}
Array.from(table.tHead.rows[0].cells).forEach((th, idx) => {{
  th.addEventListener('click', () => {{
    const desc = !sortState[idx];
    sortState = {{[idx]: desc}};
    const sorted = rows.slice().sort((a,b) => {{
      const av = cellNumber(a, idx), bv = cellNumber(b, idx);
      if (typeof av === 'number' && typeof bv === 'number') return desc ? bv-av : av-bv;
      return desc ? String(bv).localeCompare(String(av),'zh-Hant') : String(av).localeCompare(String(bv),'zh-Hant');
    }});
    sorted.forEach(row => table.tBodies[0].appendChild(row));
    applyFilters();
  }});
}});
['q','minTrades','minReturn','minPf'].forEach(id => document.getElementById(id).addEventListener('input', applyFilters));
applyFilters();
</script>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
