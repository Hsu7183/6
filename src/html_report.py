from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


YEARS = list(range(2020, 2027))

DISPLAY_COLUMNS = [
    "rule_id",
    "open_gap_min",
    "open_gap_max",
    "pullback_depth",
    "body_min",
]

for _year in YEARS:
    DISPLAY_COLUMNS.extend(
        [
            f"combined_{_year}_trade_count",
            f"combined_{_year}_win_rate",
            f"combined_{_year}_return_rate",
            f"combined_{_year}_max_drawdown_twd",
        ]
    )

DISPLAY_COLUMNS.extend(
    [
    "combined_trade_count",
    "combined_win_rate",
    "combined_avg_points",
    "combined_gross_points",
    "combined_net_profit_twd",
    "combined_return_rate",
    "combined_profit_factor",
    "combined_max_drawdown_points",
    "combined_max_drawdown_twd",
    "combined_total_fee_twd",
    "combined_total_tax_twd",
    "combined_total_slippage_twd",
    "combined_robust_score",
    "long_trade_count",
    "long_win_rate",
    "long_avg_points",
    "long_net_profit_twd",
    "long_profit_factor",
    "short_trade_count",
    "short_win_rate",
    "short_avg_points",
    "short_net_profit_twd",
    "short_profit_factor",
    ]
)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.8g}"
    return str(value)


def _build_data_text(df: pd.DataFrame) -> str:
    lines = []
    for row in df.itertuples(index=False, name=None):
        lines.append("|".join(_format_value(value) for value in row))
    return "\n".join(lines)


def _read_report_text(report_path: Path) -> str:
    if not report_path.exists():
        return ""
    return report_path.read_text(encoding="utf-8", errors="ignore")


def write_html_report(
    *,
    main_csv: Path,
    report_txt: Path,
    output_html: Path,
) -> Path:
    df = pd.read_csv(main_csv, encoding="utf-8-sig", usecols=DISPLAY_COLUMNS)
    df = df[DISPLAY_COLUMNS]
    row_count = len(df)
    data_text = _build_data_text(df)
    report_text = _read_report_text(report_txt)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROD 328,000 組條列報表</title>
  <style>
    :root {{
      --line: #dfe8e3;
      --head: #e6f0eb;
      --text: #1f2a24;
      --muted: #65726c;
      --red: #bd4636;
      --green: #287247;
      --soft-red: #fff3f0;
      --soft-green: #f0faf4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "Microsoft JhengHei", "Noto Sans TC", Arial, sans-serif;
      font-size: 18px;
      line-height: 1.45;
      background: #fff;
    }}
    header {{
      padding: 28px 10px 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 34px;
      letter-spacing: 0;
    }}
    .note {{ color: var(--muted); max-width: 1180px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(8, minmax(170px, 1fr));
      gap: 10px;
      padding: 14px 10px 26px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: #f8fbf9;
      border-radius: 8px;
      padding: 12px 14px;
      min-height: 78px;
    }}
    .label {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 4px;
    }}
    .value {{
      font-weight: 800;
      font-size: 20px;
      word-break: break-word;
    }}
    main {{ padding: 18px 8px 40px; }}
    h2 {{ margin: 10px 0 8px; font-size: 28px; }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(12, minmax(104px, 1fr));
      gap: 8px;
      align-items: end;
      margin: 12px 0 8px;
    }}
    .field label {{
      display: block;
      font-size: 14px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 3px;
    }}
    input, select, button {{
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 16px;
      padding: 4px 8px;
    }}
    button {{
      cursor: pointer;
      color: #245d86;
      background: #f8fbf9;
      font-weight: 800;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 14px;
      margin: 8px 0 10px;
      color: var(--muted);
      font-weight: 700;
    }}
    .pager {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .pager button {{ width: 88px; }}
    .pager select {{ width: 92px; }}
    .table-wrap {{
      width: 100%;
      overflow: auto;
      border: 1px solid var(--line);
      box-shadow: 0 8px 22px rgba(17, 36, 28, 0.08);
      max-height: calc(100vh - 250px);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 15px;
      background: #fff;
    }}
    th, td {{
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: middle;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--head);
      font-size: 14px;
      cursor: pointer;
      user-select: none;
    }}
    tbody tr:nth-child(even) {{ background: #f8fbf9; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }}
    td.center {{ text-align: center; }}
    .formula-cell {{ white-space: normal; }}
    .formula-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 5px;
      min-width: 270px;
    }}
    .formula-box {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 7px;
      font-family: Consolas, "Microsoft JhengHei", monospace;
      font-size: 12px;
      line-height: 1.35;
    }}
    .formula-box.long {{ background: var(--soft-red); border-color: #f0c8bf; }}
    .formula-box.short {{ background: var(--soft-green); border-color: #c9e5d2; }}
    .tag {{ display: block; font-weight: 900; margin-bottom: 2px; font-family: "Microsoft JhengHei", sans-serif; }}
    .pos {{ color: var(--red); }}
    .neg {{ color: var(--green); }}
    .muted {{ color: var(--muted); }}
    .report-text {{
      margin-top: 18px;
      padding: 12px;
      border: 1px solid var(--line);
      background: #f8fbf9;
      border-radius: 8px;
      white-space: pre-wrap;
      font-family: Consolas, monospace;
      font-size: 14px;
      color: #34413b;
    }}
    .w-id {{ width: 76px; }}
    .w-formula {{ width: 330px; }}
    .w-param {{ width: 116px; }}
    .w-year {{ width: 126px; }}
    .w-small {{ width: 90px; }}
    .w-med {{ width: 112px; }}
    @media (max-width: 1200px) {{
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .filters {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ROD C1 第一層前K實體 328,000 組條列報表</h1>
    <div class="note">
      基準價固定 C1。每一列是一組參數，同一組同步測做多、做空鏡像、以及多空合併績效。
      勝率、獲利因子、最大回撤、淨點數、淨損益與總報酬率都已扣除：進場滑點 0 點、出場滑點 2 點、手續費單邊 18 元、期交稅每邊 0.00002，且期交稅單邊四捨五入到元。點表頭可排序；第一次點降序，再點一次升序。
    </div>
  </header>
  <section class="cards">
    <div class="card"><div class="label">主檔列數</div><div class="value">{row_count:,} 組</div></div>
    <div class="card"><div class="label">開盤跳空下限</div><div class="value">2 ~ 392（每 10）</div></div>
    <div class="card"><div class="label">開盤跳空上限</div><div class="value">3 ~ 393（每 10）</div></div>
    <div class="card"><div class="label">有效跳空區間</div><div class="value">820 組</div></div>
    <div class="card"><div class="label">回踩/回抽深度</div><div class="value">1 ~ 10（每 1）</div></div>
    <div class="card"><div class="label">前 K 實體下限</div><div class="value">1 ~ 391（每 10）</div></div>
    <div class="card"><div class="label">成本設定</div><div class="value">本金25萬；小台50元/點</div></div>
    <div class="card"><div class="label">產生時間</div><div class="value">{generated_at}</div></div>
  </section>
  <main>
    <h2>條列總表</h2>
    <div class="filters">
      <div class="field"><label>最小跳空下限</label><input id="minOgL" type="number" placeholder="不限"></div>
      <div class="field"><label>最大跳空上限</label><input id="maxOgH" type="number" placeholder="不限"></div>
      <div class="field"><label>最小回踩深度</label><input id="minP" type="number" placeholder="不限"></div>
      <div class="field"><label>最小前K實體</label><input id="minBody" type="number" placeholder="不限"></div>
      <div class="field"><label>最小總次數</label><input id="minTrades" type="number" placeholder="不限"></div>
      <div class="field"><label>最小總勝率 %</label><input id="minWin" type="number" step="0.1" placeholder="不限"></div>
      <div class="field"><label>最小淨均點</label><input id="minAvg" type="number" step="0.1" placeholder="不限"></div>
      <div class="field"><label>最小總報酬率 %</label><input id="minReturn" type="number" step="0.1" placeholder="不限"></div>
      <div class="field"><label>最小獲利因子</label><input id="minPf" type="number" step="0.01" placeholder="不限"></div>
      <div class="field"><label>最大回撤金額</label><input id="maxMdd" type="number" placeholder="不限"></div>
      <div class="field"><label>每頁列數</label><select id="pageSize"><option>100</option><option selected>200</option><option>500</option><option>1000</option></select></div>
      <div class="field"><label>&nbsp;</label><button id="clearBtn">清除</button></div>
      <div class="field"><label>&nbsp;</label><button id="applyBtn">套用</button></div>
    </div>
    <div class="toolbar">
      <span id="countText">載入中...</span>
      <span id="sortText"></span>
      <span class="pager">
        <button id="prevBtn">上一頁</button>
        <span id="pageText"></span>
        <button id="nextBtn">下一頁</button>
      </span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="w-id" data-sort="rule_id">編號</th>
            <th class="w-formula">公式簡碼</th>
            <th class="w-param" data-sort="open_gap_min">參數</th>
            <th class="w-year" data-sort="combined_2020_return_rate">2020</th>
            <th class="w-year" data-sort="combined_2021_return_rate">2021</th>
            <th class="w-year" data-sort="combined_2022_return_rate">2022</th>
            <th class="w-year" data-sort="combined_2023_return_rate">2023</th>
            <th class="w-year" data-sort="combined_2024_return_rate">2024</th>
            <th class="w-year" data-sort="combined_2025_return_rate">2025</th>
            <th class="w-year" data-sort="combined_2026_return_rate">2026</th>
            <th class="w-med" data-sort="combined_trade_count">總次數</th>
            <th class="w-small" data-sort="combined_win_rate">總勝率</th>
            <th class="w-small" data-sort="combined_avg_points">淨均點</th>
            <th class="w-med" data-sort="combined_gross_points">淨點數</th>
            <th class="w-med" data-sort="combined_net_profit_twd">淨損益</th>
            <th class="w-med" data-sort="combined_return_rate">總報酬率</th>
            <th class="w-med" data-sort="combined_max_drawdown_twd">最大回撤</th>
            <th class="w-small" data-sort="combined_profit_factor">獲利因子</th>
            <th class="w-med" data-sort="combined_total_fee_twd">手續費</th>
            <th class="w-med" data-sort="combined_total_tax_twd">期交稅</th>
            <th class="w-med" data-sort="combined_total_slippage_twd">滑點成本</th>
            <th class="w-med" data-sort="combined_robust_score">穩健分數</th>
            <th class="w-med" data-sort="long_trade_count">多次數</th>
            <th class="w-small" data-sort="long_win_rate">多勝率</th>
            <th class="w-small" data-sort="long_avg_points">多淨均點</th>
            <th class="w-med" data-sort="short_trade_count">空次數</th>
            <th class="w-small" data-sort="short_win_rate">空勝率</th>
            <th class="w-small" data-sort="short_avg_points">空淨均點</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
    <pre class="report-text">{report_text}</pre>
  </main>
  <script id="dataset" type="text/plain">{data_text}</script>
  <script>
    const columns = {DISPLAY_COLUMNS!r};
    const colIndex = Object.fromEntries(columns.map((name, index) => [name, index]));
    const columnLabels = {{
      rule_id: "編號",
      open_gap_min: "開盤跳空下限",
      open_gap_max: "開盤跳空上限",
      pullback_depth: "回踩/回抽深度",
      body_min: "前K實體下限",
      combined_2020_return_rate: "2020報酬率",
      combined_2021_return_rate: "2021報酬率",
      combined_2022_return_rate: "2022報酬率",
      combined_2023_return_rate: "2023報酬率",
      combined_2024_return_rate: "2024報酬率",
      combined_2025_return_rate: "2025報酬率",
      combined_2026_return_rate: "2026報酬率",
      combined_trade_count: "總次數",
      combined_win_rate: "總勝率",
      combined_avg_points: "淨均點",
      combined_gross_points: "淨點數",
      combined_net_profit_twd: "淨損益",
      combined_return_rate: "總報酬率",
      combined_profit_factor: "獲利因子",
      combined_max_drawdown_twd: "最大回撤",
      combined_total_fee_twd: "手續費",
      combined_total_tax_twd: "期交稅",
      combined_total_slippage_twd: "滑點成本",
      combined_robust_score: "穩健分數",
      long_trade_count: "多次數",
      long_win_rate: "多勝率",
      long_avg_points: "多淨均點",
      short_trade_count: "空次數",
      short_win_rate: "空勝率",
      short_avg_points: "空淨均點",
    }};
    const raw = document.getElementById("dataset").textContent.trim();
    const rows = raw ? raw.split("\\n").map((line) => line.split("|").map(parseValue)) : [];
    let filtered = rows.slice();
    let sortKey = null;
    let sortDir = "desc";
    let page = 1;

    const tbody = document.getElementById("tbody");
    const countText = document.getElementById("countText");
    const pageText = document.getElementById("pageText");
    const sortText = document.getElementById("sortText");
    const pageSizeEl = document.getElementById("pageSize");

    function parseValue(value) {{
      if (value === "" || value.toLowerCase() === "nan") return NaN;
      if (value === "inf") return Infinity;
      if (value === "-inf") return -Infinity;
      return Number(value);
    }}

    function get(row, key) {{
      return row[colIndex[key]];
    }}

    function inputNumber(id) {{
      const text = document.getElementById(id).value.trim();
      return text === "" ? null : Number(text);
    }}

    function passes(row, f) {{
      if (f.minOgL !== null && get(row, "open_gap_min") < f.minOgL) return false;
      if (f.maxOgH !== null && get(row, "open_gap_max") > f.maxOgH) return false;
      if (f.minP !== null && get(row, "pullback_depth") < f.minP) return false;
      if (f.minBody !== null && get(row, "body_min") < f.minBody) return false;
      if (f.minTrades !== null && get(row, "combined_trade_count") < f.minTrades) return false;
      if (f.minWin !== null && get(row, "combined_win_rate") * 100 < f.minWin) return false;
      if (f.minAvg !== null && get(row, "combined_avg_points") < f.minAvg) return false;
      if (f.minReturn !== null && get(row, "combined_return_rate") * 100 < f.minReturn) return false;
      if (f.minPf !== null && get(row, "combined_profit_factor") < f.minPf) return false;
      if (f.maxMdd !== null && get(row, "combined_max_drawdown_twd") > f.maxMdd) return false;
      return true;
    }}

    function applyFilters() {{
      const f = {{
        minOgL: inputNumber("minOgL"),
        maxOgH: inputNumber("maxOgH"),
        minP: inputNumber("minP"),
        minBody: inputNumber("minBody"),
        minTrades: inputNumber("minTrades"),
        minWin: inputNumber("minWin"),
        minAvg: inputNumber("minAvg"),
        minReturn: inputNumber("minReturn"),
        minPf: inputNumber("minPf"),
        maxMdd: inputNumber("maxMdd"),
      }};
      filtered = rows.filter((row) => passes(row, f));
      page = 1;
      applySort(false);
      render();
    }}

    function applySort(toggle) {{
      if (!sortKey) return;
      const idx = colIndex[sortKey];
      filtered.sort((a, b) => {{
        const av = a[idx];
        const bv = b[idx];
        const aBad = Number.isNaN(av);
        const bBad = Number.isNaN(bv);
        if (aBad && bBad) return 0;
        if (aBad) return 1;
        if (bBad) return -1;
        return sortDir === "desc" ? bv - av : av - bv;
      }});
    }}

    function sortBy(key) {{
      if (sortKey === key) {{
        sortDir = sortDir === "desc" ? "asc" : "desc";
      }} else {{
        sortKey = key;
        sortDir = "desc";
      }}
      page = 1;
      applySort(true);
      render();
    }}

    function fmtInt(value) {{
      if (!Number.isFinite(value)) return "-";
      return Math.round(value).toLocaleString("zh-TW");
    }}

    function fmtNum(value, digits = 1) {{
      if (Number.isNaN(value)) return "-";
      if (value === Infinity) return "∞";
      if (value === -Infinity) return "-∞";
      const isInt = Math.abs(value - Math.round(value)) < 1e-9;
      return value.toLocaleString("zh-TW", {{
        minimumFractionDigits: 0,
        maximumFractionDigits: isInt ? 0 : digits,
      }});
    }}

    function fmtPct(value) {{
      if (!Number.isFinite(value)) return "-";
      return (value * 100).toLocaleString("zh-TW", {{ maximumFractionDigits: 1 }}) + "%";
    }}

    function valueClass(value) {{
      if (!Number.isFinite(value) || value === 0) return "";
      return value > 0 ? "pos" : "neg";
    }}

    function formulaHtml(row) {{
      const ogL = get(row, "open_gap_min");
      const ogH = get(row, "open_gap_max");
      const p = get(row, "pullback_depth");
      const b = get(row, "body_min");
      return `
        <div class="formula-grid">
          <div class="formula-box long"><span class="tag">做多</span>O0&gt;=C1+${{ogL}}<br>L0&lt;=C1-${{p}}<br>O0&lt;=C1+${{ogH}}<br>C1-O1&gt;=${{b}}</div>
          <div class="formula-box short"><span class="tag">做空</span>O0&lt;=C1-${{ogL}}<br>H0&gt;=C1+${{p}}<br>O0&gt;=C1-${{ogH}}<br>O1-C1&gt;=${{b}}</div>
        </div>`;
    }}

    function yearCellHtml(row, year) {{
      const count = get(row, `combined_${{year}}_trade_count`);
      if (!Number.isFinite(count) || count === 0) {{
        return '<span class="muted">無資料</span>';
      }}
      const win = get(row, `combined_${{year}}_win_rate`);
      const ret = get(row, `combined_${{year}}_return_rate`);
      const mdd = get(row, `combined_${{year}}_max_drawdown_twd`);
      return `<div>次 <b>${{fmtInt(count)}}</b></div>` +
        `<div>勝 <b>${{fmtPct(win)}}</b></div>` +
        `<div>MDD <b>${{fmtNum(mdd, 0)}}</b></div>` +
        `<div>報 <b class="${{valueClass(ret)}}">${{fmtPct(ret)}}</b></div>`;
    }}

    function rowHtml(row) {{
      const avg = get(row, "combined_avg_points");
      const gross = get(row, "combined_gross_points");
      const robust = get(row, "combined_robust_score");
      const longAvg = get(row, "long_avg_points");
      const shortAvg = get(row, "short_avg_points");
      const netProfit = get(row, "combined_net_profit_twd");
      const ret = get(row, "combined_return_rate");
      return `<tr>
        <td class="center">${{fmtInt(get(row, "rule_id"))}}</td>
        <td class="formula-cell">${{formulaHtml(row)}}</td>
        <td><b>OG=${{fmtInt(get(row, "open_gap_min"))}}~${{fmtInt(get(row, "open_gap_max"))}}</b><br><b>P=${{fmtInt(get(row, "pullback_depth"))}}</b><br><b>B=${{fmtInt(get(row, "body_min"))}}</b></td>
        <td class="num">${{yearCellHtml(row, 2020)}}</td>
        <td class="num">${{yearCellHtml(row, 2021)}}</td>
        <td class="num">${{yearCellHtml(row, 2022)}}</td>
        <td class="num">${{yearCellHtml(row, 2023)}}</td>
        <td class="num">${{yearCellHtml(row, 2024)}}</td>
        <td class="num">${{yearCellHtml(row, 2025)}}</td>
        <td class="num">${{yearCellHtml(row, 2026)}}</td>
        <td class="num">${{fmtInt(get(row, "combined_trade_count"))}}</td>
        <td class="num">${{fmtPct(get(row, "combined_win_rate"))}}</td>
        <td class="num ${{valueClass(avg)}}">${{fmtNum(avg, 2)}}</td>
        <td class="num ${{valueClass(gross)}}">${{fmtNum(gross, 0)}}</td>
        <td class="num ${{valueClass(netProfit)}}">${{fmtNum(netProfit, 0)}}</td>
        <td class="num ${{valueClass(ret)}}">${{fmtPct(ret)}}</td>
        <td class="num">${{fmtNum(get(row, "combined_max_drawdown_twd"), 0)}}</td>
        <td class="num">${{fmtNum(get(row, "combined_profit_factor"), 2)}}</td>
        <td class="num">${{fmtNum(get(row, "combined_total_fee_twd"), 0)}}</td>
        <td class="num">${{fmtNum(get(row, "combined_total_tax_twd"), 0)}}</td>
        <td class="num">${{fmtNum(get(row, "combined_total_slippage_twd"), 0)}}</td>
        <td class="num ${{valueClass(robust)}}">${{fmtNum(robust, 2)}}</td>
        <td class="num">${{fmtInt(get(row, "long_trade_count"))}}</td>
        <td class="num">${{fmtPct(get(row, "long_win_rate"))}}</td>
        <td class="num ${{valueClass(longAvg)}}">${{fmtNum(longAvg, 2)}}</td>
        <td class="num">${{fmtInt(get(row, "short_trade_count"))}}</td>
        <td class="num">${{fmtPct(get(row, "short_win_rate"))}}</td>
        <td class="num ${{valueClass(shortAvg)}}">${{fmtNum(shortAvg, 2)}}</td>
      </tr>`;
    }}

    function render() {{
      const pageSize = Number(pageSizeEl.value);
      const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
      page = Math.max(1, Math.min(page, pages));
      const start = (page - 1) * pageSize;
      const visible = filtered.slice(start, start + pageSize);
      tbody.innerHTML = visible.map(rowHtml).join("");
      countText.textContent = `篩選 ${{filtered.length.toLocaleString("zh-TW")}} / 全部 ${{rows.length.toLocaleString("zh-TW")}} 組`;
      pageText.textContent = `第 ${{page.toLocaleString("zh-TW")}} / ${{pages.toLocaleString("zh-TW")}} 頁`;
      const sortLabel = columnLabels[sortKey] || sortKey;
      sortText.textContent = sortKey ? `排序：${{sortLabel}} ${{sortDir === "desc" ? "降序" : "升序"}}` : "排序：編號原始順序";
    }}

    document.querySelectorAll("th[data-sort]").forEach((th) => {{
      th.addEventListener("click", () => sortBy(th.dataset.sort));
    }});
    document.getElementById("applyBtn").addEventListener("click", applyFilters);
    document.getElementById("clearBtn").addEventListener("click", () => {{
      document.querySelectorAll(".filters input").forEach((input) => input.value = "");
      filtered = rows.slice();
      sortKey = null;
      sortDir = "desc";
      page = 1;
      render();
    }});
    document.getElementById("prevBtn").addEventListener("click", () => {{ page -= 1; render(); }});
    document.getElementById("nextBtn").addEventListener("click", () => {{ page += 1; render(); }});
    pageSizeEl.addEventListener("change", () => {{ page = 1; render(); }});
    render();
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html
