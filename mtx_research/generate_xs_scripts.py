from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.anchor_body_bins import BIN_MODE_ANCHOR_RATIO, BIN_MODE_POINTS


DEFAULT_SUMMARY = (
    Path("report_outputs")
    / "four_layer_anchor_ratio_20000"
    / "mtx_day"
    / "summary_anchor_body_gap_bins.csv"
)
DEFAULT_OUTDIR = Path("xs_scripts") / "anchor_ratio_20000"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: int
    run_id: str
    strategy_id: str
    bin_mode: str
    anchor_mode: int
    anchor_id: str
    anchor_label: str
    body_bin: str
    gap_bin: str
    body_min: float
    body_max: float
    gap_min: float
    gap_max: float
    penetrate: float
    total_trades: int
    win_rate: float | None
    total_return_rate: float
    pf_net: float | None

    @property
    def use_ratio_mode(self) -> int:
        return 1 if self.bin_mode == BIN_MODE_ANCHOR_RATIO else 0


def _num(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _maybe_num(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num):
        return None
    return num


def _xs_num(value: float, *, fallback: float = 999999.0) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        value = fallback
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def _safe_name(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("._")
    return text or "rule"


def _row_to_spec(row: pd.Series) -> RuleSpec:
    bin_mode = str(row.get("BinMode") or BIN_MODE_POINTS)
    if bin_mode == BIN_MODE_ANCHOR_RATIO:
        body_min = _num(row.get("BodyRatioMinPct"))
        body_max = _maybe_num(row.get("BodyRatioMaxPct"))
        gap_min = _num(row.get("GapRatioMinPct"))
        gap_max = _maybe_num(row.get("GapRatioMaxPct"))
    else:
        body_min = _num(row.get("BodyMin"))
        body_max = _maybe_num(row.get("BodyMax"))
        gap_min = _num(row.get("GapMin"))
        gap_max = _maybe_num(row.get("GapMax"))
    return RuleSpec(
        rule_id=int(_num(row.get("RuleID"))),
        run_id=str(row.get("RunID")),
        strategy_id=str(row.get("StrategyID")),
        bin_mode=bin_mode,
        anchor_mode=int(_num(row.get("AnchorMode"))),
        anchor_id=str(row.get("AnchorID")),
        anchor_label=str(row.get("AnchorLabel")),
        body_bin=str(row.get("BodyBin")),
        gap_bin=str(row.get("GapBin")),
        body_min=body_min,
        body_max=999999.0 if body_max is None else body_max,
        gap_min=gap_min,
        gap_max=999999.0 if gap_max is None else gap_max,
        penetrate=_num(row.get("Penetrate"), 1.0),
        total_trades=int(_num(row.get("TotalTrades"))),
        win_rate=_maybe_num(row.get("WinRate")),
        total_return_rate=_num(row.get("TotalReturnRate")),
        pf_net=_maybe_num(row.get("PFNet")),
    )


def _header(spec: RuleSpec, script_kind: str) -> str:
    mode_text = "定錨比例" if spec.use_ratio_mode else "點數"
    return f"""//=======================================================================
// ScriptName : ROD_{spec.run_id}_{spec.anchor_id}_{script_kind}
// 說明       : A01~A08 第0層 ROD 回踩裸K策略，單一參數完整 XS {script_kind}
//
// RuleID     : {spec.rule_id}
// RunID      : {spec.run_id}
// Strategy   : {spec.strategy_id}
// 區間模式   : {mode_text}
// Anchor     : {spec.anchor_id} {spec.anchor_label}
// 前K實體    : {spec.body_bin}
// OpenGap    : {spec.gap_bin}
// Penetrate  : {spec.penetrate:g}
// 回測摘要   : 次數 {spec.total_trades}，勝率 {_pct(spec.win_rate)}，總報酬率 {_pct(spec.total_return_rate)}
//
// 做多核心：
//   Body 條件：C[1] - O[1] 落在前K實體區間
//   Gap  條件：O0 - A 落在 OpenGap 區間
//   成交條件：Low0 <= A - Penetrate，成交價 A
//
// 做空鏡像：
//   Body 條件：O[1] - C[1] 落在前K實體區間
//   Gap  條件：A - O0 落在 OpenGap 區間
//   成交條件：High0 >= A + Penetrate，成交價 A
//
// 出場規則：
//   成交後下一根 K 的 Open 出場；單根只做一個動作；先出後進；平倉當根不再進場。
//=======================================================================
"""


def _common_body(spec: RuleSpec, *, trading: bool) -> str:
    action_lines = (
        """SetPosition(0, OutPx);"""
        if trading
        else """if HoldSide = 1 then
                Plot3(OutPx);
            if HoldSide = -1 then
                Plot4(OutPx);"""
    )
    force_lines = (
        """SetPosition(0, OutPx);"""
        if trading
        else """Plot5(OutPx);"""
    )
    long_entry = (
        """SetPosition(1, InPx);"""
        if trading
        else """Plot1(InPx);"""
    )
    short_entry = (
        """SetPosition(-1, InPx);"""
        if trading
        else """Plot2(InPx);"""
    )
    plot_clear = "" if trading else """
NoPlot(1);
NoPlot(2);
NoPlot(3);
NoPlot(4);
NoPlot(5);
"""
    set_flat = "" if not trading else """
if BarNo <= 1 then
    SetPosition(0);
"""
    return f"""
input:
    UseTime1(1, "使用日盤"),
    Time1Begin(090500, "日盤可進場起點"),
    Time1End(131000, "日盤最後進場"),
    Time1Force(131200, "日盤強制平倉"),

    UseTime2(1, "使用夜盤"),
    Time2Begin(150300, "夜盤可進場起點"),
    Time2End(235500, "夜盤最後進場"),
    Time2Force(235700, "夜盤強制平倉"),

    UseTime3(1, "使用凌晨盤"),
    Time3Begin(300, "凌晨盤可進場起點"),
    Time3End(45500, "凌晨盤最後進場"),
    Time3Force(45700, "凌晨盤強制平倉"),

    UseRatioMode({spec.use_ratio_mode}, "1=定錨比例，0=點數"),
    AnchorMode({spec.anchor_mode}, "Anchor 模式 A01~A08"),
    BodyMin({_xs_num(spec.body_min)}, "前K實體下限：比例模式為百分比小數，點數模式為點"),
    BodyMax({_xs_num(spec.body_max)}, "前K實體上限：比例模式為百分比小數，點數模式為點"),
    GapMin({_xs_num(spec.gap_min)}, "OpenGap 下限：比例模式為百分比小數，點數模式為點"),
    GapMax({_xs_num(spec.gap_max)}, "OpenGap 上限：比例模式為百分比小數，點數模式為點"),
    Penetrate({_xs_num(spec.penetrate)}, "回踩 / 回抽穿越點數");

variable:
    Initialized(0),
    BarNo(0),
    LastSeenDate(0),
    LastSeenTime(0),

    O0(0),
    O1v(0),
    H1v(0),
    L1v(0),
    C1v(0),
    Range1v(0),
    M1v(0),
    BM1v(0),
    BodyTop1v(0),
    BodyBot1v(0),
    Q75v(0),
    Q25v(0),

    ALong(0),
    AShort(0),
    ALongPx(0),
    AShortPx(0),
    DenLong(0),
    DenShort(0),

    LongBodyV(0),
    ShortBodyV(0),
    LongGapV(0),
    ShortGapV(0),
    LongSetup(0),
    ShortSetup(0),

    CanEnter(0),
    ActiveForceTime(0),
    InForceTime(0),
    DidAction(0),
    CancelBlock(0),
    PendingBarNo(0),
    HoldSide(0),
    InPx(0),
    OutPx(0),
    InBarNo(0),
    LastEntryBarNo(-1000000);

{plot_clear}
if Initialized = 0 then
begin
    Initialized = 1;
    BarNo = 0;
    LastSeenDate = 0;
    LastSeenTime = 0;
    PendingBarNo = 0;
    HoldSide = 0;
    InBarNo = 0;
    InForceTime = 0;
    LastEntryBarNo = -1000000;
end;

if Date <> LastSeenDate or Time <> LastSeenTime then
begin
    BarNo = BarNo + 1;
    LastSeenDate = Date;
    LastSeenTime = Time;
end;

DidAction = 0;
CancelBlock = 0;

if PendingBarNo > 0 and BarNo > PendingBarNo then
begin
    if BarNo = PendingBarNo + 1 then
        CancelBlock = 1;
    PendingBarNo = 0;
end;

{set_flat}
if BarNo > 1 then
begin
    O0 = Open;
    O1v = Open[1];
    H1v = High[1];
    L1v = Low[1];
    C1v = Close[1];

    Range1v = H1v - L1v;
    if Range1v < 0 then
        Range1v = 0;

    BodyTop1v = C1v;
    if O1v > BodyTop1v then
        BodyTop1v = O1v;

    BodyBot1v = C1v;
    if O1v < BodyBot1v then
        BodyBot1v = O1v;

    M1v = (H1v + L1v) / 2;
    BM1v = (O1v + C1v) / 2;
    Q75v = L1v + Range1v * 0.75;
    Q25v = L1v + Range1v * 0.25;

    ALong = C1v;
    AShort = C1v;

    if AnchorMode = 2 then
    begin
        ALong = M1v;
        AShort = M1v;
    end;

    if AnchorMode = 3 then
    begin
        ALong = BM1v;
        AShort = BM1v;
    end;

    if AnchorMode = 4 then
    begin
        ALong = H1v;
        AShort = L1v;
    end;

    if AnchorMode = 5 then
    begin
        ALong = O1v;
        AShort = O1v;
    end;

    if AnchorMode = 6 then
    begin
        ALong = BodyTop1v;
        AShort = BodyBot1v;
    end;

    if AnchorMode = 7 then
    begin
        ALong = BodyBot1v;
        AShort = BodyTop1v;
    end;

    if AnchorMode = 8 then
    begin
        ALong = Q75v;
        AShort = Q25v;
    end;

    ALongPx = IntPortion(ALong + 0.5);
    AShortPx = IntPortion(AShort + 0.5);

    CanEnter = 0;
    ActiveForceTime = 0;

    if UseTime1 = 1 and Time >= Time1Begin and Time <= Time1End and Time < Time1Force then
    begin
        CanEnter = 1;
        ActiveForceTime = Time1Force;
    end;

    if UseTime2 = 1 and Time >= Time2Begin and Time <= Time2End and Time < Time2Force then
    begin
        CanEnter = 1;
        ActiveForceTime = Time2Force;
    end;

    if UseTime3 = 1 and Time >= Time3Begin and Time <= Time3End and Time < Time3Force then
    begin
        CanEnter = 1;
        ActiveForceTime = Time3Force;
    end;

    if HoldSide <> 0 and InForceTime > 0 and Time >= InForceTime then
    begin
        OutPx = Open;
        {force_lines}
        HoldSide = 0;
        InPx = 0;
        InBarNo = 0;
        InForceTime = 0;
        DidAction = 1;
    end;

    if DidAction = 0 and HoldSide <> 0 and BarNo > InBarNo then
    begin
        OutPx = Open;
        {action_lines}
        HoldSide = 0;
        InPx = 0;
        InBarNo = 0;
        InForceTime = 0;
        DidAction = 1;
    end;

    LongSetup = 0;
    ShortSetup = 0;

    if UseRatioMode = 1 then
    begin
        DenLong = ALong;
        if DenLong < 0 then
            DenLong = DenLong * -1;

        DenShort = AShort;
        if DenShort < 0 then
            DenShort = DenShort * -1;

        if DenLong > 0 then
        begin
            LongBodyV = (C1v - O1v) * 100 / DenLong;
            LongGapV = (O0 - ALong) * 100 / DenLong;
            if LongBodyV >= BodyMin and LongBodyV <= BodyMax and LongGapV >= GapMin and LongGapV <= GapMax then
                LongSetup = 1;
        end;

        if DenShort > 0 then
        begin
            ShortBodyV = (O1v - C1v) * 100 / DenShort;
            ShortGapV = (AShort - O0) * 100 / DenShort;
            if ShortBodyV >= BodyMin and ShortBodyV <= BodyMax and ShortGapV >= GapMin and ShortGapV <= GapMax then
                ShortSetup = 1;
        end;
    end
    else
    begin
        LongBodyV = C1v - O1v;
        LongGapV = O0 - ALong;
        ShortBodyV = O1v - C1v;
        ShortGapV = AShort - O0;

        if LongBodyV >= BodyMin and LongBodyV <= BodyMax and LongGapV >= GapMin and LongGapV <= GapMax then
            LongSetup = 1;

        if ShortBodyV >= BodyMin and ShortBodyV <= BodyMax and ShortGapV >= GapMin and ShortGapV <= GapMax then
            ShortSetup = 1;
    end;

    if LongSetup = 1 and ShortSetup = 1 then
    begin
        LongSetup = 0;
        ShortSetup = 0;
    end;

    if DidAction = 0 and HoldSide = 0 and CanEnter = 1 and CancelBlock = 0 and LastEntryBarNo < BarNo - 1 then
    begin
        if LongSetup = 1 then
        begin
            if Low <= ALongPx - Penetrate then
            begin
                HoldSide = 1;
                InPx = ALongPx;
                InBarNo = BarNo;
                InForceTime = ActiveForceTime;
                LastEntryBarNo = BarNo;
                PendingBarNo = 0;
                {long_entry}
                DidAction = 1;
            end
            else
                PendingBarNo = BarNo;
        end;

        if DidAction = 0 and ShortSetup = 1 then
        begin
            if High >= AShortPx + Penetrate then
            begin
                HoldSide = -1;
                InPx = AShortPx;
                InBarNo = BarNo;
                InForceTime = ActiveForceTime;
                LastEntryBarNo = BarNo;
                PendingBarNo = 0;
                {short_entry}
                DidAction = 1;
            end
            else
                PendingBarNo = BarNo;
        end;
    end;
end;
"""


def render_indicator(spec: RuleSpec) -> str:
    return _header(spec, "IND") + _common_body(spec, trading=False)


def render_trading(spec: RuleSpec) -> str:
    return _header(spec, "TRADE") + _common_body(spec, trading=True)


def generate_scripts(summary_csv: Path, outdir: Path, *, limit: int | None = None) -> dict[str, int | Path]:
    df = pd.read_csv(summary_csv)
    if limit is not None:
        df = df.head(limit).copy()

    indicator_dir = outdir / "indicator"
    trade_dir = outdir / "trade"
    indicator_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        spec = _row_to_spec(pd.Series(row))
        stem = _safe_name(
            f"{spec.run_id}_{spec.anchor_id}_B{int(_num(row.get('BodyBinIndex'))):02d}_G{int(_num(row.get('GapBinIndex'))):02d}"
        )
        indicator_path = indicator_dir / f"{stem}_IND.xs"
        trade_path = trade_dir / f"{stem}_TRADE.xs"
        indicator_path.write_text(render_indicator(spec), encoding="utf-8-sig")
        trade_path.write_text(render_trading(spec), encoding="utf-8-sig")
        manifest_rows.append(
            {
                "RuleID": spec.rule_id,
                "RunID": spec.run_id,
                "AnchorID": spec.anchor_id,
                "AnchorLabel": spec.anchor_label,
                "BodyBin": spec.body_bin,
                "GapBin": spec.gap_bin,
                "BinMode": spec.bin_mode,
                "TotalTrades": spec.total_trades,
                "WinRate": spec.win_rate,
                "TotalReturnRate": spec.total_return_rate,
                "PFNet": spec.pf_net,
                "IndicatorFile": str(indicator_path.relative_to(outdir)),
                "TradeFile": str(trade_path.relative_to(outdir)),
            }
        )

    manifest_path = outdir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    readme_path = outdir / "README.txt"
    readme_path.write_text(
        "XS 參數程式碼輸出說明\r\n"
        "========================\r\n\r\n"
        f"來源 summary：{summary_csv}\r\n"
        f"輸出組數：{len(manifest_rows):,}\r\n\r\n"
        "indicator\\：XS 指標板，每組參數一個完整 .xs。\r\n"
        "trade\\：XS 交易板，每組參數一個完整 .xs。\r\n"
        "manifest.csv：RuleID / RunID / 參數 / 檔案對照表。\r\n\r\n"
        "時間預設開啟日盤、夜盤、凌晨盤三段；只跑日盤時，把 UseTime2、UseTime3 改成 0。\r\n"
        "比例版的 BodyMin/BodyMax/GapMin/GapMax 是百分比數值，例如 0.01 表示 0.01%。\r\n",
        encoding="utf-8-sig",
    )
    return {
        "rules": len(manifest_rows),
        "indicator_files": len(manifest_rows),
        "trade_files": len(manifest_rows),
        "outdir": outdir,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one XS indicator script and one XS trading script per parameter row.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N rows for testing.")
    args = parser.parse_args()
    result = generate_scripts(args.summary, args.outdir, limit=args.limit)
    print(f"Done. rules={result['rules']:,}")
    print(f"indicator_files={result['indicator_files']:,}")
    print(f"trade_files={result['trade_files']:,}")
    print(f"outdir={result['outdir']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
