from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtx_research.anchor_body_bins import BIN_MODE_ANCHOR_RATIO, WINRATE_THRESHOLDS, _threshold_label
from mtx_research.generate_xs_scripts import _maybe_num, _num, _pct, _xs_num


DEFAULT_REPORT_DIR = Path("report_outputs") / "four_layer_anchor_ratio_20000"
DEFAULT_OUTDIR = Path("xs_scripts") / "anchor_ratio_threshold_24"


@dataclass(frozen=True)
class LayerConfig:
    key: str
    label: str
    slug: str
    use_time1: int
    use_time2: int
    use_time3: int


@dataclass(frozen=True)
class RuleCond:
    rule_id: int
    run_id: str
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


LAYERS = (
    LayerConfig("mtx_day", "小台日盤", "MTX_DAY", 1, 0, 0),
    LayerConfig("mtx_all", "小台全日", "MTX_ALL", 1, 1, 1),
    LayerConfig("tx_day", "大台日盤", "TX_DAY", 1, 0, 0),
    LayerConfig("tx_all", "大台全日", "TX_ALL", 1, 1, 1),
)


def _rule_from_row(row: pd.Series) -> RuleCond:
    body_max = _maybe_num(row.get("BodyRatioMaxPct"))
    gap_max = _maybe_num(row.get("GapRatioMaxPct"))
    return RuleCond(
        rule_id=int(_num(row.get("RuleID"))),
        run_id=str(row.get("RunID")),
        anchor_mode=int(_num(row.get("AnchorMode"))),
        anchor_id=str(row.get("AnchorID")),
        anchor_label=str(row.get("AnchorLabel")),
        body_bin=str(row.get("BodyBin")),
        gap_bin=str(row.get("GapBin")),
        body_min=_num(row.get("BodyRatioMinPct")),
        body_max=999999.0 if body_max is None else body_max,
        gap_min=_num(row.get("GapRatioMinPct")),
        gap_max=999999.0 if gap_max is None else gap_max,
        penetrate=_num(row.get("Penetrate"), 1.0),
        total_trades=int(_num(row.get("TotalTrades"))),
        win_rate=_maybe_num(row.get("WinRate")),
        total_return_rate=_num(row.get("TotalReturnRate")),
    )


def _select_rules(summary: pd.DataFrame, threshold: float) -> pd.DataFrame:
    valid = summary["TotalTrades"] > 0
    if threshold >= 1.0:
        selected = summary[valid & (summary["WinRate"] >= 1.0 - 1e-12)].copy()
    else:
        selected = summary[valid & (summary["WinRate"] > threshold)].copy()
    return selected.sort_values(
        ["TotalReturnRate", "WinRate", "TotalTrades", "RuleID"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _summary_lookup(threshold_summary: pd.DataFrame, threshold: float) -> dict[str, object]:
    rows = threshold_summary[abs(threshold_summary["Threshold"] - threshold) < 1e-12]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _time_inputs(layer: LayerConfig) -> str:
    return f"""input:
    UseTime1({layer.use_time1}, "使用日盤"),
    Time1Begin(090500, "日盤可進場起點"),
    Time1End(131000, "日盤最後進場"),
    Time1Force(131200, "日盤強制平倉"),

    UseTime2({layer.use_time2}, "使用夜盤"),
    Time2Begin(150300, "夜盤可進場起點"),
    Time2End(235500, "夜盤最後進場"),
    Time2Force(235700, "夜盤強制平倉"),

    UseTime3({layer.use_time3}, "使用凌晨盤"),
    Time3Begin(300, "凌晨盤可進場起點"),
    Time3End(45500, "凌晨盤最後進場"),
    Time3Force(45700, "凌晨盤強制平倉");
"""


def _var_block() -> str:
    return """
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

    A1L(0),
    A1S(0),
    A2L(0),
    A2S(0),
    A3L(0),
    A3S(0),
    A4L(0),
    A4S(0),
    A5L(0),
    A5S(0),
    A6L(0),
    A6S(0),
    A7L(0),
    A7S(0),
    A8L(0),
    A8S(0),
    CurALong(0),
    CurAShort(0),
    DenLong(0),
    DenShort(0),

    LongBodyV(0),
    ShortBodyV(0),
    LongGapV(0),
    ShortGapV(0),
    RuleLongOK(0),
    RuleShortOK(0),

    MatchRuleID(0),
    MatchSide(0),
    MatchLongPx(0),
    MatchShortPx(0),
    MatchPenetrate(0),

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
"""


def _anchor_expr(anchor_mode: int, side: str) -> str:
    suffix = "L" if side == "long" else "S"
    return f"A{anchor_mode}{suffix}"


def _rule_block(rule: RuleCond) -> str:
    long_anchor = _anchor_expr(rule.anchor_mode, "long")
    short_anchor = _anchor_expr(rule.anchor_mode, "short")
    return f"""
    if MatchRuleID = 0 then
    begin
        CurALong = {long_anchor};
        CurAShort = {short_anchor};
        RuleLongOK = 0;
        RuleShortOK = 0;

        DenLong = CurALong;
        if DenLong < 0 then
            DenLong = DenLong * -1;

        DenShort = CurAShort;
        if DenShort < 0 then
            DenShort = DenShort * -1;

        if DenLong > 0 then
        begin
            LongBodyV = (C1v - O1v) * 100 / DenLong;
            LongGapV = (O0 - CurALong) * 100 / DenLong;
            if LongBodyV >= {_xs_num(rule.body_min)} and LongBodyV <= {_xs_num(rule.body_max)} and LongGapV >= {_xs_num(rule.gap_min)} and LongGapV <= {_xs_num(rule.gap_max)} then
                RuleLongOK = 1;
        end;

        if DenShort > 0 then
        begin
            ShortBodyV = (O1v - C1v) * 100 / DenShort;
            ShortGapV = (CurAShort - O0) * 100 / DenShort;
            if ShortBodyV >= {_xs_num(rule.body_min)} and ShortBodyV <= {_xs_num(rule.body_max)} and ShortGapV >= {_xs_num(rule.gap_min)} and ShortGapV <= {_xs_num(rule.gap_max)} then
                RuleShortOK = 1;
        end;

        if RuleLongOK = 1 and RuleShortOK = 1 then
        begin
            RuleLongOK = 0;
            RuleShortOK = 0;
        end;

        if RuleLongOK = 1 then
        begin
            MatchRuleID = {rule.rule_id};
            MatchSide = 1;
            MatchLongPx = IntPortion(CurALong + 0.5);
            MatchPenetrate = {_xs_num(rule.penetrate)};
        end;

        if MatchRuleID = 0 and RuleShortOK = 1 then
        begin
            MatchRuleID = {rule.rule_id};
            MatchSide = -1;
            MatchShortPx = IntPortion(CurAShort + 0.5);
            MatchPenetrate = {_xs_num(rule.penetrate)};
        end;
    end;
"""


def _header(
    *,
    layer: LayerConfig,
    threshold: float,
    threshold_label: str,
    rules: list[RuleCond],
    summary_row: dict[str, object],
    script_kind: str,
) -> str:
    return f"""//=======================================================================
// ScriptName : ROD_RATIO_{layer.slug}_{_threshold_slug(threshold)}_{script_kind}
// 說明       : {layer.label}｜{threshold_label}｜定錨比例版 ROD 門檻整合 XS {script_kind}
//
// 參數池     : {len(rules):,} 組
// 報表次數   : {int(_num(summary_row.get("TotalTrades"))):,}
// 報表勝率   : {_pct(_maybe_num(summary_row.get("WinRate")))}
// 報表總報酬 : {_pct(_maybe_num(summary_row.get("TotalReturnRate")))}
// 報表PF     : {_xs_num(_num(summary_row.get("PFNet"), 0))}
//
// 重要：
//   報表統計是把門檻內所有參數逐組加總。
//   本 XS 是單一策略整合版；同一根若多組同時符合，依排序取第一組。
//   排序依總報酬率、勝率、次數由高到低。
//
// 做多：
//   Body% = (C[1] - O[1]) * 100 / abs(A)
//   Gap%  = (O0 - A) * 100 / abs(A)
//   符合參數池任一組後，Low0 <= A - Penetrate 才成交，成交價 A。
//
// 做空鏡像：
//   Body% = (O[1] - C[1]) * 100 / abs(A)
//   Gap%  = (A - O0) * 100 / abs(A)
//   符合參數池任一組後，High0 >= A + Penetrate 才成交，成交價 A。
//
// 出場：
//   成交後下一根 K 的 Open 出場；單根只做一個動作；先出後進；平倉當根不再進場。
//=======================================================================
"""


def _threshold_slug(threshold: float) -> str:
    if threshold >= 1.0:
        return "EQ100"
    return f"GT{int(threshold * 100)}"


def _threshold_file_label(threshold: float) -> str:
    if threshold >= 1.0:
        return "勝率_100"
    return f"勝率大於{int(threshold * 100)}"


def _common_script(
    *,
    layer: LayerConfig,
    threshold: float,
    threshold_label: str,
    rules: list[RuleCond],
    summary_row: dict[str, object],
    trading: bool,
) -> str:
    script_kind = "TRADE" if trading else "IND"
    force_action = "SetPosition(0, OutPx);" if trading else "Plot5(OutPx);"
    exit_action = (
        "SetPosition(0, OutPx);"
        if trading
        else """if HoldSide = 1 then
                Plot3(OutPx);
            if HoldSide = -1 then
                Plot4(OutPx);"""
    )
    long_action = "SetPosition(1, InPx);" if trading else "Plot1(InPx);"
    short_action = "SetPosition(-1, InPx);" if trading else "Plot2(InPx);"
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
    rule_blocks = "".join(_rule_block(rule) for rule in rules)
    return (
        _header(
            layer=layer,
            threshold=threshold,
            threshold_label=threshold_label,
            rules=rules,
            summary_row=summary_row,
            script_kind=script_kind,
        )
        + "\n"
        + _time_inputs(layer)
        + _var_block()
        + f"""
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

    A1L = C1v;
    A1S = C1v;
    A2L = M1v;
    A2S = M1v;
    A3L = BM1v;
    A3S = BM1v;
    A4L = H1v;
    A4S = L1v;
    A5L = O1v;
    A5S = O1v;
    A6L = BodyTop1v;
    A6S = BodyBot1v;
    A7L = BodyBot1v;
    A7S = BodyTop1v;
    A8L = Q75v;
    A8S = Q25v;

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
        {force_action}
        HoldSide = 0;
        InPx = 0;
        InBarNo = 0;
        InForceTime = 0;
        DidAction = 1;
    end;

    if DidAction = 0 and HoldSide <> 0 and BarNo > InBarNo then
    begin
        OutPx = Open;
        {exit_action}
        HoldSide = 0;
        InPx = 0;
        InBarNo = 0;
        InForceTime = 0;
        DidAction = 1;
    end;

    MatchRuleID = 0;
    MatchSide = 0;
    MatchLongPx = 0;
    MatchShortPx = 0;
    MatchPenetrate = 0;
{rule_blocks}
    if DidAction = 0 and HoldSide = 0 and CanEnter = 1 and CancelBlock = 0 and LastEntryBarNo < BarNo - 1 and MatchRuleID > 0 then
    begin
        if MatchSide = 1 then
        begin
            if Low <= MatchLongPx - MatchPenetrate then
            begin
                HoldSide = 1;
                InPx = MatchLongPx;
                InBarNo = BarNo;
                InForceTime = ActiveForceTime;
                LastEntryBarNo = BarNo;
                PendingBarNo = 0;
                {long_action}
                DidAction = 1;
            end
            else
                PendingBarNo = BarNo;
        end;

        if DidAction = 0 and MatchSide = -1 then
        begin
            if High >= MatchShortPx + MatchPenetrate then
            begin
                HoldSide = -1;
                InPx = MatchShortPx;
                InBarNo = BarNo;
                InForceTime = ActiveForceTime;
                LastEntryBarNo = BarNo;
                PendingBarNo = 0;
                {short_action}
                DidAction = 1;
            end
            else
                PendingBarNo = BarNo;
        end;
    end;
end;
"""
    )


def render_indicator(layer: LayerConfig, threshold: float, rules: list[RuleCond], summary_row: dict[str, object]) -> str:
    return _common_script(
        layer=layer,
        threshold=threshold,
        threshold_label=_threshold_label(threshold),
        rules=rules,
        summary_row=summary_row,
        trading=False,
    )


def render_trading(layer: LayerConfig, threshold: float, rules: list[RuleCond], summary_row: dict[str, object]) -> str:
    return _common_script(
        layer=layer,
        threshold=threshold,
        threshold_label=_threshold_label(threshold),
        rules=rules,
        summary_row=summary_row,
        trading=True,
    )


def generate_threshold_scripts(report_dir: Path, outdir: Path) -> dict[str, int | Path]:
    indicator_dir = outdir / "indicator"
    trade_dir = outdir / "trade"
    indicator_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)
    for folder in (indicator_dir, trade_dir):
        for old_file in folder.glob("*.xs"):
            old_file.unlink()

    manifest_rows: list[dict[str, object]] = []
    total_rules = 0
    for layer in LAYERS:
        layer_dir = report_dir / layer.key
        summary = pd.read_csv(layer_dir / "summary_anchor_body_gap_bins.csv")
        if "BinMode" in summary.columns:
            summary = summary[summary["BinMode"] == BIN_MODE_ANCHOR_RATIO].copy()
        threshold_summary = pd.read_csv(layer_dir / "winrate_threshold_summary.csv")

        for threshold in WINRATE_THRESHOLDS:
            selected = _select_rules(summary, threshold)
            rules = [_rule_from_row(pd.Series(row)) for row in selected.to_dict(orient="records")]
            summary_row = _summary_lookup(threshold_summary, threshold)
            threshold_slug = _threshold_slug(threshold)
            stem = f"{layer.slug}_{threshold_slug}"
            indicator_path = indicator_dir / f"{stem}_IND.xs"
            trade_path = trade_dir / f"{stem}_TRADE.xs"
            indicator_path.write_text(render_indicator(layer, threshold, rules, summary_row), encoding="utf-8-sig")
            trade_path.write_text(render_trading(layer, threshold, rules, summary_row), encoding="utf-8-sig")
            total_rules += len(rules)
            manifest_rows.append(
                {
                    "LayerKey": layer.key,
                    "Layer": layer.label,
                    "Threshold": threshold,
                    "ThresholdLabel": _threshold_label(threshold),
                    "RuleCount": len(rules),
                    "ReportTrades": int(_num(summary_row.get("TotalTrades"))),
                    "ReportWinRate": _maybe_num(summary_row.get("WinRate")),
                    "ReportReturnRate": _maybe_num(summary_row.get("TotalReturnRate")),
                    "FirstRuleID": rules[0].rule_id if rules else "",
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
        "XS 勝率門檻整合版輸出說明\r\n"
        "============================\r\n\r\n"
        f"來源報表：{report_dir}\r\n"
        f"輸出門檻：{len(manifest_rows):,} 套，也就是 4 層 × 6 個勝率門檻。\r\n"
        f"門檻內參數總數：{total_rules:,} 組。\r\n\r\n"
        "indicator\\：XS 指標板，每個勝率門檻一個完整 .xs。\r\n"
        "trade\\：XS 交易板，每個勝率門檻一個完整 .xs。\r\n"
        "manifest.csv：門檻、參數數量、績效摘要、檔案對照表。\r\n\r\n"
        "注意：報表統計是把門檻內所有參數逐組加總；XS 單一策略版同一根只能做一個動作，\r\n"
        "所以同一根若多組同時符合，會依總報酬率、勝率、次數排序取第一組。\r\n",
        encoding="utf-8-sig",
    )
    return {
        "groups": len(manifest_rows),
        "indicator_files": len(manifest_rows),
        "trade_files": len(manifest_rows),
        "total_rules": total_rules,
        "outdir": outdir,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 24 XS scripts for four-layer win-rate threshold buttons.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    result = generate_threshold_scripts(args.report_dir, args.outdir)
    print(f"Done. groups={result['groups']:,}")
    print(f"indicator_files={result['indicator_files']:,}")
    print(f"trade_files={result['trade_files']:,}")
    print(f"total_rules_inside_groups={result['total_rules']:,}")
    print(f"outdir={result['outdir']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
