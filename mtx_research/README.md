# mtx_research

這個資料夾放小台指 1 分 K ROD / Trend Pullback 研究程式。

## 目前主要報表

第 0 層 Anchor x Body x OpenGap：

```powershell
python mtx_research/run_anchor_body_bins.py --data "FIMTX_M1_202001020845.txt" --outdir "report_outputs/anchor_body_gap_bins_3528"
```

輸出：

```text
report_outputs/anchor_body_gap_bins_3528/anchor_body_gap_bins_report.html
report_outputs/anchor_body_gap_bins_3528/summary_anchor_body_gap_bins.csv
report_outputs/anchor_body_gap_bins_3528/by_year_anchor_body_gap_bins.csv
```

## 策略簡碼

做多：

```text
B=C1-O1 in 前K實體區間
O>=A+Gap下限
O<=A+Gap上限
L<=A-1
Entry=A
Exit=NextOpen
```

做空鏡像：

```text
B=O1-C1 in 前K實體區間
O<=A-Gap下限
O>=A-Gap上限
H>=A+1
Entry=A
Exit=NextOpen
```

## 成本

`CostConfig` 預設：

- 本金：250,000 元
- 小台 1 點：50 元
- 進場滑點：0 點
- 出場滑點：2 點
- 來回手續費：36 元
- 期交稅：單邊 0.00002，單邊四捨五入到元

## 其他研究程式

- `run_xs_anchor_rod.py`：XS Anchor ROD 18,816 組版本。
- `run_research.py`：R2A / R2B 長時間研究流程。
- `anchor_body_bins.py`：目前 GitHub 網頁入口使用的 3,528 組報表產生器。
