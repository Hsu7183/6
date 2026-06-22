# 小台指 ROD 回踩回測報表

這個專案用 Python 回測小台指 1 分 K 裸 K ROD 回踩策略，並輸出可以直接用瀏覽器開啟的 HTML 報表。

## 最新網頁報表

本機雙擊：

```bat
run.bat
```

或直接開啟：

```text
index.html
```

最新報表位置：

```text
report_outputs/anchor_body_gap_bins_3528/anchor_body_gap_bins_report.html
```

## 目前第 0 層測試

總組合：

```text
8 種 Anchor x 21 組前 K 實體區間 x 21 組 OpenGap 區間 = 3,528 組
```

做多簡碼：

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

## 成本設定

- 小台指 1 點：50 元
- 報酬率本金：250,000 元
- 進場滑點：0 點
- 出場滑點：2 點
- 手續費：單邊 18 元，來回 36 元
- 期交稅：單邊 0.00002，單邊四捨五入到元

## 重跑報表

把原始資料放在專案根目錄：

```text
FIMTX_M1_202001020845.txt
```

執行：

```powershell
python mtx_research/run_anchor_body_bins.py --data FIMTX_M1_202001020845.txt --outdir report_outputs/anchor_body_gap_bins_3528
```

## GitHub 同步範圍

原始資料與大型中間輸出不放進 GitHub，避免超過 GitHub 單檔限制與造成 repo 過大。同步範圍包含：

- Python 程式
- `index.html`
- `run.bat`
- 最新 3,528 組 HTML 報表
- 最新 summary / by_year CSV
