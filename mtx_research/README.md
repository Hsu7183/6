# mtx_research

這個資料夾放台指 / 小台指 1 分 K ROD 與 Trend Pullback 研究程式。

## 資料來源

所有預設底層資料集中設定在：

```text
mtx_research/data_sources.py
```

小台指預設 `--instrument mtx`：

```text
C:\XQ\data\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt
```

大台指使用 `--instrument tx`：

```text
C:\XQ\data\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt
```

## 第 0 層 Anchor x Body x OpenGap 報表

小台指：

```powershell
python mtx_research/run_anchor_body_bins.py --outdir report_outputs/anchor_body_gap_bins_11152
```

大台指：

```powershell
python mtx_research/run_anchor_body_bins.py --instrument tx --outdir report_outputs/anchor_body_gap_bins_11152_tx
```

## XS Anchor ROD 18,816 組

小台指：

```powershell
python mtx_research/run_xs_anchor_rod.py --outdir report_outputs/xs_anchor_rod_18816
```

大台指：

```powershell
python mtx_research/run_xs_anchor_rod.py --instrument tx --outdir report_outputs/xs_anchor_rod_18816_tx
```

## 成本設定

`CostConfig` 預設：

- 本金：250,000 元
- 小台 1 點：50 元
- 進場滑點：0 點
- 出場滑點：2 點
- 手續費：單邊 18 元，一次進出 36 元
- 期交稅率：單邊 0.00002
