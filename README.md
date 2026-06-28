# 小台指 ROD 回踩回測報表

這個專案用 Python 回測台指 / 小台指 1 分 K 裸 K ROD 回踩策略，並輸出可以直接用瀏覽器開啟的 HTML 報表。

## 底層資料

程式預設使用小台指全日近 6 年合併資料：

```text
C:\XQ\data\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343\FIMTXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt
```

大台指全日近 6 年合併資料可用 `--instrument tx` 指定：

```text
C:\XQ\data\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343\FITXN_1.TF_M1_FULL_MERGED_201912311500_202606261343.txt
```

資料來源集中設定在：

```text
mtx_research/data_sources.py
```

## 開啟報表

雙擊：

```bat
run.bat
```

入口頁：

```text
index.html
```

目前主要報表：

```text
report_outputs/anchor_body_gap_bins_11152/anchor_body_gap_bins_report.html
```

## 重新產生報表

小台指預設資料：

```powershell
python mtx_research/run_anchor_body_bins.py --outdir report_outputs/anchor_body_gap_bins_11152
```

大台指全日資料：

```powershell
python mtx_research/run_anchor_body_bins.py --instrument tx --outdir report_outputs/anchor_body_gap_bins_11152_tx
```

若要指定其他資料檔，可用 `--data` 覆蓋 `--instrument`：

```powershell
python mtx_research/run_anchor_body_bins.py --data "C:\path\your_data.txt"
```

## 第 0 層公式

做多：

```text
O >= A + GapMin
O <= A + GapMax
L <= A - 1
Entry = A
Exit = NextOpen
```

做空鏡像：

```text
O <= A - GapMin
O >= A - GapMax
H >= A + 1
Entry = A
Exit = NextOpen
```

## 成本設定

目前成本設定集中在 `CostConfig`：

- 本金：250,000 元
- 小台 1 點：50 元
- 進場滑點：0 點
- 出場滑點：2 點
- 手續費：單邊 18 元，一次進出 36 元
- 期交稅率：單邊 0.00002，並以單邊四捨五入到元計算
