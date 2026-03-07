# NPB Marcel Weight Study

Marcel法のパラメータをNPBデータで最適化する検証プロジェクト。

## 背景

[Marcel法](https://www.tangotiger.net/marcel/)はTom Tangoが考案した選手成績予測手法で、NPB成績予測（[npb-prediction](https://github.com/yasumorishima/npb-prediction)）でも採用している。

Marcel法には3つのパラメータがある:

| パラメータ | 意味 | MLBデフォルト |
|---|---|---|
| w0 / w1 / w2 | 直近3年の重み（直近年 / 1年前 / 2年前） | 5 / 4 / 3 |
| REGRESSION_PA | 平均回帰の強さ（打者、PA換算） | 1200 |
| REGRESSION_IP | 平均回帰の強さ（投手、IP換算） | 600 |

これらはすべて**MLBデータで設定されたもの**。NPBでも同じ値が最適とは限らない。

## 検証内容

### グリッドサーチ

| 対象 | 探索パラメータ | 組み合わせ数 |
|---|---|---|
| 打者 | w0(3-8) × w1(1-5) × w2(1-4) × REG_PA(6種) | 720通り |
| 投手 | w0(3-8) × w1(1-5) × w2(1-4) × REG_IP(5種) | 600通り |

評価指標:
- 打者: OPS / OBP / SLG / AVG（いずれも MAE）
- 投手: ERA / WHIP（いずれも MAE）

### クロスバリデーション

評価年: 2019〜2025（7年）。各年の成績を直近3年から予測し、実績と比較。

2つのシナリオで評価:
- **2020込み**: コロナ短縮シーズンを含む
- **2020除外**: 120試合の特殊シーズンを除いた6年

### ブートストラップ検定

最適パラメータ vs MLBデフォルト（5/4/3）の改善が統計的に有意かを確認（300回リサンプリング）。

## 暫定結果（第1回: 重みのみ探索）

> 第1回はREG_PAを1200固定、OPSのみ評価した暫定結果。

| 重み | OPS MAE (7年平均) |
|---|---|
| **4/3/1** | **0.06146** |
| 5/2/2 | 0.06155 |
| 5/3/1 | 0.06156 |
| 5/4/3（MLBデフォルト） | 0.06227（50位/120位） |

**改善率: 1.30%**

傾向:
- 最新年の重みは4で十分（MLBの5より低い）
- 2年前の重みは1〜2が最適（MLBの3より低い）→ **NPBは直近重視、古いデータの効きが弱い**

> 包括検証（REG_PA最適化・投手・ブートストラップ含む）は実行中。results/ に結果が保存される。

## 結果ファイル（results/）

| ファイル | 内容 |
|---|---|
| `hitter_grid_full.csv` | 打者グリッドサーチ全結果（2020込み） |
| `hitter_grid_no2020.csv` | 打者グリッドサーチ全結果（2020除外） |
| `pitcher_grid_full.csv` | 投手グリッドサーチ全結果（2020込み） |
| `pitcher_grid_no2020.csv` | 投手グリッドサーチ全結果（2020除外） |
| `bootstrap.csv` | ブートストラップ分布（差分） |

## 実行方法

GitHub Actions（手動トリガー）:

```
Actions → Optimize Marcel Weights → Run workflow
```

ローカルでの実行は非推奨（クラウド処理推奨）。

## データソース

- 打者・投手成績: [yasumorishima/npb-prediction](https://github.com/yasumorishima/npb-prediction) — NPB公式データ（baseball-data.com + npb.jp）
- 対象: 2015〜2025年

## 関連プロジェクト

- [npb-prediction](https://github.com/yasumorishima/npb-prediction): Marcel法によるNPB成績予測・順位予測
- [baseball-mlops](https://github.com/yasumorishima/baseball-mlops): MLB Statcastデータを使ったLightGBM + Marcel比較
