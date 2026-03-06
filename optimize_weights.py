"""
Marcel法の重み最適化スクリプト

NPBデータ（2015-2025）を使って、Marcel法の年別重み w0/w1/w2 の
最適な組み合わせを探索する。

デフォルト重み: 5/4/3（Tom Tangoオリジナル、MLB用）
NPBで最適な重みは異なる可能性がある？
"""

import os
from datetime import date
from itertools import product

import numpy as np
import pandas as pd

# ---- 設定 ----
DATA_URL = (
    "https://raw.githubusercontent.com/yasunorim/npb-prediction"
    "/master/data/raw/npb_hitters_2015_2025.csv"
)
BIRTHDAY_URL = (
    "https://raw.githubusercontent.com/yasunorim/npb-prediction"
    "/master/data/raw/npb_player_birthdays.csv"
)

REGRESSION_PA = 1200   # 平均回帰の強さ（PA基準）
PEAK_AGE = 29          # 年齢ピーク
AGE_FACTOR = 0.003     # 1歳あたりの変化率
MIN_PA_PROJ = 100      # 予測PAの最低ライン（フィルタ用）
MIN_PA_EVAL = 200      # 評価時の最低PA（実績）
EVAL_YEARS = list(range(2019, 2026))  # 2019-2025（7年）

# 探索範囲
W0_VALS = range(3, 9)   # 最新年: 3〜8
W1_VALS = range(1, 6)   # 1年前: 1〜5
W2_VALS = range(1, 5)   # 2年前: 1〜4
# 合計 6×5×4 = 120 通り


# ---- データ読み込み ----

def load_data():
    print("データ読み込み中...")
    df = pd.read_csv(DATA_URL)
    for col in ["AVG", "OBP", "SLG", "OPS", "PA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        bdays = pd.read_csv(BIRTHDAY_URL)
        bdays["birthday"] = pd.to_datetime(bdays["birthday"], errors="coerce")
        birthday_dict = dict(zip(bdays["player"], bdays["birthday"]))
        print(f"  打者データ: {len(df)} 行")
        print(f"  誕生日データ: {len(birthday_dict)} 選手")
    except Exception as e:
        print(f"  誕生日データ読み込み失敗（年齢調整なし）: {e}")
        birthday_dict = {}

    return df, birthday_dict


# ---- ヘルパー関数 ----

def calc_age(birthday, target_year: int) -> float:
    if pd.isna(birthday):
        return np.nan
    opening = date(target_year, 4, 1)
    age = opening.year - birthday.year
    if (opening.month, opening.day) < (birthday.month, birthday.day):
        age -= 1
    return float(age)


def age_adjustment(age: float) -> float:
    if np.isnan(age):
        return 0.0
    return (PEAK_AGE - age) * AGE_FACTOR


# ---- Marcel計算（OPS・PA予測） ----

def compute_marcel(
    df: pd.DataFrame,
    target_year: int,
    weights: list[int],
    birthday_dict: dict,
) -> pd.DataFrame:
    """指定した重み [w0, w1, w2] でMarcel予測を計算し、DataFrame を返す"""
    years = [target_year - 1, target_year - 2, target_year - 3]

    # リーグ平均OPS（PA加重）を各年で計算
    lg_ops = {}
    for y in years:
        season = df[df["year"] == y]
        total_pa = season["PA"].sum()
        if total_pa > 0:
            lg_ops[y] = (season["OPS"] * season["PA"]).sum() / total_pa

    if not lg_ops:
        return pd.DataFrame()

    lg_ref = lg_ops.get(years[0], lg_ops[max(lg_ops)])

    past_data = df[df["year"].isin(years)]
    players = past_data["player"].unique()

    results = []
    for player in players:
        pdata = past_data[past_data["player"] == player]

        total_w = 0
        w_pa = 0.0
        w_ops = 0.0

        for i, y in enumerate(years):
            w = weights[i]
            season = pdata[pdata["year"] == y]
            if len(season) == 0:
                continue
            row = season.iloc[0]
            pa = row["PA"]
            ops = row["OPS"]
            if pd.isna(pa) or pa == 0 or pd.isna(ops):
                continue
            total_w += w
            w_pa += pa * w
            w_ops += ops * pa * w

        if total_w == 0 or w_pa == 0:
            continue

        avg_pa = w_pa / total_w
        raw_ops = w_ops / w_pa

        # 平均回帰
        proj_ops = (raw_ops * w_pa + lg_ref * REGRESSION_PA) / (w_pa + REGRESSION_PA)

        # 年齢調整
        bd = birthday_dict.get(player)
        age = calc_age(bd, target_year) if bd is not None else np.nan
        proj_ops += age_adjustment(age)

        results.append({"player": player, "proj_OPS": proj_ops, "proj_PA": avg_pa})

    return pd.DataFrame(results)


# ---- MAE計算 ----

def compute_mae(proj_df: pd.DataFrame, actual_df: pd.DataFrame) -> float | None:
    if len(proj_df) == 0:
        return None

    proj_q = proj_df[proj_df["proj_PA"] >= MIN_PA_PROJ][["player", "proj_OPS"]]
    actual_q = actual_df[actual_df["PA"] >= MIN_PA_EVAL][["player", "OPS"]].rename(
        columns={"OPS": "actual_OPS"}
    )
    merged = proj_q.merge(actual_q, on="player")

    if len(merged) < 10:
        return None

    return float((merged["proj_OPS"] - merged["actual_OPS"]).abs().mean())


# ---- メイン ----

def main():
    df, birthday_dict = load_data()

    combos = list(product(W0_VALS, W1_VALS, W2_VALS))
    print(f"\n探索する重みの組み合わせ: {len(combos)} 通り")
    print(f"評価年: {EVAL_YEARS}")
    print(f"評価条件: 予測PA >= {MIN_PA_PROJ}, 実績PA >= {MIN_PA_EVAL}")
    print(f"デフォルト重み（MLB基準）: 5/4/3\n")

    results = []
    for i, (w0, w1, w2) in enumerate(combos):
        weights = [w0, w1, w2]
        year_maes: dict[int, float] = {}

        for target_year in EVAL_YEARS:
            if target_year not in df["year"].values:
                continue
            proj = compute_marcel(df, target_year, weights, birthday_dict)
            actual = df[df["year"] == target_year]
            mae = compute_mae(proj, actual)
            if mae is not None:
                year_maes[target_year] = mae

        if not year_maes:
            continue

        avg_mae = float(np.mean(list(year_maes.values())))
        row = {
            "weights_str": f"{w0}/{w1}/{w2}",
            "w0": w0, "w1": w1, "w2": w2,
            "mae_avg": round(avg_mae, 5),
        }
        for y in EVAL_YEARS:
            row[f"mae_{y}"] = round(year_maes.get(y, float("nan")), 5)
        results.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == len(combos):
            print(f"  {i + 1}/{len(combos)} 完了")

    results_df = pd.DataFrame(results).sort_values("mae_avg").reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    out_path = "results/weight_search_results.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # サマリー表示
    best = results_df.iloc[0]
    default = results_df[results_df["weights_str"] == "5/4/3"]

    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"最適な重み: {best['weights_str']}  (MAE: {best['mae_avg']:.5f})")

    if len(default) > 0:
        default_mae = default.iloc[0]["mae_avg"]
        improvement = (default_mae - best["mae_avg"]) / default_mae * 100
        rank = results_df[results_df["weights_str"] == "5/4/3"].index[0] + 1
        print(f"デフォルト 5/4/3:  (MAE: {default_mae:.5f}, 順位: {rank}/{len(results_df)})")
        print(f"改善率: {improvement:.2f}%")

    print(f"\nTop 15 weight combinations (OPS MAE, {MIN_PA_EVAL}PA+):")
    mae_cols = ["weights_str", "mae_avg"] + [f"mae_{y}" for y in EVAL_YEARS]
    print(results_df[mae_cols].head(15).to_string(index=False))

    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    main()
