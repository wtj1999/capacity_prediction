import json
import os
import unicodedata
from load_pack_data import *

# ------------------ 配置 ------------------
PACK_MAP_JSON = "./dataset/xz2_pack_data/4000005_pack_data/0BT/0BT_pack_cells_features.json"
OUTPUT_DIR = "./baseline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 要使用的 cell-level 数值特征（示例）
cell_feature_names = [
    "Cell_Capacity", "Cell_OCV4", "Cell_OCV5",
    "Cell_K", "Cell_InR4", "Cell_InR5", "Cell_Times3"
]

# 要预测的 pack-level 目标（示例，和你的 pack_feature 字段对应）
pack_target_names = [
    "13|压差计算(mV)-充电末端压差",
    "1|压差计算(mV)-静态末端压差",
    "5|压差计算(mV)-充电末端压差",
    "6|压差计算(mV)-静态末端压差",
    "7|压差计算(mV)-放电末端压差",
    "7|容量计算(AH)-阶段放电容量",
    "8|压差计算(mV)-静态末端压差"
]

N_CELLS_EXPECTED = 114

# ------------------ 2) Baseline 训练函数 ------------------
import os
import re
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入模型
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def make_safe_name(s: str, max_len: int = 150) -> str:
    """
    将字符串转成安全的文件名：
    - 替换 | / \ 空格 等特殊字符为下划线
    - 移除非 ASCII 的不可见字符
    - 只保留 [A-Za-z0-9._-]
    - 去掉连续的下划线
    - 限制长度
    """
    # 转换为 NFC 规范，避免奇怪的 Unicode 合并字符
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("|", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._")
    reserved = {"CON","PRN","AUX","NUL"} | {f"COM{i}" for i in range(1,10)} | {f"LPT{i}" for i in range(1,10)}
    if s.upper() in reserved:
        s = f"_{s}_"

    return s[:max_len] if len(s) > max_len else s

def train_baseline_models(
    X: "pd.DataFrame",
    y: "pd.DataFrame",
    algos=["catboost", "lgbm", "rf"],   # 可选模型
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "./baseline_outputs",
    num_boost_round: int = 1000
):
    """
    Train selected algorithms per target:
      - CatBoostRegressor (if 'catboost' in algos)
      - LGBMRegressor (if 'lgbm' in algos)
      - RandomForestRegressor (if 'rf' in algos)
    """
    os.makedirs(output_dir, exist_ok=True)

    # split + scaler
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(output_dir, "baseline_scaler.pkl"))

    models = {algo: {} for algo in algos}
    metrics = {algo: {} for algo in algos}

    for target in y.columns:
        y_tr = y_train[target].values
        y_va = y_val[target].values
        safe_target = make_safe_name(target)

        # 1) CatBoost
        if "catboost" in algos:
            cat_model = CatBoostRegressor(
                iterations=num_boost_round,
                learning_rate=0.05,
                depth=4,
                verbose=False,
                random_state=random_state
            )
            cat_model.fit(X_train_s, y_tr)
            y_pred = cat_model.predict(X_val_s)
            models["catboost"][target] = cat_model
            joblib.dump(cat_model, os.path.join(output_dir, f"catboost_{safe_target}.pkl"))
            mse = mean_squared_error(y_va, y_pred)
            metrics["catboost"][target] = {
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_va, y_pred)),
                "mape": float(np.mean(np.abs((y_va - y_pred) / (y_va + 1e-8))) * 100),
                "r2": float(r2_score(y_va, y_pred))
            }
            print(f"[CatBoost] target={target} "
                  f"RMSE={metrics['catboost'][target]['rmse']:.4f}, "
                  f"MAE={metrics['catboost'][target]['mae']:.4f}, "
                  f"MAPE={metrics['catboost'][target]['mape']:.4f}, "
                  f"R2={metrics['catboost'][target]['r2']:.4f}")

            # === 特征重要性 ===
            fi = cat_model.get_feature_importance()
            fi_df = pd.DataFrame({"feature": X.columns, "importance": fi})
            fi_df.sort_values(by="importance", ascending=False, inplace=True)
            fi_path = os.path.join(output_dir, f"catboost_fi_{safe_target}.csv")
            fi_df.to_csv(fi_path, index=False)
            print(f"[CatBoost] Feature importance saved to {fi_path}")
            print(fi_df.head(10))

        # 2) LightGBM
        if "lgbm" in algos:
            lgb_model = LGBMRegressor(
                objective='regression',
                learning_rate=0.05,
                num_leaves=31,
                n_estimators=num_boost_round,
                random_state=random_state,
                verbosity=-1
            )
            lgb_model.fit(X_train_s, y_tr)
            y_pred = lgb_model.predict(X_val_s)
            models["lgbm"][target] = lgb_model
            joblib.dump(lgb_model, os.path.join(output_dir, f"lgbm_{safe_target}.pkl"))
            mse = mean_squared_error(y_va, y_pred)
            metrics["lgbm"][target] = {
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_va, y_pred)),
                "r2": float(r2_score(y_va, y_pred))
            }
            print(f"[LGBM] target={target} "
                  f"RMSE={metrics['lgbm'][target]['rmse']:.4f}, "
                  f"MAE={metrics['lgbm'][target]['mae']:.4f}, "
                  f"R2={metrics['lgbm'][target]['r2']:.4f}")

            # === 特征重要性 ===
            fi = lgb_model.feature_importances_
            fi_df = pd.DataFrame({"feature": X.columns, "importance": fi})
            fi_df.sort_values(by="importance", ascending=False, inplace=True)
            fi_path = os.path.join(output_dir, f"lgbm_fi_{safe_target}.csv")
            fi_df.to_csv(fi_path, index=False)
            print(f"[LGBM] Feature importance saved to {fi_path}")
            print(fi_df.head(10))

        # 3) RandomForest
        if "rf" in algos:
            rf_model = RandomForestRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            )
            rf_model.fit(X_train_s, y_tr)
            y_pred = rf_model.predict(X_val_s)
            models["rf"][target] = rf_model
            joblib.dump(rf_model, os.path.join(output_dir, f"rf_{safe_target}.pkl"))
            mse = mean_squared_error(y_va, y_pred)
            metrics["rf"][target] = {
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_va, y_pred)),
                "r2": float(r2_score(y_va, y_pred)),
                "mape": float(np.mean(np.abs((y_va - y_pred) / (y_va + 1e-8))) * 100)

            }
            print(f"[RF] target={target} "
                  f"RMSE={metrics['rf'][target]['rmse']:.4f}, "
                  f"MAE={metrics['rf'][target]['mae']:.4f}, "
                  f"MAPE={metrics['rf'][target]['mape']:.4f}, "
                  f"R2={metrics['rf'][target]['r2']:.4f}")

            # === 特征重要性 ===
            fi = rf_model.feature_importances_
            fi_df = pd.DataFrame({"feature": X.columns, "importance": fi})
            fi_df.sort_values(by="importance", ascending=False, inplace=True)
            fi_path = os.path.join(output_dir, f"rf_fi_{safe_target}.csv")
            fi_df.to_csv(fi_path, index=False)
            print(f"[RF] Feature importance saved to {fi_path}")
            print(fi_df.head(10))

    return models, scaler, metrics


# ------------------ 3) 主流程示例：调用 build_dataset_from_pack_map 然后训练 ------------------
def main():
    print("加载 pack_data ...")
    with open(PACK_MAP_JSON, "r", encoding="utf-8") as f:
        pack_map = json.load(f)
        del pack_map["03HPB0BT0001EYF260000102"]

    print("构建数据集...")
    X_tensor, X_agg_df, y_df, index_to_pack = build_dataset_from_pack_map(
        pack_map, cell_feature_names, pack_target_names, n_cells=N_CELLS_EXPECTED
    )

    print(f"找到 {len(X_agg_df)} 个可用的 pack，用于训练 baseline。")

    X_agg_df.to_csv(os.path.join(OUTPUT_DIR, "X_agg_df.csv"))
    y_df.to_csv(os.path.join(OUTPUT_DIR, "y_df.csv"))

    models, scaler, metrics = train_baseline_models(X_agg_df, y_df, algos=["catboost"])
    print("训练完成，输出存放在：", OUTPUT_DIR)
    print("metrics:\n", pd.DataFrame(metrics).T)

if __name__ == "__main__":
    main()
