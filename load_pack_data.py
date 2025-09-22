import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
from collections import Counter
try:
    from scipy import stats as _ss
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def build_dataset_from_pack_map(pack_map: Dict,
                                cell_feature_names: List[str],
                                pack_target_names: List[str],
                                n_cells: int = 114
                                ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    适配 pack_map 结构为:
    {
      "pack_code1": {
         "pack_level": { ... },
         "cell_level": {
             "cell_code1": { "Cell_Capacity": "...", ... },
             "cell_code2": { ... },
             ...
         }
      }, ...
    }

    """

    def _entropy_from_list(l):
        if len(l) == 0:
            return 0.0
        c = Counter(l)
        total = sum(c.values())
        ent = 0.0
        for v in c.values():
            p = v / total
            ent -= p * np.log(p + 1e-12)
        return float(ent)

    def _skew_kurt(col):
        # 返回 (skew, kurtosis). 如果有 scipy 用 scipy，否则用 numpy 近似（中心矩）
        if _HAS_SCIPY:
            try:
                return float(_ss.skew(col)), float(_ss.kurtosis(col))
            except Exception:
                pass
        # numpy approximate (may be less stable)
        m = np.mean(col)
        s = np.std(col, ddof=0) + 1e-12
        z = (col - m) / s
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
        return skew, kurt



    X_tensor_list = []
    agg_records = []
    y_records = []
    index_to_pack = []

    # 指定需要进行异常判定的字段及阈值（可以按需扩展）
    ocv_outlier_fields = {"Cell_OCV4": 5.0, "Cell_OCV5": 5.0}

    for pack_code, content in pack_map.items():
        # 基本检查
        if not isinstance(content, dict):
            continue
        pack_level = content.get("pack_level")
        cell_level = content.get("cell_level")
        if pack_level is None or cell_level is None:
            continue

        # cell 数量要求
        if len(cell_level) != n_cells:
            # 可根据需要打印/记录被跳过的 pack_code
            continue

        # 确定 cell 顺序（这里按 cell key 字符串排序，保证顺序确定）
        cell_keys = sorted(cell_level.keys())

        cell_vectors = []
        ok = True
        for ck in cell_keys:
            fv = cell_level.get(ck)
            if not isinstance(fv, dict):
                ok = False
                break
            row_vals = []
            for fn in cell_feature_names:
                if fn not in fv:
                    ok = False
                    break
                val = fv[fn]
                # 检查非空
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    ok = False
                    break
                # 尝试转换为 float（Baseline 假设数值特征）
                try:
                    fval = float(val)
                except Exception:
                    ok = False
                    break

                # 异常值剔除：如果是 OCV4/OCV5 且大于阈值 -> 整个 pack 跳过
                if fn in ocv_outlier_fields and fval > ocv_outlier_fields[fn]:
                    # 打印日志，便于排查
                    print(f"跳过 pack {pack_code}：电芯 {ck} 的 {fn}={fval} 超过阈值 {ocv_outlier_fields[fn]}")
                    ok = False
                    break

                row_vals.append(fval)
            if not ok:
                break
            cell_vectors.append(row_vals)
        if not ok:
            continue

        # pack targets 检查并转换为 float
        try:
            yv = [float(pack_level[t]) for t in pack_target_names]
        except Exception:
            continue

        # 保存 tensor 数据
        X_tensor_list.append(np.array(cell_vectors, dtype=np.float32))
        index_to_pack.append(pack_code)
        y_records.append(yv)

        # 计算聚合统计（用于 baseline）
        arr = np.array(cell_vectors, dtype=np.float64)  # shape (n_cells, m)
        rec = {"pack_product_code": pack_code}
        for j, fn in enumerate(cell_feature_names):
            col = arr[:, j]

            mean_v = float(np.mean(col))
            std_v = float(np.std(col, ddof=0))
            min_v = float(np.min(col))
            max_v = float(np.max(col))
            median_v = float(np.median(col))
            p10 = float(np.percentile(col, 10))
            p90 = float(np.percentile(col, 90))
            rng = max_v - min_v
            iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
            skew, kurt = _skew_kurt(col)

            # 相邻差分统计（对串联特别重要）
            adj = np.abs(np.diff(col))  # length n_cells-1
            adj_mean = float(np.mean(adj))
            adj_max = float(np.max(adj))
            adj_std = float(np.std(adj, ddof=0))

            # top/bottom k（以k=3）
            k = min(3, len(col))
            topk_mean = float(np.mean(np.sort(col)[-k:]))
            bottomk_mean = float(np.mean(np.sort(col)[:k]))

            # 百分比异常（小于 mean - 2*std）
            thr = mean_v - 2 * std_v
            pct_below_thr = float((np.sum(col < thr) / len(col)) * 100.0)

            # CV
            cv = float(std_v / (mean_v + 1e-12))

            # 写入 rec
            rec[f"{fn}_mean"] = mean_v
            rec[f"{fn}_std"] = std_v
            rec[f"{fn}_min"] = min_v
            rec[f"{fn}_max"] = max_v
            rec[f"{fn}_median"] = median_v
            rec[f"{fn}_p10"] = p10
            rec[f"{fn}_p90"] = p90
            rec[f"{fn}_range"] = rng
            rec[f"{fn}_iqr"] = iqr
            rec[f"{fn}_skew"] = skew
            rec[f"{fn}_kurtosis"] = kurt
            rec[f"{fn}_adj_mean"] = adj_mean
            rec[f"{fn}_adj_max"] = adj_max
            rec[f"{fn}_adj_std"] = adj_std
            rec[f"{fn}_top{k}_mean"] = topk_mean
            rec[f"{fn}_bottom{k}_mean"] = bottomk_mean
            rec[f"{fn}_pct_below_mean_minus_2std"] = pct_below_thr
            rec[f"{fn}_cv"] = cv

        # 额外的 cross-feature / domain-specific 特征
        # 如果包含 OCV4 和 OCV5，计算 OCV5-OCV4 的统计（反映不同测试点的差异）
        if "Cell_OCV4" in cell_feature_names and "Cell_OCV5" in cell_feature_names:
            i4 = cell_feature_names.index("Cell_OCV4")
            i5 = cell_feature_names.index("Cell_OCV5")
            ocv_diff = arr[:, i5] - arr[:, i4]
            rec["OCV5_minus_OCV4_mean"] = float(np.mean(ocv_diff))
            rec["OCV5_minus_OCV4_std"] = float(np.std(ocv_diff, ddof=0))
            rec["OCV5_minus_OCV4_max"] = float(np.max(ocv_diff))
            rec["OCV5_minus_OCV4_p90"] = float(np.percentile(ocv_diff, 90))
            rec["OCV_diff_adj_mean"] = float(np.mean(np.abs(np.diff(ocv_diff))))
            rec["OCV_diff_adj_max"] = float(np.max(np.abs(np.diff(ocv_diff))))

        # 如果包含 InR4 和 InR5，计算差值
        if "Cell_InR4" in cell_feature_names and "Cell_InR5" in cell_feature_names:
            j4 = cell_feature_names.index("Cell_InR4")
            j5 = cell_feature_names.index("Cell_InR5")
            inr_diff = arr[:, j5] - arr[:, j4]
            rec["InR5_minus_InR4_mean"] = float(np.mean(inr_diff))
            rec["InR5_minus_InR4_std"] = float(np.std(inr_diff, ddof=0))
            rec["InR_diff_adj_mean"] = float(np.mean(np.abs(np.diff(inr_diff))))

        # 容量相关的特殊统计（如果有 Cell_Capacity）
        if "Cell_Capacity" in cell_feature_names:
            ic = cell_feature_names.index("Cell_Capacity")
            cap_col = arr[:, ic]
            rec["capacity_cv"] = float(np.std(cap_col, ddof=0) / (float(np.mean(cap_col)) + 1e-12))
            rec["capacity_top3_mean"] = float(np.mean(np.sort(cap_col)[-3:]))
            rec["capacity_bottom3_mean"] = float(np.mean(np.sort(cap_col)[:3]))
            rec["capacity_pct_below_mean_minus_2std"] = float(
                (np.sum(cap_col < (np.mean(cap_col) - 2 * np.std(cap_col))) / len(cap_col)) * 100.0)
            rec["capacity_adj_max"] = float(np.max(np.abs(np.diff(cap_col))))

        # 整体邻位差汇总（跨所有特征）：计算每个相邻对在所有特征上的 L2 差，然后统计
        adj_diffs = np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))  # length n_cells-1
        rec["adj_pair_l2_mean"] = float(np.mean(adj_diffs))
        rec["adj_pair_l2_max"] = float(np.max(adj_diffs))
        rec["adj_pair_l2_std"] = float(np.std(adj_diffs, ddof=0))

        # 尝试从原始 cell_level 中提取类别信息（如果存在），比如 Cell_Lot / Cell_MatchName
        cell_dicts = [cell_level[ck] for ck in cell_keys]
        lots = []
        matches = []
        for cd in cell_dicts:
            if isinstance(cd, dict):
                lot = cd.get("Cell_Lot") or cd.get("CellLot") or cd.get("lot")
                match = cd.get("Cell_MatchName") or cd.get("CellMatchName") or cd.get("match")
                if lot is not None:
                    lots.append(str(lot))
                if match is not None:
                    matches.append(str(match))
        rec["lot_unique_count"] = int(len(set(lots)))
        rec["lot_entropy"] = float(_entropy_from_list(lots))
        rec["match_unique_count"] = int(len(set(matches)))
        rec["match_entropy"] = float(_entropy_from_list(matches))
        if len(lots) > 0:
            most_common_lot_frac = max(Counter(lots).values()) / len(lots)
            rec["lot_most_common_frac"] = float(most_common_lot_frac)
        else:
            rec["lot_most_common_frac"] = 0.0

        agg_records.append(rec)


    if len(X_tensor_list) == 0:
        raise ValueError("没有找到满足条件的 pack（请检查 pack_map、特征名或 n_cells）。")

    # 堆叠结果并构建 DataFrame
    X_tensor = np.stack(X_tensor_list, axis=0)  # (N, n_cells, m)
    X_agg_df = pd.DataFrame(agg_records).set_index("pack_product_code")
    y_df = pd.DataFrame(y_records, index=index_to_pack, columns=pack_target_names)
    # 对齐顺序（保证 y_df 与 X_agg_df 行顺序一致）
    y_df = y_df.loc[X_agg_df.index]

    return X_tensor, X_agg_df, y_df, index_to_pack


if __name__ == "__main__":
    PACK_MAP_JSON = "./dataset/xz2_pack_data/4000005_pack_data/0BT/0BT_pack_cells_features.json"
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
    print("加载 pack_data ...")
    with open(PACK_MAP_JSON, "r", encoding="utf-8") as f:
        pack_map = json.load(f)
        del pack_map["03HPB0BT0001EYF260000102"]

    print("构建数据集...")
    X_tensor, X_agg_df, y_df, index_to_pack = build_dataset_from_pack_map(
        pack_map, cell_feature_names, pack_target_names, n_cells=N_CELLS_EXPECTED
    )
    print(111111111111)
