# merge_json_cellvolt.py
import os
import sys
import json
import numpy as np
from glob import glob

def load_json_files(folder, pattern="*.json"):
    files = sorted(glob(os.path.join(folder, pattern)))
    return files

def pad_tail_with_last(arr, target_len):
    """
    arr: numpy array shape (cells, t)
    target_len: desired t'
    返回 pad 后的 (cells, target_len)，若 t >= target_len 则截断到 target_len
    """
    cells, t = arr.shape
    if t == 0:
        # 直接用 zeros 填充（但这种情况通常不应出现）
        return np.zeros((cells, target_len), dtype=arr.dtype)
    if t >= target_len:
        return arr[:, :target_len]
    # pad by repeating last column
    pad_len = target_len - t
    last_col = arr[:, -1][:, None]  # shape (cells,1)
    pads = np.repeat(last_col, pad_len, axis=1)
    return np.concatenate([arr, pads], axis=1)

def process_folder(folder, out_prefix="combined_cells", file_pattern="*_data.json", expected_cells_per_pack=102, save_each_pack=False):
    files = load_json_files(folder, pattern=file_pattern)
    if not files:
        raise RuntimeError(f"No json files found in {folder} with pattern {file_pattern}")

    per_pack_arrays = []   # list of arrays shape (cells, time)
    pack_codes_list = []   # track pack code strings in same order

    max_t = 0
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            j = json.load(fh)
        # j expected: { pack_code: {"time_seconds": [...], "BMS_CellVolt": [[..],[..],...]} , ... }
        for pack_code, v in j.items():
            if not isinstance(v, dict):
                print(f"跳过 {f} 的键 {pack_code}（不是 dict）")
                continue
            cellvolt = v.get("BMS_CellVolt") or v.get("BMS_CellVolt".strip())
            if cellvolt is None:
                print(f"文件 {f} 中 pack {pack_code} 未找到 BMS_CellVolt，跳过")
                continue
            # convert to numpy array
            arr = np.array(cellvolt, dtype=float)  # shape (time, cells_per_pack)
            if arr.ndim != 2:
                print(f"警告: pack {pack_code} 在文件 {f} 的 BMS_CellVolt 维度异常: {arr.shape}, 跳过")
                continue

            cells = arr.shape[0]
            if expected_cells_per_pack is not None and cells != expected_cells_per_pack:
                print(f"注意: pack {pack_code} 的单体数量为 {cells}，与期望 {expected_cells_per_pack} 不一致")
            t = arr.shape[1]
            print(f"pack {pack_code} 的时间长度为 {t}")
            if t > max_t:
                max_t = t
            per_pack_arrays.append(arr)
            pack_codes_list.append(pack_code)

    if len(per_pack_arrays) == 0:
        raise RuntimeError("未找到任何可处理的 pack 数据。")

    # 统一填充到 max_t
    padded_per_pack = []
    for arr in per_pack_arrays:
        padded = pad_tail_with_last(arr, max_t)
        padded_per_pack.append(padded)

    # 合并：把每个 pack 的 102 行纵向拼接 -> (N_packs * cells_per_pack, max_t)
    all_cells = np.vstack(padded_per_pack)

    # 保存文件
    out_combined_npy = os.path.join(folder, f"{out_prefix}.npy")
    np.save(out_combined_npy, all_cells)
    print(f"已保存合并数组：{out_combined_npy}  形状: {all_cells.shape}")

    # 保存每个 pack 单独文件（可选）
    if save_each_pack:
        for idx, (code, arr) in enumerate(zip(pack_codes_list, padded_per_pack)):
            safe_code = "".join(c if c.isalnum() or c in "-_" else "_" for c in code)
            outp = os.path.join(folder, f"pack_{idx+1}_{safe_code}.npy")
            np.save(outp, arr)
        print(f"已保存每个 pack 的单独 npy 文件（共 {len(padded_per_pack)} 个）")

    return all_cells, padded_per_pack, pack_codes_list

if __name__ == "__main__":
    all_cells, per_pack, codes = process_folder('D:\jz_pack_data\pack_json')
