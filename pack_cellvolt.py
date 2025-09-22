# resample_pack_cellvolt.py
import os
import sys
import re
import json
import math
from typing import List, Dict
import pandas as pd
import numpy as np
import tqdm as tqdm

def read_csv_with_fallback(path):
    """尝试多种编码读取 CSV（utf-8 -> gbk），返回 DataFrame"""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding='gbk', low_memory=False)
        except Exception as e:
            raise RuntimeError(f"无法读取 CSV 文件 {path}: {e}")

def find_record_file(step_path: str, all_files_in_dir: List[str]):
    """
    给定工步层文件名 step_path（完整路径），在同目录尝试找到对应的记录层文件。
    先尝试直接把 '工步层' 替换为 '记录层'，如找不到再在目录中匹配共享前缀。
    """
    dirname = os.path.dirname(step_path)
    step_name = os.path.basename(step_path)

    # 1) 直接替换
    candidate = step_name.replace("工步层", "记录层")
    candidate_path = os.path.join(dirname, candidate)
    if os.path.exists(candidate_path):
        return candidate_path

    # 2) 若文件名包含 '@工步层' 形式，替换前面的那一段
    if "工步层" in step_name:
        fallback_candidate = step_name.replace("工步层.csv", "记录层.csv")
        fallback_path = os.path.join(dirname, fallback_candidate)
        if os.path.exists(fallback_path):
            return fallback_path

    # 3) 若仍未找到，尝试按 '@' 分隔取工步层前半部分为前缀，匹配目录中任一包含 '记录层' 且包含该前缀的文件
    prefix = None
    m = re.search(r"^(.+?)工步层", step_name)
    if m:
        prefix = m.group(1)
    else:
        # 退而求其次：去掉尾部 '工步层' 之前的最后一个 '@' 后面的段
        if "@工步层" in step_name:
            prefix = step_name.split("@工步层")[0]
    if prefix:
        for f in all_files_in_dir:
            if "记录层" in f and f.startswith(prefix):
                return os.path.join(dirname, f)

    # 4) 最后尝试在该目录中任意包含 '记录层' 并且与工步文件名有很大相似度的文件（同一批次）
    for f in all_files_in_dir:
        if "记录层" in f and step_name.split("@")[1:3] == f.split("@")[1:3]:
            return os.path.join(dirname, f)

    return None

def natural_sort_cell_cols(cols: List[str]) -> List[str]:
    """把 BMS_CellVolt1..BMS_CellVoltN 按数字排序"""
    def keyfun(c):
        m = re.search(r"(\d+)$", c)
        return int(m.group(1)) if m else 0
    return sorted(cols, key=keyfun)

def process_pair(step_csv: str, record_csv: str, output_dir: str):
    print(f"处理: \n  工步层: {step_csv}\n  记录层: {record_csv}")
    # 1. 读取工步层，取第一列前 4 个电池包码（按出现顺序）
    df_step = read_csv_with_fallback(step_csv)
    if df_step.shape[1] == 0:
        print(f"警告: 工步层文件 {step_csv} 无列。跳过。")
        return

    first_col_name = df_step.columns[0]
    pack_codes = list(pd.Series(df_step[first_col_name].dropna().astype(str).str.replace(r'\t', '', regex=True)).unique())
    # 只取前 4 个（如果多于 4），如果少于4 就取所有
    pack_codes = pack_codes[:4]
    if len(pack_codes) == 0:
        print(f"警告: 在 {step_csv} 中没有找到电池包码（第一列为空）。跳过。")
        return

    # 2. 读取记录层
    df_rec = read_csv_with_fallback(record_csv)
    if "累计时间" not in df_rec.columns:
        # 有些文件列名可能带 BOM 或空格，尝试 strip 列名
        df_rec.columns = [c.strip() for c in df_rec.columns]
    if "累计时间" not in df_rec.columns:
        raise RuntimeError(f"记录层文件 {record_csv} 中找不到 '累计时间' 列，请检查列名。当前列样例：{df_rec.columns[:20]}")

    # 提取 BMS_CellVolt* 列
    cell_cols = [c for c in df_rec.columns if str(c).strip().startswith("BMS_CellVolt")]
    cell_cols = natural_sort_cell_cols(cell_cols)

    if len(cell_cols) == 0:
        raise RuntimeError(f"记录层文件 {record_csv} 中未找到任何以 'BMS_CellVolt' 开头的列。")

    # 计算每包 cell 数（通常 102），以及包的数量
    total_cells = len(cell_cols)
    # 尝试与工步层中读取 pack 数（len(pack_codes)）一致，否则以 total_cells // 102 为准（向下取整）
    guessed_pack_count = total_cells // 102
    pack_count = max(len(pack_codes), guessed_pack_count) if guessed_pack_count >= 1 else len(pack_codes)
    # 若 columns 无法整除 102，就把每包数设为 total_cells // pack_count
    if pack_count > 0:
        per_pack = total_cells // pack_count
    else:
        per_pack = 102

    # 如果整除有余数会丢弃最后不足一包的列（并提示）
    if per_pack * pack_count != total_cells:
        print(f"警告: 总 BMS_CellVolt 列数 {total_cells} 不能被 pack_count {pack_count} 整除，按每包 {per_pack} 处理，丢弃末尾 {total_cells - per_pack*pack_count} 列。")

    # 处理时间索引 - 把累计时间转为整数秒
    s = (df_rec['累计时间'].astype(str)
         .str.replace(r'\\t', '', regex=True)  # 字面 "\t"
         .str.replace(r'\\', '', regex=True)  # 所有反斜杠 \
         .str.replace('\t', '', regex=False)  # 真实制表符
         .str.strip()
         .str.strip('"')  # 去双引号
         )

    # 2) 去掉小数秒部分（例如 .650）—— 保留 HH:MM:SS
    s = s.str.replace(r'\.\d+$', '', regex=True)

    # 3) 转为 Timedelta（如果解析失败会得到 NaT）
    td = pd.to_timedelta(s, errors='coerce')

    # 4) 安全地把 Timedelta 转为秒（Int64 支持缺失值）
    if pd.api.types.is_timedelta64_dtype(td.dtype):
        df_rec['累计时间_seconds'] = td.dt.total_seconds().astype('Int64')
    else:
        # 保底方案：逐行调用 total_seconds（不会触发 .dt 错误）
        df_rec['累计时间_seconds'] = td.apply(lambda x: int(x.total_seconds()) if pd.notna(x) else pd.NA).astype(
            'Int64')

    df_rec['累计时间'] = df_rec['累计时间_seconds']

    if df_rec['累计时间'].isnull().all():
        raise RuntimeError(f"记录层 {record_csv} 的 '累计时间' 列无法解析为数字。")
    df_rec = df_rec.dropna(subset=['累计时间']).copy()
    df_rec = df_rec.sort_values('累计时间')
    df_rec = df_rec.drop_duplicates('累计时间', keep='last')  # 每秒应该只有一行，若有重复保留最后一条

    max_sec = int(df_rec['累计时间'].max())
    minute_points = list(range(0, (max_sec // 60 + 1) * 60, 60))  # 0,60,120,...

    # 将 DataFrame 以累计时间为索引，并对 cell 列进行 reindex -> minute_points，用 ffill 填充
    df_cells = df_rec.set_index('累计时间')[cell_cols]

    minute_index = pd.Index(minute_points)

    df_min = df_cells.reindex(minute_index)
    df_min = df_min.ffill().bfill()
    # reindex 到每秒全量索引（以防有丢秒），但只要每分钟点即可：直接 reindex 到 minute_points（秒），然后前向填充
    # df_min = df_cells.reindex(minute_points, method='ffill')  # method ffill: 用上一个数据填充
    # 处理开头若仍是 NaN（即最早 minute 在第一个记录之前），使用 bfill 填充
    # df_min = df_min.fillna(method='bfill').fillna(method='ffill')

    # 构建输出 json 结构
    output = {}
    # 如果 pack_codes 数量小于实际分包数量，使用占位码
    for i in range(pack_count):
        if i < len(pack_codes):
            code = pack_codes[i]
        else:
            code = f"UNKNOWN_PACK_{i+1}"
        start = i * per_pack
        end = start + per_pack
        cols_for_pack = cell_cols[start:end]
        # 每分钟的矩阵：行数 = len(minute_points)，列数 = per_pack
        arr = df_min[cols_for_pack].to_numpy(dtype=float).transpose()
        # 将 numpy 转为嵌套列表
        volt_list = arr.tolist()
        output[code] = {
            "time_seconds": minute_points,   # 每个样本对应的累计秒数（0,60,120,...）
            "BMS_CellVolt": volt_list       # list of lists: 每行是 per_pack 个 volt
        }

    # 保存 json
    pack_name = "_".join(pack_codes)
    out_fname = f"{pack_name}_data.json"
    out_path = os.path.join(output_dir, out_fname)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"已保存：{out_path}")

def main(root_dir: str):
    # 输出目录
    output_dir = os.path.join(root_dir, "pack_json")
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有文件，先收集所有文件名（用于匹配记录层）
    all_files = []
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                all_files.append(os.path.join(dirpath, f))

    # 找到所有工步层文件
    step_files = [p for p in all_files if "工步层" in os.path.basename(p)]
    if len(step_files) == 0:
        print("未在目录中找到任何包含 '工步层' 的 csv 文件。")
        return

    for step_path in tqdm.tqdm(step_files):
        dirpath = os.path.dirname(step_path)
        files_in_dir = [os.path.basename(x) for x in all_files if os.path.dirname(x) == dirpath]
        record_path = find_record_file(step_path, files_in_dir)
        if record_path is None:
            print(f"未找到与 {step_path} 对应的记录层文件，跳过。")
            continue
        process_pair(step_path, record_path, output_dir)

if __name__ == "__main__":
    main(root_dir="D:\jz_pack_data")
