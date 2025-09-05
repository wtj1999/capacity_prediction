import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import seaborn as sns
from outlier_detect import detect_and_remove_outliers
from dataset.outlier_cleaning import keep_rows_within_iqr

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


save_dir = "./corr_plots_for_cap"
os.makedirs(save_dir, exist_ok=True)


def plot_top_corr(df, target_col, top_n=20, title="整体相关性",
                  save_path=None, show=False, cmap="plasma"):
    """
    df: DataFrame
    target_col: 目标列
    top_n: 显示前N个特征
    cmap: 渐变配色方案 (默认 plasma，可选 'coolwarm' 'Blues' 'magma' 'Spectral' 等)
    """
    # 计算相关系数
    corr = df.corr(numeric_only=True)[target_col].dropna()

    # 排序并取前n
    top_corr = corr.abs().sort_values(ascending=False).head(top_n + 1)  # +1 因为包含自身
    top_corr = top_corr.drop(target_col, errors='ignore')

    # 可视化
    plt.figure(figsize=(10, 6))
    values = top_corr.sort_values()

    # 使用渐变色
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(values)))

    ax = values.plot(kind='barh', color=colors, edgecolor="black")
    plt.title(title, fontsize=12)
    plt.xlabel("Pearson相关系数", fontsize=12)

    # 调整字体
    plt.yticks(fontsize=12)
    ax.tick_params(axis="x", labelsize=12)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    return top_corr

def process_folder(folder_path, target_col="电芯总容量", top_n=20, cmap="plasma", output_dir="output_corr_plots"):
    """
    批量处理文件夹下的 csv 文件，绘制相关性图
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.endswith(".csv") and "test_data" in file:
            file_path = os.path.join(folder_path, file)
            print(f"正在处理: {file}")

            try:
                df = pd.read_csv(file_path)
                df = drop_cols_func(df)
                title = f"{file} - {target_col}相关性"
                save_path = os.path.join(output_dir, file.replace(".csv", "_corr.png"))

                plot_top_corr(df, target_col, top_n=top_n, title=title, save_path=save_path, cmap=cmap)

            except Exception as e:
                print(f"⚠️ 文件 {file} 处理失败: {e}")


def drop_cols_func(df: pd.DataFrame):
    drop_cols = [
        col for col in df.columns
        if re.search(r"分容工步(\d+)", col)
           and (
                   int(re.search(r"分容工步(\d+)", col).group(1)) < 1
                   or int(re.search(r"分容工步(\d+)", col).group(1)) > 5
           )
    ]
    df = df.drop(columns=drop_cols)
    df = df.drop(columns=['高温浸润_静置时间'])
    df = df.dropna()
    df = df[df['电芯总容量'] > 0]
    df, _, report_iqr_all = keep_rows_within_iqr(df, k=6, how='all')

    return df


df = pd.read_csv('./xz1_tl_data/process_origin_data/test_data_20250712_20250713.csv')
print(df)



if __name__ == "__main__":
    folder = "./xz1_tl_data/process_origin_data"  # 修改成你的文件夹路径
    process_folder(folder, target_col="电芯总容量", top_n=20, cmap="magma")


# ==== 数据处理 ====
df = pd.read_csv('./xz1_tl_data/process_train_data/train_data_for_1899643023932506114.csv')

df = drop_cols_func(df)
# 整体相关性
plot_top_corr(
    df,
    target_col="电芯总容量",
    top_n=20,
    title="整体相关性Top20",
    save_path=os.path.join(save_dir, "整体相关性Top20.png")
)

# 按月分组相关性
df["生产结束时间"] = pd.to_datetime(df["生产结束时间"])
df["月份"] = df["生产结束时间"].dt.to_period("M")

for month, group in df.groupby("月份"):
    print(f"\n==== {month} 月 ====")
    if group.shape[0] < 5:  # 太少数据跳过
        continue
    filename = f"{month}_相关性Top20.png"
    save_path = os.path.join(save_dir, filename)
    top_corr = plot_top_corr(
        group,
        target_col="电芯总容量",
        top_n=20,
        title=f"{month} 月相关性Top20",
        save_path=save_path
    )







