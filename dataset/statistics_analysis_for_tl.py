import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from outlier_detect import *
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']   # 指定默认字体
rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块







def analyze_and_plot(df, columns, output_dir="analysis_plots_tl"):
    """
    对指定的三列数据进行详细统计分析并生成可视化图表
    结果统一保存到 output_dir 文件夹
    """
    # 创建文件夹
    os.makedirs(output_dir, exist_ok=True)

    # df, _, _, _ = detect_and_remove_outliers(
    #     df,
    #     numeric_cols=None,
    #     methods=('mad', 'iqr', 'mahalanobis', 'isolation_forest'),
    #     combine_mode='vote',
    #     params={'mad_thresh': 3.5, 'iqr_k': 1.5, 'mahalanobis_p': 0.001, 'iso_contamination': 0.05},
    #     verbose=True
    # )


    # 统计分析
    stats = df[columns].describe(percentiles=[0.25, 0.5, 0.75])
    stats.to_csv(os.path.join(output_dir, "statistics.csv"), encoding="utf-8-sig")
    print(stats)

    # 绘制每列的直方图 + KDE
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        median_val = df[col].median()

        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=200, color="skyblue")

        # 均值线
        plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.2f}")
        # ±1 标准差
        plt.axvline(mean_val - std_val, color='green', linestyle=':', label=f"-1 SD: {mean_val - std_val:.2f}")
        plt.axvline(mean_val + std_val, color='green', linestyle=':', label=f"+1 SD: {mean_val + std_val:.2f}")
        # 中位数
        plt.axvline(median_val, color='purple', linestyle='-.', label=f"Median: {median_val:.2f}")

        # 添加文本统计信息（右上角）
        text_str = f"Mean = {mean_val:.2f}\nMedian = {median_val:.2f}\nStd = {std_val:.2f}"
        plt.text(0.98, 0.95, text_str, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

        plt.title(f"{col} - Histogram & KDE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist_kde.png"))
        plt.close()

if __name__ == "__main__":
    is_train = True
    if not is_train:
        folder_path = './xz1_tl_data/process_origin_data'
        csv_files = sorted(
            [
                f for f in os.listdir(folder_path)
                if f.endswith('.csv')
                   and f.startswith('test_data')
                   and "202508" in f
            ],
            key=lambda x: x
        )
        dfs = []

        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path, encoding='utf-8-sig')

            if df.empty:
                print(f"文件 {csv_file} 为空，跳过")
                continue

            dfs.append(df)

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df = merged_df[(merged_df['分容工步3_结束容量'] < 65) & (merged_df['分容工步3_结束容量'] > 50)]
            merged_df = merged_df[merged_df['电芯总容量'] > 80]
            print("合并后的 DataFrame 形状:", merged_df.shape)

    else:
        file_path = './xz1_tl_data/process_train_data/train_data_for_1899643023932506114.csv'
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        merged_df = df[(df['分容工步3_结束容量'] > 70) & (df['分容工步3_结束容量'] < 90)]
        merged_df = merged_df[merged_df['电芯总容量'] > 80]
        print("合并后的 DataFrame 形状:", merged_df.shape)




    analyze_and_plot(df=merged_df,
                     columns=['分容工步3_结束容量', '分容工步5_结束容量', '电芯总容量'],
                     output_dir="analysis_plots_tl",
                     )