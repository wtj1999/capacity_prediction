import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re

'''
df = pd.read_csv('../dataset/xz1_tl_data/process_test_data/results_for_test_data_20250721_20250722.csv')
df = df[df['y_true']> 90].iloc[200:1200]

plt.figure(figsize=(10,6))

# 绘制 y_true
plt.scatter(df.index, df["y_true"], label="y_true", alpha=0.7)

# 绘制 y_pred
plt.scatter(df.index, df["y_pred"], label="y_pred", alpha=0.7)

plt.xlabel("Cell Index")
plt.ylabel("Value")
plt.title("y_true vs y_pred 1000 cells in 20250713")
plt.legend()
plt.grid(True)
plt.show()

plt.hist(df["y_true"], bins=50, alpha=0.6, label="y_true")
plt.hist(df["y_pred"], bins=50, alpha=0.6, label="y_pred")

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of y_true and y_pred in 20250714")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["y_true"], y=df["y_pred"], alpha=0.6)
plt.plot([df["y_true"].min(), df["y_true"].max()],
         [df["y_true"].min(), df["y_true"].max()],
         'r--', label="Ideal Fit")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.title("y_true vs y_pred (Scatter with Ideal Line)")
plt.legend()
plt.grid(True)
plt.show()

residuals = df["y_pred"] - df["y_true"]

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=50, alpha=0.7, color="orange")
plt.axvline(0, color="red", linestyle="--", label="Zero Error")
plt.xlabel("Residual (y_pred - y_true)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(df.index, residuals, alpha=0.7, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Cell Index")
plt.ylabel("Residual (y_pred - y_true)")
plt.title("Residuals vs Cell Index")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(y=residuals, color="lightblue")
plt.axhline(0, color="red", linestyle="--")
plt.ylabel("Residual (y_pred - y_true)")
plt.title("Residuals Boxplot")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_normal_with_scores(mu=0, sigma=1):
    """
    绘制正态分布曲线并标注区间得分
    区间划分:
        x < μ-2σ          -> 得分1
        μ-2σ ~ μ-σ        -> 得分0.8
        μ-σ ~ μ+σ         -> 得分0.6
        μ+σ ~ μ+2σ        -> 得分0.2
        x >= μ+2σ         -> 得分0
    """
    # 横坐标范围
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 800)
    y = norm.pdf(x, mu, sigma)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label="Key Parameter Distribution", color="blue")

    # 区间定义
    regions = [
        (x < mu - 2*sigma, "score=1"),
        ((x >= mu - 2*sigma) & (x < mu - sigma), "score=0.8"),
        ((x >= mu - sigma) & (x < mu + sigma), "score=0.6"),
        ((x >= mu + sigma) & (x < mu + 2*sigma), "score=0.2"),
        (x >= mu + 2*sigma, "score=0"),
    ]

    colors = ["#FF9999", "#FFD700", "#90EE90", "#87CEFA", "#D3D3D3"]

    # 绘制不同区间
    for i, (mask, label) in enumerate(regions):
        plt.fill_between(x, 0, y, where=mask, color=colors[i], alpha=0.6)
        if mask.any():  # 确保区间非空
            x_mid = x[mask].mean()
            y_mid = y[mask].max() * 0.5
            plt.text(x_mid, y_mid, label, ha="center", fontsize=12, weight="bold")

    # plt.title(f"正态分布区间得分 (μ={mu}, σ={sigma})", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

# 示例调用
plot_normal_with_scores(mu=0, sigma=1)




