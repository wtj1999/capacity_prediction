import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv, CatboostError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from graphviz import Digraph
import os
import json
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
import shap
from dataset.outlier_detect import detect_and_remove_outliers
from dataset.outlier_cleaning import keep_rows_within_iqr
import re
from io import StringIO

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, max_error
)

def plot_shap(model, data):
    # 构建 shap解释器

    explainer = shap.TreeExplainer(model)

    # 计算测试集的shap值

    shap_values = explainer.shap_values(data)

    labels = data.columns
    plt.figure()
    shap.summary_plot(shap_values, data, feature_names=labels, plot_type="dot")


def data_split(X, y, test_size, time_series, random_state):
    if time_series:
        n_splits = 5  # 可以根据需要选择拆分次数
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # 获取最后一次拆分作为测试集，前面的作为训练+验证
        splits = list(tscv.split(X))
        train_index, test_index = splits[-1]  # 最后一折用作测试集
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]


    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )             #按时间划分

    print(f"样本数量 — 训练: {len(X_train_full)}, 测试: {len(X_test)}")

    return X_train_full, X_test, y_train_full, y_test



def train_catboost_predict(df,
                           val_size=0.01,
                           random_state=2021,
                           model='Catboost',
                           ):
    """
    使用 CatBoost 回归模型预测 'frsj10006'。

    参数:
      df：包含特征及目标值的 DataFrame
      test_size: 测试集比例
      random_state：随机种子

    返回:
      model：训练好的 CatBoost 模型
      X_train, X_test, y_train, y_test：训练/测试分割后的数据
    """
    # 目标列
    target = '电芯总容量'
    if target not in df.columns:
        raise ValueError(f"表中不存在目标列 {target}")

    # 特征列：剔除目标，并排除非数值类型
    feature_cols = [c for c in df.columns if c != target]
    X_train_full, X_val, y_train_full, y_val = data_split(X=df[feature_cols],
                                                            y=df[target],
                                                            test_size=val_size,
                                                            time_series=False,
                                                            random_state=random_state)

    if model == 'Catboost':
        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.1,
            depth=3,
            loss_function='RMSE',
            #eval_metric="R2",
            random_seed=random_state,
            od_type='Iter',
            od_wait=2000,
            verbose=100
        )

        # 传入验证集作为 eval_set 实现早停
        model.fit(
            X_train_full, y_train_full,
            eval_set=(X_val, y_val),
            use_best_model=True
        )

        model.save_model('./model_path/catboost_model_for1899643023932506114.cbm')
        print(f"模型已保存")

        plot_shap(model, X_val)


        # # 测试集评估
        # y_pred = model.predict(X_test)
        #
        # mse = mean_squared_error(y_test, y_pred)
        # r2 = r2_score(y_test, y_pred)
        # mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        # print(f"Test MSE: {mse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        #
        # # 训练集评估
        # y_pred = model.predict(X_train)
        #
        # mse = mean_squared_error(y_train, y_pred)
        # r2 = r2_score(y_train, y_pred)
        # mape = np.mean(np.abs((y_train - y_pred) / y_train)) * 100
        # print(f"Train MSE: {mse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")


        # 特征重要性输出
        fi = model.get_feature_importance()
        fi_df = pd.DataFrame({'feature': feature_cols, 'importance': fi})
        print("Top features:")
        print(fi_df.sort_values('importance', ascending=False).head(20))

    elif model == 'Linear':
        lr = LinearRegression().fit(X_train_full, y_train_full)
        joblib.dump(lr, "./model_path/lr_model.joblib")

    return model


def test_catboost_predict(model='Catboost',
                          folder_path='./dataset/xz1_tl_data/process_origin_data',
                          save_path='./dataset/xz1_tl_data/process_test_data'):
    """
    测试 CatBoost 模型预测。

    参数:
      model_path: 模型文件路径
    """
    if model == 'Catboost':
        model = CatBoostRegressor()
        model.load_model('./model_path/catboost_model_for1899643023932506114.cbm')
    elif model == 'Linear':
        model = joblib.load("./model_path/lr_model.joblib")

    target_col = '电芯总容量'

    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('test_data')],
        key=lambda x: x
    )
    if not csv_files:
        raise FileNotFoundError(f"在 {folder_path} 中未找到符合条件的测试CSV文件")

    results = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        if target_col not in df.columns:
            print(f"文件 {csv_file} 缺少目标列 {target_col}，跳过")
            continue

        if df.empty:
            print(f"文件 {csv_file} 为空，跳过")
            continue


        _df = cols_drop(df, False)

        # 特征列（排除目标列）
        feature_cols = [c for c in _df.columns if c != target_col]
        X_test = _df[feature_cols]
        y_true = _df[target_col]

        # 预测
        y_pred = model.predict(X_test)

        # === 将预测结果和逐行误差保存回 df ===
        df.loc[_df.index, 'y_true'] = y_true
        df.loc[_df.index, 'y_pred'] = y_pred
        df.loc[_df.index, 'abs_error'] = np.abs(y_true - y_pred)
        df.loc[_df.index, 'squared_error'] = (y_true - y_pred) ** 2
        df.loc[_df.index, 'mape'] = np.abs((y_true - y_pred) / y_true) * 100

        df.to_csv(os.path.join(save_path, f'results_for_{csv_file}'), index=False, encoding='utf-8-sig')


        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        maxerr = max_error(y_true, y_pred)

        date_range = csv_file.replace("test_data_", "").replace(".csv", "")
        results.append({
            "file": csv_file,
            "samples": len(df),
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MedAE": medae,
            "MAPE(%)": mape,
            "R²": r2,
            "MaxError": maxerr
        })

        print(f"({date_range}) 测试完成：RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.4f}%")

    # === 绘制特征重要性 ===
    print("\n绘制特征重要性图...")
    fi = model.get_feature_importance()
    feature_names = model.feature_names_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
    fi_df = fi_df.sort_values("importance", ascending=False).head(20)

    # mapping_df = pd.read_csv(os.path.join('./dataset/xz1_tl_data', 'dwd_tl_fr_field_comments.csv'), encoding='utf-8-sig')
    # col_mapping = dict(zip(mapping_df['COLUMN_NAME'], mapping_df['COLUMN_COMMENT']))
    # fi_df["feature"] = fi_df["feature"].map(lambda x: col_mapping.get(x, x))
    # print(fi_df)

    plt.figure(figsize=(8, 6))
    plt.barh(fi_df["feature"], fi_df["importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()  # 让最重要的特征在上面
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_catboost_oblivious_trees(model_path, output_dir="./tree_plots", num_trees=10):
    os.makedirs(output_dir, exist_ok=True)

    model = CatBoostRegressor()
    model.load_model(model_path)
    print(f"模型已加载：{model_path}")

    json_path = os.path.join(output_dir, "model.json")
    model.save_model(json_path, format="json")

    with open(json_path, "r") as f:
        model_json = json.load(f)

    trees = model_json["oblivious_trees"]
    total_trees = len(trees)
    print(f"模型共有 {total_trees} 棵 Oblivious Tree, 将绘制前 {min(num_trees, total_trees)} 棵")

    for idx, tree in enumerate(trees[:num_trees]):
        leaf_values = tree["leaf_values"]
        leaf_count = len(leaf_values)
        tree_depth = int(math.log2(leaf_count))

        dot = Digraph(comment=f"CatBoost Tree #{idx}")

        # 简单示意：每个叶子一个节点
        for leaf_idx, val in enumerate(leaf_values):
            dot.node(f"leaf{leaf_idx}", f"{val:.4f}")

        # 简单连接叶子节点
        for leaf_idx in range(len(leaf_values)-1):
            dot.edge(f"leaf{leaf_idx}", f"leaf{leaf_idx+1}")

        png_path = os.path.join(output_dir, f"tree_{idx}.png")
        dot.render(png_path, format="png", cleanup=True)
        print(f"✅ 保存 Tree #{idx} 图片: {png_path}.png")


def cols_drop(df: pd.DataFrame, is_train: bool):
    _df = df.copy(deep=True)
    _df = _df.drop(columns=['工艺路线ID', '产线名称', '生产结束时间'])
    drop_cols = [
        col for col in _df.columns
        if re.search(r"分容工步(\d+)", col)
           and (
                   int(re.search(r"分容工步(\d+)", col).group(1)) < 1
                   or int(re.search(r"分容工步(\d+)", col).group(1)) > 5
           )
    ]
    _df = _df.drop(columns=drop_cols)
    # _df = _df[['电芯总容量', '电芯重量', '电芯厚度', '注液前重量', '注液后重量', '一次注液_注液前重量', '一次注液_注液后重量',
    #            '分容工步3_结束容量', '分容工步5_结束容量']]
    if is_train:
        # _df = _df.drop(columns=[col for col in df.columns if col.endswith("涂布面密度测量值")])
        _df = _df.dropna()
        # _df, _, report_iqr_all = keep_rows_within_iqr(_df, k=8, how='all')
        _df = _df[_df['电芯总容量'] > 90]

    else:
        _df = _df[_df['电芯总容量'] > 0]
        _df = _df[(df['分容工步3_结束容量'] > 81.77 - 3 * 2.20) & (df['分容工步3_结束容量'] < 81.77 + 3 * 2.20)]
        _df = _df[(df['分容工步5_结束容量'] > 83.59 - 3 * 0.96) & (df['分容工步5_结束容量'] < 83.59 + 3 * 0.96)]


    # df = df.drop(columns=['工艺路线ID', '产线名称'])
    # df = df.drop(columns=[
    #                       '负极涂布_B涂布面密度测量值', '正极涂布_B涂布面密度测量值', '负极涂布_A涂布面密度测量值',
    #                        '正极涂布_A涂布面密度测量值','电芯正极片重量', '电芯负极片重量', '负极涂布_基材涂布面密度测量值',
    #                       '正极涂布_基材涂布面密度测量值', '高温浸润_静置时间', '工步4_结束容量', '工步2_结束容量'
    #                       ])
    # df = df.drop(df.filter(regex=r"^化成工步").columns, axis=1)
    # df = df[['电芯总容量', '化成工步1_开始电压', '化成工步6_结束电压', '一次注液_注液前重量', '一次注液_注液后重量', '壳体重量',
    #          '负极涂布_B涂布面密度测量值','负极涂布_基材涂布面密度测量值', '正极涂布_基材涂布面密度测量值'
    #          '正极涂布_B涂布面密度测量值', '负极涂布_A涂布面密度测量值', '负极涂布_B涂布面密度测量值', '正极涂布_A涂布面密度测量值', '正极涂布_B涂布面密度测量值',
    #                               '电芯正极片重量', '电芯负极片重量',
    #          '电芯正极片料区面积总', '电芯负极片料区面积总', '注液前重量', '注液后重量', '工步1_化成工步1OCV检测', '工步4_化成工步4恒流充电上限电压',
    #          '生产结束时间']]   03HCB01L0000BYF810501955 03HCB01L0000BYF810502087  03HCB01L0000BYF810502973
    # df = df.dropna()
    # df, outliers, report, masks = detect_and_remove_outliers(
    #     df,
    #     numeric_cols=None,
    #     methods=('iqr'),
    #     combine_mode='or',
    #     params={'iqr_k': 1.5},
    #     verbose=True
    # )

    return _df

def _intermediate_reward(self):
    '''
    计算每一步（非终止步）的即时奖励 r_t，用于鼓励已分配电芯集合的一致性（容量、内阻、自放电率）。

    设计思路：
      - 当已分配电芯数 <= 1 时，无一致性可言，返回 0。
      - 否则分别计算已分配电芯在容量(cap)、直流内阻(dcr)、自放电率(sdr)上的相对方差：
          norm_var(x) = var(x) / (mean(x)^2 + eps)
      - 即时奖励 r_t = - (α * cap_var_norm + β * dcr_var_norm + γ * sdr_var_norm)
        三类方差的加权负和，方差越小（越一致）奖励越大（值趋近 0），否则越负。
    '''
    if len(self.assigned) <= 1:
        return 0.0
    acaps = np.array([self.candidates[i]['cap'] for i in self.assigned], dtype=np.float32)
    adcrs = np.array([self.candidates[i]['dcr'] for i in self.assigned], dtype=np.float32)
    asdrs = np.array([self.candidates[i]['sdr'] for i in self.assigned], dtype=np.float32)

    def norm_var(x):
        m = max(1e-3, x.mean())
        return x.var() / (m * m)

    alpha, beta, gamma = 0.5, 0.3, 0.2
    score = - (alpha * norm_var(acaps) + beta * norm_var(adcrs) + gamma * norm_var(asdrs))   #过程一致性惩罚
    return float(score)

def _terminal_reward(self, metrics):
    '''
    计算 episode 终止时的整体奖励 R_T，用以衡量最终 Pack 性能（容量接近度、动态压差与电芯利用率）。

    设计思路：
      - 终止奖励由三部分线性组合构成：
          1) 容量误差项：-cap_err (cap_err = |pack_capacity - target|/target)
          2) 动态压差：-vdrop
          3) 利用率奖励 util_score (只有当 pack_capacity >= target 且在容忍范围内时才给正激励；未达标时按比例惩罚)
      - 最终 R_T = w1*(-cap_err) + w2*(-vdrop) + w3 * util_score
    '''
    pack_capacity = metrics['pack_capacity']
    vdrop = metrics['dynamic_voltage_drop_V']

    cap_err = abs(pack_capacity - self.target_capacity) / max(1.0, self.target_capacity)

    if pack_capacity > self.target_capacity:
        if cap_err < self.capacity_tolerance:
            util_score = 1.0 - (cap_err / self.capacity_tolerance)   #正奖励
        else:
            util_score = - ((pack_capacity - self.target_capacity) / (self.capacity_tolerance * self.target_capacity)) #弱惩罚
    else:
        util_score = -self.lambda_pen * cap_err  #强惩罚

    w1, w2, w3 = 0.5, 0.3, 0.2
    R_T = w1 * (-cap_err) + w2 * (-vdrop) + w3 * util_score
    return R_T



if __name__ == '__main__':
    # 读取数据
    is_train = True
    if is_train:
        df = pd.read_csv('./dataset/xz1_tl_data/process_train_data/train_data_for_1899643023932506114.csv')
        df["生产结束时间"] = pd.to_datetime(df["生产结束时间"], errors="coerce")
        df = df.sort_values(by="生产结束时间", ascending=True).reset_index(drop=True)
        # df = df[(df["生产结束时间"] >= "2025-03-01") & (df["生产结束时间"] <= "2025-07-01")]
        df = cols_drop(df, True)
        _ = train_catboost_predict(df)

        test_catboost_predict()

    else:
        test_catboost_predict()
    # === 示例调用 ===
    # visualize_catboost_oblivious_trees(
    #     model_path="./model_path/catboost_model_for1899643023932506114.cbm",
    #     output_dir="./tree_plots",
    #     num_trees=10
    # )
