import numpy as np
import pandas as pd
import re
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

def data_process():
    df = pd.read_excel('./dataset/pack_data/20250726081641854_03HPB0BT0001EYF7S0000035_DJ2318D_测试数据.xlsx', sheet_name="详细数据")
    df["累计时间"] = df["累计时间"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    df = df[[col for col in df.columns if re.search(r'[\u4e00-\u9fff]', col)]]
    df = df[["累计时间"] + ["工步序号"] + [col for col in df.columns if re.match(r"单体电压\d+", col)]]
    df = df.melt(
        id_vars=["累计时间", "工步序号"],
        var_name="电芯实际位置",
        value_name="单体电压"
    )
    df["电芯实际位置"] = df["电芯实际位置"].str.extract(r"单体电压(\d+)").astype(int)

    df_dx = pd.read_excel('./dataset/pack_data/20250726163120470.xlsx')
    df_dx = df_dx.drop(columns=['ID', 'Pack条码', '大模组序号', '大模组条码', '小模组序号', '小模组条码', '模块序号', '模块条码', '电芯序号', '电芯批次号'])
    df_merged = pd.merge(df, df_dx, on="电芯实际位置", how="left")
    df_merged["累计时间_秒"] = pd.to_timedelta(df_merged["累计时间"]).dt.total_seconds()
    df_merged.to_csv('./dataset/pack_data/03HPB0BT0001EYF7S0000035.csv', index=False)


def train_model():
    df = pd.read_csv('./dataset/pack_data/03HPB0BT0001EYF7S0000035.csv')
    unique_cells = df['电芯条码'].unique()
    np.random.seed(42)
    train_cells = np.random.choice(unique_cells, size=100, replace=False)
    test_cells = [cell for cell in unique_cells if cell not in train_cells]

    train_df = df[df['电芯条码'].isin(train_cells)].copy()
    test_df = df[df['电芯条码'].isin(test_cells)].copy()

    feature_cols = [col for col in df.columns if col not in ['电芯条码', '单体电压', '工步序号', '电芯实际位置', '累计时间', '时间', '电芯OCV4时间']]
    X_train = train_df[feature_cols]
    y_train = train_df['单体电压']
    X_test = test_df[feature_cols]
    y_test = test_df['单体电压']

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100
    )

    # 4. 训练模型
    model.fit(X_train, y_train)

    fi = model.get_feature_importance()
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': fi})
    print("Top features:")
    print(fi_df.sort_values('importance', ascending=False).head(20))

    # 5. 预测
    y_pred = model.predict(X_test)

    # 6. 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    result_df = test_df[['电芯条码', '工步序号']].copy()
    result_df['真实单体电压'] = y_test.values
    result_df['预测单体电压'] = y_pred
    result_df.to_csv('./dataset/pack_data/prediction_results.csv', index=False, encoding='utf-8-sig')
    print("预测结果已保存到 ./dataset/pack_data/prediction_results.csv")

if __name__ == '__main__':
    # data_process()
    train_model()
