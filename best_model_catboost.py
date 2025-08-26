import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class CatBoostSklearnWrapper(CatBoostRegressor):
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        # 存到自定义变量，避免和只读 property 冲突
        self._sk_feature_importances_ = np.asarray(
            super().get_feature_importance(type='FeatureImportance')
        )
        return self

    @property
    def feature_importances_(self):
        return self._sk_feature_importances_

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def determine_best_iteration_with_cv(X, y, params, cv_folds=5, early_stopping_rounds=2000, random_state=2025):
    """
    在 X/y 上使用 catboost.cv 来确定最佳迭代数（避免验证泄漏）。
    返回 best_iter（int）。
    """
    pool = Pool(X, y)
    cv_params = params.copy()
    cv_params.update({'loss_function': 'RMSE', 'verbose': 1000, 'random_seed': random_state})

    cv_res = cv(pool=pool, params=cv_params, fold_count=cv_folds, early_stopping_rounds=early_stopping_rounds,
                partition_random_seed=random_state, as_pandas=True)
    # 找到 test-RMSE-mean 最小的迭代索引
    try:
        best_iter = int(cv_res['test-RMSE-mean'].idxmin()) + 1
    except Exception:
        best_iter = len(cv_res)

    return best_iter


def train_catboost_with_bset_config(df,
                                      target='frsj10006',
                                      test_size=0.2,
                                      random_state=2025,
                                      final_early_stopping=2000,
                                      save_model_path='catboost_final.joblib'):

    if target not in df.columns:
        raise ValueError(f"目标列 {target} 不在数据集中")

    select_cols = ['hcsj10001_step1', 'hcsj10007_step1', 'hcsj10014_step1', 'hcsj10009_step2',
                         'hcsj10001_step3', 'hcsj10001_step4', 'hcsj10002_step4', 'hcsj10005_step4',
                         'hcsj10002_step5', 'hcsj10001_step6', 'hcsj10007_step6', 'hcsj10009_step6',
                         'gwjz10003', 'yczy10016', 'fjgf10082', 'zjgf10062', 'zjgf10063', 'fjtb10221',
                         'fjtb10222', 'fjtb10223', 'zjtb10252', 'zjtb10253', 'zjtb10254', 'shell_weight',
                         'positive_electrode_area_sum']
    X = df[select_cols].copy()
    y = df[target].copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"样本数量 — 训练: {len(X_train_full)}, 测试: {len(X_test)}")

    best_params = {
        'iterations': 100000,
        'learning_rate': 0.1,
        'depth': 9,
    }

    # --------- 在 train+val 上用 CV 确定最佳迭代数（避免验证集泄漏） ---------
    print('用 train+val 做 CV 确定最佳迭代数...')
    best_iter = determine_best_iteration_with_cv(X_train_full, y_train_full, best_params,
                                                 cv_folds=5,
                                                 early_stopping_rounds=final_early_stopping, random_state=random_state)
    print('确定的最佳迭代数:', best_iter)

    final_params = best_params.copy()
    final_params['iterations'] = int(best_iter * 1.2)
    final_model = CatBoostSklearnWrapper(**final_params)
    print('训练最终模型...')
    final_model.fit(X_train_full, y_train_full, verbose=100)

    # 测试集评估
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    print(f"最终模型在测试集上的表现：MSE={mse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")

    # 特征重要性
    try:
        importances = final_model.get_feature_importance()
        fi_df = pd.DataFrame({'feature': select_cols, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
        print('特征重要性（选中特征）:')
        print(fi_df.head(30))
    except Exception:
        fi_df = None

    # 保存最终模型
    try:
        joblib.dump(final_model, save_model_path)
        print(f"最终模型已保存至 {save_model_path}")
    except Exception as e:
        print(f"无法保存模型: {e}")

    metrics = {'mse': mse, 'r2': r2, 'mape': mape}
    return final_model, select_cols, metrics, fi_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./dataset/xz1_tl_data.csv')
    parser.add_argument('--target', type=str, default='frsj10006')
    parser.add_argument('--save', type=str, default='catboost_final.joblib')
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    model, features, metrics, fi_df = train_catboost_with_bset_config(
        df,
        target=args.target,
        save_model_path=args.save
    )

    print('Done.')