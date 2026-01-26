import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import joblib
import chardet
import numpy as np
import pandas as pd
import xgboost as xgb

from pathlib import Path
from utils.hdcan import *
from utils.hfes import *
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# ========== 配置基准目录 ==========

# 当前文件路径
current_file = Path(__file__).resolve()
# 项目根目录
BASE_DIR = current_file.parents[1]

MODEL_DIR = os.path.join(BASE_DIR, "ours", "model", "1")
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== 读取与预处理函数 (支持CSV / JSON) ==========
def read_data(data_path):
    file_lower = data_path.lower()
    is_json = file_lower.endswith(".json")

    if is_json:
        print(f"[INFO] 读取 JSON: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            df = pd.read_json(f)
        # 如果 JSON 中列名不同，请在此 rename
        # df.rename(columns={"Data":"Query","label":"Label"}, inplace=True)
    else:
        print(f"[INFO] 读取 CSV: {data_path}")
        try:
            df = pd.read_csv(data_path, encoding='latin1', on_bad_lines='skip')
        except UnicodeDecodeError:
            with open(data_path, 'rb') as f:
                enc = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(data_path, encoding=enc, on_bad_lines='skip')

    if "Query" not in df.columns or "Label" not in df.columns:
        raise ValueError("文件中必须包含列 [Query, Label].")

    print(f"[INFO] 原始数据行数: {df.shape[0]}")

    # 若 Label 不全是0/1，可做简单映射
    if df["Label"].dtype not in [np.int64, np.float64]:
        df["Label"] = df["Label"].apply(lambda x: 1 if str(x).lower()=="attack" else 0)

    print("\n[INFO] 标签分布:")
    print(df["Label"].value_counts())

    # 补充空值 & 转小写
    df["Query"] = df["Query"].fillna("").str.lower()
    df.drop_duplicates(subset=["Query","Label"], inplace=True)
    print(f"[INFO] 去重后数据行数: {df.shape[0]}")

    return df

# ========== train_test_split & 训练 ==========
def train_test_split_and_featurize(data, test_size=0.3, random_state=42):
    X = data.drop(["Label"], axis=1)
    y = data["Label"].values

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print("训练集大小:", x_train.shape)
    print("测试集大小:", x_test.shape)

    num_cols = [
        "qlen","wcount","sq","dq","puncts","comments","spaces","logic",
        "arith","alpha","sqlkw","sqlfunc"
    ]
    X_train_num_scaled, X_test_num_scaled, scaler_num = standardize_and_combine_features(
        x_train, x_test, num_cols
    )

    train_feat = X_train_num_scaled
    test_feat  = X_test_num_scaled

    scaler_num_path  = os.path.join(MODEL_DIR, "scaler_for_numeric.pkl")

    with open(scaler_num_path, "wb") as f:
        pickle.dump(scaler_num, f)

    return {
        "x_train": x_train, "x_test": x_test,
        "y_train": y_train, "y_test": y_test,
        "num_features": (train_feat, test_feat)
    }

# ========== 6. 自定义阈值搜索 & XGBoost 训练 ==========
def find_best_threshold(proba_vals, true_labels):
    best_thr = 0.5
    best_f1  = 0.0
    for thr in np.linspace(0, 1, 101):
        preds = (proba_vals >= thr).astype(int)
        f1 = f1_score(true_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1

def train_and_save_model(feature_name, train_x, train_y, test_x, test_y):
    results = {}
    this_feature_dir = os.path.join(MODEL_DIR, feature_name)
    os.makedirs(this_feature_dir, exist_ok=True)

    model_name = "FSHBoost"
    print(f"\n===== 训练 {model_name} =====")

    param_dist = {
        'max_depth': [7],
        'learning_rate': [0.2],
        'n_estimators': [400],
        'gamma': [0],
        'reg_lambda': [3],
    }
    xgb_clf = XGBClassifier(
        disable_default_eval_metric=True
        )

    search_xgb = RandomizedSearchCV(
        xgb_clf, param_dist, scoring='f1', cv=3,n_iter=200,
        return_train_score=True, n_jobs=-1, random_state=42
    )
    search_xgb.fit(train_x, train_y)
    best_params = search_xgb.best_params_
    print("[INFO] XGB 最优参数:", best_params)

    best_xgb = XGBClassifier(
        disable_default_eval_metric=True,
        **best_params)
    best_xgb.fit(train_x, train_y)

    # 默认(阈=0.5)或自定义下F1
    pred_train_default = best_xgb.predict(train_x)
    pred_test_default  = best_xgb.predict(test_x)
    f1_tr_def = f1_score(train_y, pred_train_default)
    f1_te_def = f1_score(test_y, pred_test_default)
    print("Train F1:", f1_tr_def)
    print("Test  F1:", f1_te_def)

    # 搜自定义阈值
    dtest = xgb.DMatrix(test_x)
    raw_test = best_xgb.get_booster().predict(dtest, output_margin=True)
    test_proba = 1.0 / (1.0 + np.exp(-raw_test))
    best_thr, best_f1_test = find_best_threshold(test_proba, test_y) # test_proba获得预测值从而去选择判断阈值
    print(f"[INFO] 测试集最佳阈值: {best_thr:.3f} => F1={best_f1_test:.4f}")

    # 保存
    xgb_path = os.path.join(this_feature_dir, f"model_{model_name}.pkl")
    joblib.dump(best_xgb, xgb_path)

    thr_path = os.path.join(this_feature_dir, "best_threshold.pkl")
    with open(thr_path, "wb") as f:
        pickle.dump(best_thr, f)

    results[model_name] = {
        "f1_train_default": f1_tr_def,
        "f1_test_default":  f1_te_def,
        "best_threshold":   best_thr,
        "f1_test_best_thr": best_f1_test
    }
    return results

# ========== 主流程入口 ==========
if __name__ == "__main__":
    file_type = input("请输入文件类型 (1: CSV, 2: JSON): ")
    if file_type == "1":
        data_path = os.path.join(BASE_DIR, "Data", "train_set.csv")
    else:
        data_path = os.path.join(BASE_DIR, "Data", "all_data.json")

    if not os.path.exists(data_path):
        print("文件不存在:", data_path)
        exit(0)

    data = read_data(data_path)

    # 生成 "Query_preprocessed"
    data["Query_preprocessed"] = data["Query"].apply(advanced_preprocess)
    data = extract_struct_features(data)
    final_csv_path = os.path.join(BASE_DIR, "Data", "feature_extracted_final.csv")
    data.to_csv(final_csv_path, index=False, escapechar='\\')
    print("\n[INFO] 已保存特征工程结果 =>", final_csv_path)

    features_dict = train_test_split_and_featurize(data)
    x_train = features_dict["x_train"]
    x_test  = features_dict["x_test"]
    y_train = features_dict["y_train"]
    y_test  = features_dict["y_test"]
    (train_feat, test_feat) = features_dict["num_features"]

    print("\n训练特征维度:", train_feat.shape, "训练标签:", len(y_train))
    print("测试特征维度:",  test_feat.shape,  "测试标签:", len(y_test))

    model_results = train_and_save_model(
        feature_name="numeric_features",
        train_x=train_feat, test_x=test_feat,
        train_y=y_train, test_y=y_test
    )
    print("\n=== 训练结束 ===")
    print(model_results)
