# fe/utils.py
import os
import re
import csv
import pickle
import joblib
import chardet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV # <--- 确保引入 GridSearchCV

# ---------- 全局常量 ----------
RANDOM_STATE = 42
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CSV_RESULT_FILE = "results/csv/results_raw.csv"

# XGBoost 搜索空间
PARAM_XGB = {
    "max_depth": [7],
    "learning_rate": [0.2],
    "n_estimators": [400],
    "gamma": [0],
    "reg_lambda": [3]
}

# ---------- 文本处理 ----------
def preprocess_text(text: str) -> str:
    """基础预处理：转小写 + 去除首尾空格"""
    return str(text).lower().strip()

def tokenize_text(text: str):
    """基础分词：正则提取"""
    return TOKEN_RE.findall(text)

# ---------- IO 操作 ----------
def read_data(path: str) -> pd.DataFrame:
    """通用的数据读取函数，处理 CSV/JSON 和编码问题"""
    if path.lower().endswith(".json"):
        df = pd.read_json(path)
    else:
        try:
            df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                enc = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")

    required_cols = {"Query", "Label"}
    if required_cols - set(df.columns):
        raise ValueError(f"数据必须包含列: {required_cols}")

    # 预处理 Query
    df["Query"] = df["Query"].fillna("").map(preprocess_text)

    # 标签归一化 (Attack -> 1, Normal -> 0)
    if not np.issubdtype(df["Label"].dtype, np.integer):
        df["Label"] = df["Label"].apply(lambda x: 1 if str(x).lower() == "attack" else 0)

    # 去重
    before_len = len(df)
    df.drop_duplicates(subset=["Query", "Label"], inplace=True)
    if len(df) < before_len:
        print(f"[INFO] 移除重复行: {before_len - len(df)}")

    return df

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------- 训练流程 ----------
def train_xgb_model(X_train, y_train, X_test, y_test, n_iter=200, cv=3):
    """
    统一的 XGBoost 训练与评估
    自动判断参数空间大小，选择 GridSearch 或 RandomizedSearch
    """
    print("[INFO] 开始训练 XGBoost...")
    xgb_base = XGBClassifier(disable_default_eval_metric=True)
    
    # 1. 计算参数空间总大小
    total_combinations = 1
    for params in PARAM_XGB.values():
        total_combinations *= len(params)
    
    # 2. 根据空间大小选择搜索策略
    # 如果组合数很少(<= n_iter)，直接跑完所有组合(GridSearch)，避免 RandomizedSearch 的警告
    if total_combinations <= n_iter:
        print(f"[INFO] 参数组合较少 ({total_combinations} 种)，使用 GridSearchCV 全量搜索...")
        search_engine = GridSearchCV(
            xgb_base, 
            PARAM_XGB, 
            scoring="f1",
            cv=cv, 
            n_jobs=-1
        )
    else:
        print(f"[INFO] 参数组合较多，使用 RandomizedSearchCV 随机搜索 {n_iter} 次...")
        search_engine = RandomizedSearchCV(
            xgb_base, 
            PARAM_XGB, 
            n_iter=n_iter, 
            scoring="f1",
            cv=cv, 
            n_jobs=-1, 
            random_state=RANDOM_STATE
        )
    
    # 3. 训练
    search_engine.fit(X_train, y_train)
    best_model = search_engine.best_estimator_
    
    # 如果需要，这里可以再 fit 一次 (GridSearchCV 默认 refit=True，其实已经 fit 好了)
    # best_model.fit(X_train, y_train) 
    
    # 4. 评估
    test_f1 = f1_score(y_test, best_model.predict(X_test))
    return best_model, test_f1

# ---------- 评估与绘图流程 ----------
def evaluate_models(X, y, model_paths, method_name, output_dir, threshold=0.5):
    """
    统一评估逻辑：计算指标 -> 打印 -> 写入CSV -> 画ROC图
    """
    # 准备 CSV 表头
    if not os.path.exists(CSV_RESULT_FILE):
        os.makedirs(os.path.dirname(CSV_RESULT_FILE), exist_ok=True)
        with open(CSV_RESULT_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Method", "ACC", "PREC", "TPR", "F1", "AUC", "TP", "TN", "FP", "FN"])

    roc_curves = {}

    for name, pth in model_paths.items():
        if not os.path.exists(pth):
            print(f"[WARN] 模型文件不存在: {pth}")
            continue

        model = joblib.load(pth)
        
        # 预测
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # 兼容处理：如果是二分类，取 index 1；如果 output 是一维则直接用
            if proba.ndim > 1 and proba.shape[1] > 1:
                proba = proba[:, 1]
            y_pred = (proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X)

        # 指标计算
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        
        auc_val = "N/A"
        if proba is not None:
            auc_score_val = roc_auc_score(y, proba)
            auc_val = f"{auc_score_val:.4f}"
            fpr, tpr, _ = roc_curve(y, proba)
            roc_curves[name] = (fpr, tpr, auc_score_val)

        # 打印日志
        print(f"\n{name}: ACC={acc:.4f} PREC={prec:.4f} TPR={rec:.4f} F1={f1:.4f} AUC={auc_val}")
        print(f"      TP={TP} TN={TN} FP={FP} FN={FN}")

        # 写入 CSV
        with open(CSV_RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                method_name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
                auc_val, TP, TN, FP, FN
            ])

    # 绘制 ROC
    if roc_curves:
        plt.figure(figsize=(6, 6))
        for name, (fpr, tpr, auc) in roc_curves.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
        
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves ({method_name})")
        plt.legend(loc="lower right")
        plt.grid(True)
        
        out_path = os.path.join(output_dir, f"{method_name.lower()}_roc_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] ROC 曲线已保存: {out_path}")
