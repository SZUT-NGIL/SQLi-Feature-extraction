import warnings, re, os, sys, joblib, chardet
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import csv

# CSV 文件路径
csv_file = "results/csv/results_raw.csv"
SAVE_PLOTS = os.environ.get("SQLI_SKIP_PLOTS") != "1"

# 如果文件不存在，写入表头
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')  # 逗号分隔
        writer.writerow(["Method", "ACC", "PREC", "TPR", "F1", "AUC", "TP", "TN", "FP", "FN"])

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt

# 与训练脚本保持一致
BASE_DIR  = "fe/bow"
MODEL_DIR = os.path.join(BASE_DIR, "model", "1")

# BoW 向量器路径（在训练脚本里保存的）
VEC_PATH = os.path.join(MODEL_DIR, "bow_vectorizer.pkl")

# 各模型路径
MODEL_PATHS = {
    "XGB": os.path.join(MODEL_DIR, "model_XGB.pkl"),
}

# 与训练脚本一致的预处理
def preprocess(txt: str) -> str:
    return str(txt).lower().strip()

def read_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".json"):
        df = pd.read_json(path)
    else:
        try:
            df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                enc = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")

    if {"Query", "Label"} - set(df.columns):
        raise ValueError("需包含 Query, Label 列")

    df["Query"] = df["Query"].fillna("").map(preprocess)

    # Label 归一化为 0/1，逻辑与训练脚本保持一致
    if not np.issubdtype(df["Label"].dtype, np.integer):
        df["Label"] = df["Label"].apply(lambda x: 1 if str(x).lower() == "attack" else 0)

    return df

def evaluate(df: pd.DataFrame, threshold: float = 0.5):
    # 加载 BoW 向量器
    if not os.path.exists(VEC_PATH):
        raise FileNotFoundError(f"找不到 BoW 向量器: {VEC_PATH}")
    vectorizer = joblib.load(VEC_PATH)

    # 将文本转为 BoW 特征
    X = vectorizer.transform(df["Query"].values)
    y = df["Label"].values

    # 用于保存各模型的 ROC 曲线数据
    roc_curves = {}  # name -> (fpr, tpr, auc)

    for name, pth in MODEL_PATHS.items():
        if not os.path.exists(pth):
            print(f"[WARN] 跳过 {name}, 文件不存在: {pth}")
            continue

        model = joblib.load(pth)

        proba = None
        # 二分类预测概率，取正类概率
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            y_pred = (proba >= threshold).astype(int)
        else:
            # 极端情况：没有 predict_proba 的模型，这里退化为直接预测
            y_pred = model.predict(X)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

        # 计算 AUC & ROC 曲线（只有在有 proba 的情况下）
        if proba is not None:
            auc = roc_auc_score(y, proba)
            fpr, tpr, _ = roc_curve(y, proba)
            roc_curves[name] = (fpr, tpr, auc)
        else:
            auc = None

        # 打印时也把 AUC 打出来
        if auc is not None:
            print(
                f"\n{name}: ACC={acc:.4f}  PREC={prec:.4f}  "
                f"TPR={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
                f"TP={TP} TN={TN} FP={FP} FN={FN}"
            )
        else:
            print(
                f"\n{name}: ACC={acc:.4f}  PREC={prec:.4f}  "
                f"TPR={rec:.4f}  F1={f1:.4f}  AUC=N/A    "
                f"TP={TP} TN={TN} FP={FP} FN={FN}"
            )

        # 追加写入 CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                "BoW", 
                f"{acc:.4f}", 
                f"{prec:.4f}", 
                f"{rec:.4f}", 
                f"{f1:.4f}", 
                f"{auc:.4f}" if auc is not None else "N/A", 
                TP, TN, FP, FN
            ])

    # 统一绘制 ROC 曲线
    if SAVE_PLOTS and roc_curves:
        plt.figure(figsize=(6, 6))
        for name, (fpr, tpr, auc) in roc_curves.items():
           plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

        # 画对角线（随机分类器）
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (BoW)")
        plt.legend(loc="lower right")
        plt.grid(True)

        out_path = os.path.join(MODEL_DIR, "bow_roc_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[INFO] ROC 曲线已保存到: {out_path}")
    elif roc_curves:
        print("\n[INFO] 已跳过 ROC 曲线保存 (SQLI_SKIP_PLOTS=1)")
    else:
        print("\n[WARN] 没有任何模型提供 predict_proba，无法绘制 ROC 曲线。")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        test_path = sys.argv[1]
    else:
        test_path = input("输入测试文件路径: ").strip()

    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)

    df_test = read_data(test_path)
    evaluate(df_test)
