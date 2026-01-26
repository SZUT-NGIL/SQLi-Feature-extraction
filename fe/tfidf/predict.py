# -*- coding: utf-8 -*-
import warnings, re, os, sys, pickle, joblib, chardet, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer   # 仅为类型提示

import matplotlib.pyplot as plt
import os
import csv

# CSV 文件路径
csv_file = "results/csv/results_raw.csv"

# 如果文件不存在，写入表头
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')  # 逗号分隔
        writer.writerow(["Method", "ACC", "PREC", "TPR", "F1", "AUC", "TP", "TN", "FP", "FN"])

BASE_DIR  = "fe/tfidf"
MODEL_DIR = os.path.join(BASE_DIR, "model", "1")
VEC_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

MODEL_PTH = {
    "XGB": os.path.join(MODEL_DIR, "model_XGB.pkl"),
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
def preprocess(t): return str(t).lower().strip()
def tokenizer(t): return TOKEN_RE.findall(t)

def read_data(p):  # 与训练同规则
    if p.lower().endswith(".json"):
        df = pd.read_json(p)
    else:
        try:
            df = pd.read_csv(p, encoding="latin1", on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(p, "rb") as f:
                enc = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(p, encoding=enc, on_bad_lines="skip")

    if {"Query","Label"} - set(df.columns):
        raise ValueError("需包含 Query, Label 列")

    df["Query"] = df["Query"].fillna("").map(preprocess)

    if not np.issubdtype(df["Label"].dtype, np.integer):
        df["Label"] = df["Label"].apply(lambda x: 1 if str(x).lower()=="attack" else 0)

    return df

def evaluate(df, thr=0.5):
    # 载入 TF-IDF 向量器
    with open(VEC_PATH, "rb") as f:
        vec: TfidfVectorizer = pickle.load(f)

    X = vec.transform(df["Query"].tolist())
    y = df["Label"].values

    X_dense = X.toarray().astype(np.float32)

    # 用于保存各模型的 ROC 曲线数据
    roc_curves = {}   # name -> (fpr, tpr, auc)

    for n, pth in MODEL_PTH.items():
        if not os.path.exists(pth):
            print(f"[WARN] 跳过 {n}, 文件不存在: {pth}")
            continue

        mdl = joblib.load(pth)

        proba = None

        # ---- 预测概率（如果支持）----
        if hasattr(mdl, "predict_proba"):
            if n == "Cat":
                proba = mdl.predict_proba(X_dense)[:, 1]
            else:                   # 其他模型直接用 sparse
                proba = mdl.predict_proba(X)[:, 1]

            y_pred = (proba >= thr).astype(int)
        else:
            # 不支持 predict_proba 的兜底
            if n == "Cat":
                y_pred = mdl.predict(X_dense)
            else:
                y_pred = mdl.predict(X)

        # ---- 基本指标 ----
        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()

        # ---- AUC & ROC ----
        if proba is not None:
            auc = roc_auc_score(y, proba)
            fpr, tpr, _ = roc_curve(y, proba)
            roc_curves[n] = (fpr, tpr, auc)
        else:
            auc = None

        # ---- 打印结果 ----
        if auc is not None:
            print(
                f"\n{n}: ACC={acc:.4f} PREC={prec:.4f} TPR={rec:.4f} F1={f1:.4f} "
                f"AUC={auc:.4f}  TP={TP} TN={TN} FP={FP} FN={FN}"
            )
        else:
            print(
                f"\n{n}: ACC={acc:.4f} PREC={prec:.4f} TPR={rec:.4f} F1={f1:.4f} "
                f"AUC=N/A   TP={TP} TN={TN} FP={FP} FN={FN}"
            )

        # 追加写入 CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                "TF-IDF", 
                f"{acc:.4f}", 
                f"{prec:.4f}", 
                f"{rec:.4f}", 
                f"{f1:.4f}", 
                f"{auc:.4f}" if auc is not None else "N/A", 
                TP, TN, FP, FN
            ])
    
    # ---- 统一绘制 ROC 曲线 ----
    if roc_curves:
        plt.figure(figsize=(6, 6))
        for n, (fpr, tpr, auc) in roc_curves.items():
            plt.plot(fpr, tpr, label=f"{n} (AUC={auc:.4f})")

        # 随机基线
        plt.plot([0, 1], [0, 1], "k--", label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (TF-IDF)")
        plt.legend(loc="lower right")
        plt.grid(True)

        out_path = os.path.join(MODEL_DIR, "tfidf_roc_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[INFO] ROC 曲线已保存到: {out_path}")
    else:
        print("\n[WARN] 没有任何模型提供 predict_proba，无法绘制 ROC 曲线。")

if __name__ == "__main__":
    test_path = sys.argv[1] if len(sys.argv)>=2 else input("测试文件: ").strip()
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)
    evaluate(read_data(test_path))