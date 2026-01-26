# -*- coding: utf-8 -*-
import warnings, re, os, sys, joblib, chardet, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
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
import os
import csv

# CSV 文件路径
csv_file = "results/csv/results_raw.csv"

# 如果文件不存在，写入表头
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')  # 逗号分隔
        writer.writerow(["Method", "ACC", "PREC", "TPR", "F1", "AUC", "TP", "TN", "FP", "FN"])

BASE_DIR  = "fe/w2v"
MODEL_DIR = os.path.join(BASE_DIR, "model", "1")
W2V_PATH  = os.path.join(MODEL_DIR, "word2vec.model")

MODEL_PATHS = {
    "XGB": os.path.join(MODEL_DIR, "model_XGB.pkl"),
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def preprocess(txt):
    return str(txt).lower().strip()

def tokenize(txt):
    return TOKEN_RE.findall(preprocess(txt))

def sent2vec(tok, w2v):
    if not tok:
        return np.zeros(w2v.vector_size, np.float32)
    vecs = [w2v.wv[t] for t in tok if t in w2v.wv]
    return np.mean(vecs, 0).astype(np.float32) if vecs else np.zeros(w2v.vector_size, np.float32)

def read_data(path):
    if path.lower().endswith(".json"):
        df = pd.read_json(path)
    else:
        try:
            df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                enc = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")

    if {"Query","Label"} - set(df.columns):
        raise ValueError("需包含 Query, Label 列")

    df["Query"] = df["Query"].fillna("").map(preprocess)

    if not np.issubdtype(df["Label"].dtype, np.integer):
        df["Label"] = df["Label"].apply(lambda x: 1 if str(x).lower()=="attack" else 0)

    return df

def evaluate(df, threshold=0.5):
    # 加载 Word2Vec
    if not os.path.exists(W2V_PATH):
        raise FileNotFoundError(f"[ERR] 未找到 Word2Vec 模型: {W2V_PATH}")

    w2v = Word2Vec.load(W2V_PATH)

    # 句向量
    X = np.vstack([sent2vec(tokenize(q), w2v) for q in df["Query"]])
    y = df["Label"].values

    # 用来保存各模型 ROC 曲线数据
    roc_curves = {}   # name -> (fpr, tpr, auc)

    for name, pth in MODEL_PATHS.items():
        if not os.path.exists(pth):
            print(f"[WARN] 跳过 {name}, 文件不存在: {pth}")
            continue

        model = joblib.load(pth)

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            y_pred = (proba >= threshold).astype(int)
        else:
            # 极端情况兜底
            y_pred = model.predict(X)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()

        # 算 AUC & ROC
        if proba is not None:
            auc = roc_auc_score(y, proba)
            fpr, tpr, _ = roc_curve(y, proba)
            roc_curves[name] = (fpr, tpr, auc)
        else:
            auc = None

        # 印时带上 AUC
        if auc is not None:
            print(
                f"\n{name}: ACC={acc:.4f}  PREC={prec:.4f}  TPR={rec:.4f}  F1={f1:.4f}  "
                f"AUC={auc:.4f}  TP={TP} TN={TN} FP={FP} FN={FN}"
            )
        else:
            print(
                f"\n{name}: ACC={acc:.4f}  PREC={prec:.4f}  TPR={rec:.4f}  F1={f1:.4f}  "
                f"AUC=N/A    TP={TP} TN={TN} FP={FP} FN={FN}"
            )

        # 追加写入 CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                "Word2Vec", 
                f"{acc:.4f}", 
                f"{prec:.4f}", 
                f"{rec:.4f}", 
                f"{f1:.4f}", 
                f"{auc:.4f}" if auc is not None else "N/A", 
                TP, TN, FP, FN
            ])

    # 统一绘制 ROC 曲线
    if roc_curves:
        plt.figure(figsize=(6, 6))
        for name, (fpr, tpr, auc) in roc_curves.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

        # 随机基线
        plt.plot([0, 1], [0, 1], "k--", label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (Word2Vec)")
        plt.legend(loc="lower right")
        plt.grid(True)

        out_path = os.path.join(MODEL_DIR, "w2v_roc_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[INFO] ROC 曲线已保存到: {out_path}")
    else:
        print("\n[WARN] 没有任何模型提供 predict_proba，无法绘制 ROC 曲线。")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        test_path = sys.argv[1]
    else:
        test_path = input("输入测试文件路径: ").strip()
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)
    evaluate(read_data(test_path))