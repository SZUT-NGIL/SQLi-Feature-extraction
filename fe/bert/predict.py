import warnings, re, os, sys, joblib, chardet
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from transformers import BertTokenizer, BertModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt
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

# ---------- 公共参数（与训练脚本保持一致） ----------
BASE_DIR  = "fe/bert"
MODEL_DIR     = os.path.join(BASE_DIR, "model", "1")

BERT_BASE_NAME   = "bert-base-uncased"
BERT_DOMAIN_DIR  = os.path.join(MODEL_DIR, "bert_domain")
BERT_MAX_LEN     = 64
BERT_BATCH_SIZE  = 32

# XGBoost
MODEL_PATHS = {
    "XGB": os.path.join(MODEL_DIR, "model_XGB.pkl"),
}

# ---------- 预处理 ----------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

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

# ---------- BERT 加载 & 编码 ----------
def load_bert():
    """
    加载模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.isdir(BERT_DOMAIN_DIR) and os.listdir(BERT_DOMAIN_DIR):
        print(f"[INFO] 使用领域版 BERT: {BERT_DOMAIN_DIR}")
        tokenizer = BertTokenizer.from_pretrained(BERT_DOMAIN_DIR)
        model = BertModel.from_pretrained(BERT_DOMAIN_DIR)
    else:
        print(f"[WARN] 未检测到领域 BERT，改用基础模型: {BERT_BASE_NAME}")
        tokenizer = BertTokenizer.from_pretrained(BERT_BASE_NAME)
        model = BertModel.from_pretrained(BERT_BASE_NAME)

    model.to(device)
    model.eval()
    return tokenizer, model, device

def bert_encode(texts, tokenizer, model, device,
                batch_size=BERT_BATCH_SIZE, max_len=BERT_MAX_LEN):
    """
    将若干句子编码为 BERT 句向量（CLS 向量）
    texts: List[str]
    return: np.ndarray, shape = (len(texts), hidden_size)
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 取 [CLS] 向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)

# ---------- 评估 ----------
def evaluate(df: pd.DataFrame, threshold: float = 0.5):
    # 加载 BERT
    tokenizer, bert_model, device = load_bert()

    # 提取 BERT 特征
    print("[INFO] 使用 BERT 提取特征用于评估...")
    X = bert_encode(df["Query"].tolist(), tokenizer, bert_model, device)
    y = df["Label"].values

    # 用来保存各模型的 ROC 曲线数据
    roc_curves = {}  # name -> (fpr, tpr, auc)

    for name, pth in MODEL_PATHS.items():
        if not os.path.exists(pth):
            print(f"[WARN] 跳过 {name}, 模型文件不存在: {pth}")
            continue

        model = joblib.load(pth)

        # 优先使用 predict_proba
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            y_pred = (proba >= threshold).astype(int)
        else:
            # 回退：某些模型可能没有 predict_proba
            y_pred = model.predict(X)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

        # ---- 计算 AUC & ROC----
        if proba is not None:
            auc = roc_auc_score(y, proba)
            fpr, tpr, _ = roc_curve(y, proba)
            roc_curves[name] = (fpr, tpr, auc)
        else:
            auc = None

        # 打印指标
        if auc is not None:
            print(
                f"\n{name}: "
                f"ACC={acc:.4f}  PREC={prec:.4f}  TPR={rec:.4f}  F1={f1:.4f}  "
                f"AUC={auc:.4f}  "
                f"TP={TP} TN={TN} FP={FP} FN={FN}"
            )
        else:
            print(
                f"\n{name}: "
                f"ACC={acc:.4f}  PREC={prec:.4f}  TPR={rec:.4f}  F1={f1:.4f}  "
                f"AUC=N/A   "
                f"TP={TP} TN={TN} FP={FP} FN={FN}"
            )

        # 追加写入 CSV
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                "BERT",
                f"{acc:.4f}", 
                f"{prec:.4f}", 
                f"{rec:.4f}", 
                f"{f1:.4f}", 
                f"{auc:.4f}" if auc is not None else "N/A", 
                TP, TN, FP, FN
            ])

    # ---- 统一画 ROC 曲线 ----
    if SAVE_PLOTS and roc_curves:
        plt.figure(figsize=(6, 6))
        for name, (fpr, tpr, auc) in roc_curves.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

        # 画对角线
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)

        # 保存图片到当前目录
        out_path = os.path.join(MODEL_DIR, "bert_roc_curves.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[INFO] ROC 曲线已保存到: {out_path}")
    elif roc_curves:
        print("\n[INFO] 已跳过 ROC 曲线保存 (SQLI_SKIP_PLOTS=1)")
    else:
        print("\n[WARN] 没有任何模型提供 predict_proba，无法绘制 ROC 曲线。")

# ---------- 主流程 ----------
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        test_path = sys.argv[1]
    else:
        test_path = input("输入测试文件路径: ").strip()

    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)

    df_test = read_data(test_path)
    evaluate(df_test)
