import warnings, re, os, pickle, joblib, chardet
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from xgboost import XGBClassifier

# ---------- 公共参数 ----------

BASE_DIR      = "fe/bert"
DATA_DIR      = "Data"
MODEL_DIR     = os.path.join(BASE_DIR, "model", "1")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE  = 42
CV_FOLDS      = 3
N_ITER_RS     = 200  # 每模型随机搜索次数 (可调)

# ---------- BERT 参数 ----------
# 1) 公开基础模型名称
BERT_BASE_NAME   = "bert-base-uncased"
BERT_MAX_LEN     = 64                      # 句子最大长度
BERT_BATCH_SIZE  = 32

# 2) 本地训练好BERT 保存目录
BERT_DOMAIN_DIR  = os.path.join(MODEL_DIR, "bert_domain")  # 存在时使用

# ---------- 简易预处理 & 分词 ----------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def preprocess(text: str) -> str:
    """转小写 + 去掉不可见字符"""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

def tokenize(text: str):
    # BERT 实际上不需要这个分词，但保留以保持接口一致
    return TOKEN_RE.findall(text)

# ---------- 读取数据 ----------
def read_data(path: str) -> pd.DataFrame:
    is_json = path.lower().endswith(".json")
    if is_json:
        df = pd.read_json(path)
    else:
        try:
            df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                enc = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")

    if {"Query", "Label"} - set(df.columns):
        raise ValueError("数据必须包含列: Query, Label")
    df["Query"] = df["Query"].fillna("").map(preprocess)
    # Label 归一化为 0/1
    if not np.issubdtype(df["Label"].dtype, np.integer):
        df["Label"] = df["Label"].apply(lambda x: 1 if str(x).lower()=="attack" else 0)

    df.drop_duplicates(subset=["Query", "Label"], inplace=True)
    print(f"[INFO] 数据行数: {len(df)}, 标签分布: \n{df['Label'].value_counts()}")
    return df

# ---------- BERT 特征提取 ----------
def load_bert():
    """
    加载 BERT 分词器和模型，并放到对应设备上。
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("[INFO] Using device:", device)


    if os.path.isdir(BERT_DOMAIN_DIR) and os.listdir(BERT_DOMAIN_DIR):
        print(f"[INFO] 检测到本地领域 BERT，使用: {BERT_DOMAIN_DIR}")
        tokenizer = BertTokenizer.from_pretrained(BERT_DOMAIN_DIR)
        model = BertModel.from_pretrained(BERT_DOMAIN_DIR)
    else:
        print(f"[WARN] 未检测到领域 BERT 目录 {BERT_DOMAIN_DIR}，改用基础模型: {BERT_BASE_NAME}")
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

    # 按 batch 编码，防止一次性占用太多显存/内存
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
            # 取 [CLS] 向量作为句向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)

# ---------- 随机搜索空间 ----------
param_xgb = {
    "max_depth":      [7],
    "learning_rate":  [0.2],
    "n_estimators":   [400],
    "gamma":          [0],
    "reg_lambda":     [3]
}

# ---------- 训练函数 ----------
def train_models(X_tr, y_tr, X_te, y_te):
    results, models = {}, {}

    xgb_base = XGBClassifier(disable_default_eval_metric=True)
    rs_xgb = RandomizedSearchCV(
        xgb_base, param_xgb, n_iter=N_ITER_RS, scoring="f1",
        cv=CV_FOLDS, n_jobs=-1
    ).fit(X_tr, y_tr)
    mdl_xgb = rs_xgb.best_estimator_.fit(X_tr, y_tr)
    models["XGB"] = mdl_xgb
    results["XGB"] = f1_score(y_te, mdl_xgb.predict(X_te))

    return models, results

# ---------- 主流程 ----------
if __name__ == "__main__":
    data_path = input("输入训练文件路径 (CSV/JSON): ").strip()
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    df = read_data(data_path)

    X_all_texts = df["Query"].values
    y_all = df["Label"].values

    # 注意：这里保持和原来一样，对 df 本身划分
    df_train, df_test, y_train, y_test = train_test_split(
        df, y_all, test_size=0.3, stratify=y_all, random_state=RANDOM_STATE
    )

    tokenizer, bert_model, device = load_bert()

    # BERT 句向量（用原来的 df_train/df_test["Query"]）
    print("[INFO] 使用 BERT 提取训练集特征...")
    X_train = bert_encode(df_train["Query"].tolist(), tokenizer, bert_model, device)

    print("[INFO] 使用 BERT 提取测试集特征...")
    X_test  = bert_encode(df_test["Query"].tolist(), tokenizer, bert_model, device)

    # 训练 XGBoost 模型
    models, f1_results = train_models(X_train, y_train, X_test, y_test)

    # 保存模型
    for name, mdl in models.items():
        joblib.dump(mdl, os.path.join(MODEL_DIR, f"model_{name}.pkl"))

    # 保存训练指标
    with open(os.path.join(MODEL_DIR, "train_metrics.pkl"), "wb") as f:
        pickle.dump(f1_results, f)

    print("\n====== BERT 特征训练完成 ======")
    for k, v in f1_results.items():
        print(f"{k}  Test F1 = {v:.4f}")