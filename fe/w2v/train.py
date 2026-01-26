import warnings, re, os, pickle, joblib, chardet
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from xgboost import XGBClassifier

# ---------- 公共变量 ----------
BASE_DIR      = "fe/w2v"
DATA_DIR      = "Data"
MODEL_DIR     = os.path.join(BASE_DIR, "model", "1")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE  = 42
CV_FOLDS      = 3
N_ITER_RS     = 200

W2V_SIZE      = 200
W2V_WINDOW    = 5
W2V_MIN_CNT   = 2
W2V_SG        = 1
W2V_EPOCHS    = 10
W2V_WORKERS   = 4

# ---------- 简易预处理 & 分词 ----------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def preprocess(text: str) -> str:
    """转小写 + 去掉不可见字符"""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

def tokenize(text: str):
    return TOKEN_RE.findall(text)

# ---------- 读取数据 ----------
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
        raise ValueError("数据必须包含列: Query, Label")
    df["Query"] = df["Query"].fillna("").map(preprocess)
    # Label 归一化为 0/1
    if not np.issubdtype(df["Label"].dtype, np.integer):
        df["Label"] = df["Label"].apply(lambda x: 1 if str(x).lower()=="attack" else 0)

    df.drop_duplicates(subset=["Query", "Label"], inplace=True)
    print(f"[INFO] 数据行数: {len(df)}, 标签分布: \n{df['Label'].value_counts()}")
    return df

# ---------- Word2Vec ----------
def build_w2v(token_lists):
    print("[INFO] 训练 Word2Vec...")
    return Word2Vec(
        sentences = token_lists,
        vector_size = W2V_SIZE,
        window = W2V_WINDOW,
        min_count = W2V_MIN_CNT,
        sg = W2V_SG,
        workers = W2V_WORKERS,
        epochs = W2V_EPOCHS,
        seed = RANDOM_STATE
    )

def sent2vec(tokens, model):
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, 0).astype(np.float32)

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

    X_all_tokens = df["Query"].map(tokenize).tolist()
    y_all = df["Label"].values

    df_train, df_test, y_train, y_test = train_test_split(
        df, y_all, test_size=0.3, stratify=y_all, random_state=RANDOM_STATE
    )

    # Word2Vec 仅用训练集语料
    w2v = build_w2v(df_train["Query"].map(tokenize).tolist())
    w2v.save(os.path.join(MODEL_DIR, "word2vec.model"))

    # 句向量
    X_train = np.vstack([sent2vec(t, w2v) for t in df_train["Query"].map(tokenize)])
    X_test  = np.vstack([sent2vec(t, w2v) for t in df_test["Query"].map(tokenize)])

    models, f1_results = train_models(X_train, y_train, X_test, y_test)

    # 保存模型
    for name, mdl in models.items():
        joblib.dump(mdl, os.path.join(MODEL_DIR, f"model_{name}.pkl"))
    with open(os.path.join(MODEL_DIR, "train_metrics.pkl"), "wb") as f:
        pickle.dump(f1_results, f)

    print("\n====== 训练完成 ======")
    for k, v in f1_results.items():
        print(f"{k}  Test F1 = {v:.4f}")