import warnings, re, os, pickle, joblib, chardet
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer

from xgboost import XGBClassifier

# ---------- 公共参数 ----------

BASE_DIR      = "fe/bow"
DATA_DIR      = "Data"
MODEL_DIR     = os.path.join(BASE_DIR, "model", "1")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE  = 42
CV_FOLDS      = 3
N_ITER_RS     = 200  # 每模型随机搜索次数 (可调)

# BoW 相关参数
BOW_MAX_FEATURES = 5000   # 词表大小上限，可按需要调整
BOW_NGRAM_RANGE  = (1, 1) # 使用 uni-gram，可改为 (1,2) 做 n-gram

# ---------- 简易预处理 & 分词 ----------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def preprocess(text: str) -> str:
    """转小写 + 去掉不可见字符"""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()

def tokenize(text: str):
    # 如果后面 CountVectorizer 使用自定义 tokenizer，这个函数仍然可以直接复用
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

# ---------- BoW 特征 ----------
def build_bow_vectorizer(train_texts):
    """
    拟合 BoW 向量器并返回：
    - vectorizer: CountVectorizer 实例
    - X_train: 训练集 BoW 特征 (稀疏矩阵)
    """
    print("[INFO] 训练 BoW（Bag-of-Words）特征...")
    vectorizer = CountVectorizer(
        max_features=BOW_MAX_FEATURES,
        ngram_range=BOW_NGRAM_RANGE,
        token_pattern=r"[A-Za-z0-9_]+",  # 与 TOKEN_RE 保持一致
        lowercase=True                   # 已经在 preprocess 中转小写，这里再转一次也无妨
        # 也可以使用自定义 tokenizer：
        # tokenizer=tokenize,
        # preprocessor=preprocess,
    )
    X_train = vectorizer.fit_transform(train_texts)
    print(f"[INFO] 词表大小: {len(vectorizer.vocabulary_)}")
    return vectorizer, X_train

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

    # 直接对文本做划分
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X_all_texts, y_all,
        test_size=0.3,
        stratify=y_all,
        random_state=RANDOM_STATE
    )

    # 只用训练集语料拟合 BoW
    vectorizer, X_train = build_bow_vectorizer(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    # 训练模型
    models, f1_results = train_models(X_train, y_train, X_test, y_test)

    # 保存模型和向量器
    for name, mdl in models.items():
        joblib.dump(mdl, os.path.join(MODEL_DIR, f"model_{name}.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "bow_vectorizer.pkl"))

    with open(os.path.join(MODEL_DIR, "train_metrics.pkl"), "wb") as f:
        pickle.dump(f1_results, f)

    print("\n====== 训练完成 ======")
    for k, v in f1_results.items():
        print(f"{k}  Test F1 = {v:.4f}")