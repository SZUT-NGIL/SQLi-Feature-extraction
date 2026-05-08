import warnings, re, os, pickle, joblib, chardet, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score

from xgboost  import XGBClassifier

# ---------- 公共变量 ----------
BASE_DIR      = "fe/tfidf"
DATA_DIR      = "Data"
MODEL_DIR     = os.path.join(BASE_DIR, "model", "1")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE  = 42
CV_FOLDS      = 3
N_ITER_RS     = 200

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

# ---------- 随机搜索空间 ----------
param_xgb = {
    "max_depth":      [7],
    "learning_rate":  [0.2],
    "n_estimators":   [400],
    "gamma":          [0],
    "reg_lambda":     [3]
}

# ---------- 训练四函数 ----------
def train_models(Xtr, ytr, Xte, yte):
    results, models = {}, {}

    rs_xgb = RandomizedSearchCV(
        estimator=XGBClassifier(disable_default_eval_metric=True),
        param_distributions=param_xgb,
        n_iter=N_ITER_RS,
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    rs_xgb.fit(Xtr, ytr)
    mdl_xgb = rs_xgb.best_estimator_
    mdl_xgb.fit(Xtr, ytr)
    models["XGB"] = mdl_xgb
    results["XGB"] = f1_score(yte, mdl_xgb.predict(Xte))

    return models, results

# ---------- 主流程 ----------
if __name__ == "__main__":
    data_path = input("训练文件 (CSV/JSON): ").strip()
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    df = read_data(data_path)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df["Query"], df["Label"].values,
        test_size=0.3, stratify=df["Label"], random_state=RANDOM_STATE
    )

    # 拟合 TF-IDF（仅用训练集，避免泄露）
    tfidf = TfidfVectorizer(tokenizer=tokenize,
                            ngram_range=(1,2),
                            min_df=2,
                            max_features=50000,
                            lowercase=False)
    X_train = tfidf.fit_transform(X_train_df.tolist())
    X_test  = tfidf.transform(X_test_df.tolist())
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    models, f1_res = train_models(X_train, y_train, X_test, y_test)

    for name, mdl in models.items():
        joblib.dump(mdl, os.path.join(MODEL_DIR, f"model_{name}.pkl"))
    with open(os.path.join(MODEL_DIR, "train_metrics.pkl"), "wb") as f:
        pickle.dump(f1_res, f)

    print("\n====== TF-IDF 训练完成 ======")
    for k, v in f1_res.items():
        print(f"{k}: Test-F1 = {v:.4f}")
