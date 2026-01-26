import pandas as pd

# 读取原始 CSV
df = pd.read_csv("results/csv/results_raw.csv")

# 定义函数计算新增指标
def calculate_metrics(row):
    TP = row['TP']
    TN = row['TN']
    FP = row['FP']
    FN = row['FN']

    # FNR = FN / (FN + TP)
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

    # FPR = FP / (FP + TN)
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

    return pd.Series([FNR, FPR])

# 计算新增指标
df[['FNR', 'FPR']] = df.apply(calculate_metrics, axis=1)

# 中文模型 CSV
metrics = ['ACC','PREC','TPR','F1','FNR','FPR','AUC']
methods = ['BoW','TF-IDF','Word2Vec','FastText','Bert','Ours']
df = df[df['Method'].isin(methods)][['Method'] + metrics]
df.to_csv('results/csv/results.csv', index=False, encoding='utf-8')
print("已生成 results.csv")

