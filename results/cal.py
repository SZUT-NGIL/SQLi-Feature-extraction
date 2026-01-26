import pandas as pd
import math

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

    # TNR = TN / (TN + FP)
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0

    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    denominator = math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    MCC = ((TP * TN) - (FP * FN)) / denominator if denominator != 0 else 0

    # F2 = (1 + 2^2) * (PREC * TPR) / (4 * PREC + TPR)
    PREC = row['PREC']
    TPR = row['TPR']
    F2 = (1 + 2**2) * (PREC * TPR) / (4 * PREC + TPR) if (4 * PREC + TPR) != 0 else 0

    return pd.Series([FNR, FPR, TNR, MCC, F2])

# 计算新增指标
df[['FNR', 'FPR', 'TNR', 'MCC', 'F2']] = df.apply(calculate_metrics, axis=1)

print(df.columns.tolist())

# 中文模型 CSV
cn_metrics = ['ACC','PREC','TPR','F1','FNR','FPR','AUC']
cn_methods = ['BoW','TF-IDF','Word2Vec','FastText','Bert','Ours']
df_cn = df[df['Method'].isin(cn_methods)][['Method'] + cn_metrics]
df_cn.to_csv('results/csv/results_cn.csv', index=False, encoding='utf-8')
print("已生成 results_cn.csv")
