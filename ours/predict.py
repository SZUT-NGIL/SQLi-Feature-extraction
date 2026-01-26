import os
import re
import sys
import pickle
import joblib
import chardet
import xgboost as xgb
import numpy as np
import pandas as pd
from utils.hdcan import *
from utils.hfes import *
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
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

# ========== 配置目录，与训练脚本保持一致 ==========

# 当前文件路径
current_file = Path(__file__).resolve()
# 项目根目录
BASE_DIR = current_file.parents[1]

MODEL_DIR = os.path.join(BASE_DIR, "ours", "model", "1")
NUMERIC_FEATURES_DIR = os.path.join(MODEL_DIR, "numeric_features")

MODEL_FILENAME     = "model_FSHBoost.pkl"
SCALER_FILENAME    = "scaler_for_numeric.pkl"
THRESHOLD_FILENAME = "best_threshold.pkl"

CM_SAVE_PATH   = os.path.join(NUMERIC_FEATURES_DIR, "confusion_matrix.png")
PREDICTION_CSV = os.path.join(NUMERIC_FEATURES_DIR, "predictions.csv")

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p,eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1.0-p))

def plot_metrics_single(metrics_dict):

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'figure.figsize': (8, 6),
        'savefig.bbox': 'tight'
    })
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots()
    
    # 指标标签
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_labels))  # 指标位置
    values = [
        metrics_dict['acc'],
        metrics_dict['prec'],
        metrics_dict['rec'],
        metrics_dict['f1']
    ]
    
    # 创建颜色映射 (从蓝色到绿色)
    colors = plt.cm.Blues(np.linspace(0.6, 1, len(metrics_labels)))
    
    # 绘制条形图
    bars = ax.bar(x, values, width=0.7, color=colors, edgecolor='black', linewidth=0.7)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10,
                    fontweight='bold')
    
    # 设置标题和标签
    ax.set_title('Classification Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0, 1.15)  # 留出空间给标签
    
    
    # 添加脚注
    plt.figtext(0.5, 0.01, 
                "Model Performance Metrics", 
                ha="center", fontsize=10, style='italic')
    
    # 展示
    plt.tight_layout()
    # plt.show()

# ========== 绘制并保存混淆矩阵 ==========
def save_confusion_matrix(true_y, pred_y, save_path, title_suffix=""):
    cm = confusion_matrix(true_y, pred_y)
    recall_m = (cm.T / cm.sum(axis=1)).T
    precision_m = (cm / cm.sum(axis=0))

    labels = [0, 1]
    plt.figure(figsize=(18, 5))
    cmap = sns.light_palette("blue", as_cmap=True)

    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix {title_suffix}")

    # Precision Matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(precision_m, annot=True, cmap=cmap, fmt=".3f",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Precision Matrix {title_suffix}")

    # Recall Matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(recall_m, annot=True, cmap=cmap, fmt=".3f",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Recall Matrix {title_suffix}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def compute_plot_auc(y_true, y_score, model_name="Model", plot_path=None):
    """
    计算 AUC 并绘制 ROC 曲线。
    """
    auc_value = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.4f})", linewidth=2)
    plt.plot([0,1],[0,1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] ROC 曲线已保存到: {plot_path}")
    else:
        plt.show()
    
    return auc_value, fpr, tpr

# ========== 加载模型等 ==========
def load_artifacts():
    scaler_path   = os.path.join(MODEL_DIR, SCALER_FILENAME)
    model_path    = os.path.join(NUMERIC_FEATURES_DIR, MODEL_FILENAME)
    thr_path      = os.path.join(NUMERIC_FEATURES_DIR, THRESHOLD_FILENAME)

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("[Error] 缺少数值特征 Scaler: "+scaler_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError("[Error] 缺少模型文件: "+model_path)
    if not os.path.exists(thr_path):
        raise FileNotFoundError("[Error] 缺少最佳阈值: "+thr_path)

    with open(scaler_path, "rb") as fs:
        numeric_scaler = pickle.load(fs)
    with open(thr_path, "rb") as ft:
        best_threshold = pickle.load(ft)
    loaded_model = joblib.load(model_path)

    print("[OK] 成功加载模型,向量器,scaler 和最佳阈值.")
    return numeric_scaler, loaded_model, best_threshold

# ========== 4. 主评估函数: 可读CSV或JSON ==========
def read_data_for_predict(data_path):
    file_lower = data_path.lower()
    is_json = file_lower.endswith(".json")

    if is_json:
        print(f"[INFO] 读取 JSON: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            df = pd.read_json(f)
    else:
        print(f"[INFO] 读取 CSV: {data_path}")
        try:
            df = pd.read_csv(data_path, encoding='latin1', on_bad_lines='skip')
        except UnicodeDecodeError:
            with open(data_path, 'rb') as f:
                enc = chardet.detect(f.read())["encoding"]
            df = pd.read_csv(data_path, encoding=enc, on_bad_lines='skip')

    if "Query" not in df.columns or "Label" not in df.columns:
        raise ValueError("[Error] 数据中必须包含 Query 和 Label 两列.")
    return df

def evaluate_on_testset(data_path):
    df_test = read_data_for_predict(data_path)
    print(f"[INFO] 测试数据行数: {df_test.shape[0]}")

    # 如果Label不是0/1可做映射
    if df_test["Label"].dtype not in [np.int64, np.float64]:
        df_test["Label"] = df_test["Label"].apply(lambda x: 1 if str(x).lower()=="attack" else 0)

    numeric_scaler, loaded_model, best_thr = load_artifacts()

    test_queries = df_test["Query"].astype(str).tolist()
    test_labels  = df_test["Label"].values

    feats_all = []
    valid_idx = []

    for i, raw_q in enumerate(test_queries):
        # 结构特征
        feats = extract_struct_features_single(raw_q)
        feats_all.append(feats)
        valid_idx.append(i)

    if len(valid_idx)==0:
        print("[Warning] 全部Query无效, 终止.")
        return

    arr_struct = np.array(feats_all, dtype=np.float64)

    # 数值特征标准化
    arr_struct_scaled = numeric_scaler.transform(arr_struct)

    mat_struct = csr_matrix(arr_struct_scaled)
    X_final = mat_struct

    # 替换 A 与 B
    booster = loaded_model.get_booster()
    dtest = xgb.DMatrix(X_final)
    raw_margin = booster.predict(dtest,output_margin=True)
    proba_raw = _sigmoid(raw_margin)

    thr_default = 0.5
    pred_default = (proba_raw >= thr_default).astype(int)
    pred_bestthr = (proba_raw >= best_thr).astype(int)
    margin_bestthr = _logit(best_thr)

    # # (A) 默认阈=0.5
    # pred_default = loaded_model.predict(X_final)

    # # (B) 自定义阈值
    # proba = loaded_model.predict_proba(X_final)[:,1]
    # pred_bestthr = (proba >= best_thr).astype(int)

    # “填回df_test”
    n_total = len(df_test)
    final_pred_default  = np.full(n_total, fill_value=-1, dtype=int)
    final_pred_best_thr = np.full(n_total, fill_value=-1, dtype=int)
    final_raw_margin    = np.full(n_total, fill_value=np.nan, dtype=float)
    final_proba_raw     = np.full(n_total, fill_value=np.nan, dtype=float)

    for i, idx in enumerate(valid_idx):
        final_pred_default[idx]  = pred_default[i]
        final_pred_best_thr[idx] = pred_bestthr[i]
        final_raw_margin[idx]    = raw_margin[i]
        final_proba_raw[idx]     = proba_raw[i]

    mask_valid = (final_pred_default!=-1)
    sub_labels = test_labels[mask_valid]
    sub_def    = final_pred_default[mask_valid]
    sub_best   = final_pred_best_thr[mask_valid]

    print(f"\n[INFO] 有效行数: {len(sub_labels)}/{len(df_test)}")
    if not np.issubdtype(sub_labels.dtype, np.integer):
        sub_labels = sub_labels.astype(int)
    
    valid_label_1 = np.sum(sub_labels == 1)
    valid_label_0 = np.sum(sub_labels == 0)

    print("\n[有效样本统计]")
    print(f"有效测试样例总数: {len(sub_labels)}")
    print(f"标签0的有效样例数: {valid_label_0}")
    print(f"标签1的有效样例数: {valid_label_1}")

    # 评估 (默认阈=0.5)
    acc_def  = accuracy_score(sub_labels, sub_def)
    prec_def = precision_score(sub_labels, sub_def)
    rec_def  = recall_score(sub_labels, sub_def)
    f1_def   = f1_score(sub_labels, sub_def)

    metrics_default = {
        'acc': acc_def,
        'prec': prec_def,
        'rec': rec_def,
        'f1': f1_def
    }

    plot_metrics_single(metrics_dict=metrics_default)

    # 计算混淆矩阵元素和指标
    cm_default = confusion_matrix(sub_labels, sub_def, labels=[0, 1])
    TN_def, FP_def, FN_def, TP_def = cm_default.ravel()
    TPR_def = TP_def / (TP_def + FN_def) if (TP_def + FN_def) != 0 else 0.0
    FPR_def = FP_def / (FP_def + TN_def) if (FP_def + TN_def) != 0 else 0.0
    FNR_def = FN_def / (TP_def + FN_def) if (TP_def + FN_def) != 0 else 0.0

    # 计算默认阈值下的 AUC 并绘制 ROC
    auc_def, fpr_def, tpr_def = compute_plot_auc(
        y_true=sub_labels,
        y_score=proba_raw[mask_valid],  # 使用预测概率
        model_name="FSHBoost_Default0.5",
        plot_path=CM_SAVE_PATH.replace(".png","_default_roc.png")
    )

    print(f"\n[默认阈=0.5 详细指标]")
    print(f"TP: {TP_def}, TN: {TN_def}, FP: {FP_def}, FN: {FN_def}")
    print(f"TPR: {TPR_def:.4f}, FPR: {FPR_def:.4f}, FNR: {FNR_def:.4f}")
    print(f"ACC={acc_def:.4f}, PREC={prec_def:.4f}, TPR={rec_def:.4f}, F1={f1_def:.4f}")
    print(f"[INFO] 默认阈=0.5 的 AUC: {auc_def:.4f}")

    # 追加到 CSV
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([
            "Ours",
            f"{acc_def:.4f}", f"{prec_def:.4f}", f"{rec_def:.4f}", f"{f1_def:.4f}",
            f"{auc_def:.4f}", TP_def, TN_def, FP_def, FN_def
        ])

    save_confusion_matrix(sub_labels, sub_def,
        CM_SAVE_PATH.replace(".png","_default.png"), "(Default=0.5)")

    # 评估 (best_thr)
    acc_b  = accuracy_score(sub_labels, sub_best)
    prec_b = precision_score(sub_labels, sub_best)
    rec_b  = recall_score(sub_labels, sub_best)
    f1_b   = f1_score(sub_labels, sub_best)
    
    # 计算混淆矩阵元素和指标
    cm_best = confusion_matrix(sub_labels, sub_best, labels=[0, 1])
    TN_best, FP_best, FN_best, TP_best = cm_best.ravel()
    TPR_best = TP_best / (TP_best + FN_best) if (TP_best + FN_best) != 0 else 0.0
    FPR_best = FP_best / (FP_best + TN_best) if (FP_best + TN_best) != 0 else 0.0
    FNR_best = FN_best / (TP_best + FN_best) if (TP_best + FN_best) != 0 else 0.0

    print(f"\n[自定义阈={best_thr:.3f} 详细指标]")
    print(f"TP: {TP_best}, TN: {TN_best}, FP: {FP_best}, FN: {FN_best}")
    print(f"TPR: {TPR_best:.4f}, FPR: {FPR_best:.4f}, FNR: {FNR_best:.4f}")
    print(f"ACC={acc_b:.4f}, PREC={prec_b:.4f}, TPR={rec_b:.4f}, F1={f1_b:.4f}")

    save_confusion_matrix(sub_labels, sub_best,
        CM_SAVE_PATH.replace(".png","_bestthr.png"), f"(BestThr={best_thr:.3f})")

    # 写出结果
    df_test["RawMargin"]    = final_raw_margin
    df_test["Prob_Raw"]     = final_proba_raw
    df_test["Pred_Default"] = final_pred_default          # p>=0.5
    df_test["Pred_BestThr"] = final_pred_best_thr         # p>=best_thr
    df_test["BestThr"]      = best_thr                    # 常数列（便于追踪）
    df_test["BestThr_Margin"] = _logit(best_thr)          # 常数列（margin 域阈值）
    outpath = PREDICTION_CSV
    df_test.to_csv(outpath, index=False, encoding="utf-8")
    print("[OK] 预测结果已写出:", outpath)
    print("[OK] 混淆矩阵已保存到:",
          CM_SAVE_PATH.replace(".png","_default.png"),
          "和",
          CM_SAVE_PATH.replace(".png","_bestthr.png"))

# ========== 主入口 ==========
if __name__=="__main__":
    if len(sys.argv)>=2:
        data_path = sys.argv[1]
    else:
        data_path = os.path.join(BASE_DIR, "Data", "test_set.csv")

    if not os.path.exists(data_path):
        print("[Error] 文件不存在:", data_path)
        sys.exit(0)

    evaluate_on_testset(data_path)
    print("\n=== Done ===")
