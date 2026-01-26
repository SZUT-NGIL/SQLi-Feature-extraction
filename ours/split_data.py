import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 当前文件路径
current_file = Path(__file__).resolve()
# 项目根目录
BASE_DIR = current_file.parents[1]

# 1. 读取数据并去重
df = pd.read_csv(
    os.path.join(BASE_DIR, "Data", "All_SQL_Dataset.csv"), encoding="latin1"
)
print(f"原始数据量: {len(df)}")

# 按Query字段去重
df = df.drop_duplicates(subset="Query")
print(f"去重后数据量: {len(df)}")

# 2. 检查标签分布
label_dist = df["Label"].value_counts(normalize=True)
print("\n原始标签分布:")
print(label_dist)

# 3. 分层划分数据集（保持标签比例）
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Label"],  # 按标签分层抽样
    random_state=42       # 固定随机种子保证可复现
)

# 4. 验证划分后的分布
print("\n训练集标签分布:")
print(train_df["Label"].value_counts(normalize=True))
print("\n测试集标签分布:")
print(test_df["Label"].value_counts(normalize=True))

# 5. 保存结果
train_df.to_csv(os.path.join(BASE_DIR, "Data", "train_set.csv"), index=False)
test_df.to_csv(os.path.join(BASE_DIR, "Data", "test_set.csv"), index=False)
