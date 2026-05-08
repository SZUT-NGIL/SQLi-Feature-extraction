from __future__ import annotations


# 与论文中的 12 维特征保持一致。
FEATURE_COLUMNS = [
    "qlen",
    "wcount",
    "sq",
    "dq",
    "puncts",
    "comments",
    "spaces",
    "logic",
    "arith",
    "alpha",
    "sqlkw",
    "sqlfunc",
]


# 这里给出一个可解释、可直接改写的默认分组。
# 如果正文里已经定义了更严格的分组边界，直接改这里即可。
FEATURE_GROUPS = {
    "structure": ["qlen", "spaces", "alpha", "wcount"],
    "obfuscation": ["sq", "dq", "puncts", "comments", "arith"],
    "semantic": ["logic", "sqlkw", "sqlfunc"],
}


XGB_PARAMS = {
    "max_depth": 7,
    "learning_rate": 0.2,
    "n_estimators": 400,
    "gamma": 0,
    "reg_lambda": 3,
    "disable_default_eval_metric": True,
    #"random_state": 42,
}
