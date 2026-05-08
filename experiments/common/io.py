from __future__ import annotations

from pathlib import Path

import chardet
import numpy as np
import pandas as pd

from .bootstrap import ROOT_DIR


DATA_DIR = ROOT_DIR / "Data"


def ensure_dir(path: Path | str) -> Path:
    """确保输出目录存在。"""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_labels(series: pd.Series) -> pd.Series:
    """将标签统一映射为 0/1。"""
    if np.issubdtype(series.dtype, np.number):
        return series.astype(int)
    return series.apply(lambda x: 1 if str(x).strip().lower() == "attack" else 0).astype(int)


def read_dataset(path: Path | str) -> pd.DataFrame:
    """读取 CSV/JSON 数据，并统一列名和标签类型。"""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if data_path.suffix.lower() == ".json":
        df = pd.read_json(data_path)
    else:
        try:
            df = pd.read_csv(data_path, encoding="latin1", on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(data_path, "rb") as file_obj:
                enc = chardet.detect(file_obj.read())["encoding"] or "utf-8"
            df = pd.read_csv(data_path, encoding=enc, on_bad_lines="skip")

    rename_map: dict[str, str] = {}
    column_aliases = {
        "sentence": "Query",
        "query": "Query",
        "payload": "Query",
        "data": "Query",
        "label": "Label",
        "labels": "Label",
        "class": "Label",
    }
    existing_columns = set(df.columns)
    for column in df.columns:
        target = column_aliases.get(str(column).strip().lower())
        if target and target not in existing_columns:
            rename_map[column] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = {"Query", "Label"} - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} 缺少必要列: {sorted(missing)}")

    out_df = df.copy()
    out_df["Query"] = out_df["Query"].fillna("").astype(str)
    out_df["Label"] = normalize_labels(out_df["Label"])
    return out_df


def resolve_test_path(user_path: str | None = None) -> Path:
    if user_path:
        return Path(user_path)
    for candidate in (DATA_DIR / "test_set.csv", DATA_DIR / "All_SQL_Dataset.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到测试数据，请显式传入 --data-path。")


def resolve_train_test_paths(
    train_path: str | None = None,
    test_path: str | None = None,
) -> tuple[Path | None, Path | None]:
    """优先使用现成划分；若不存在，则由调用方自行决定如何切分全量数据。"""
    train_candidate = Path(train_path) if train_path else DATA_DIR / "train_set.csv"
    test_candidate = Path(test_path) if test_path else DATA_DIR / "test_set.csv"

    if train_candidate.exists() and test_candidate.exists():
        return train_candidate, test_candidate
    return None, None
