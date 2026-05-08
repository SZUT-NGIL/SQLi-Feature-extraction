from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OURS_DIR = ROOT_DIR / "ours"
for candidate in (ROOT_DIR, OURS_DIR):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import ours.train as ours_train

from cross_dataset.common import (
    RUNS_DIR,
    load_source_dataset,
    normalize_labels,
    prepare_dataset_splits,
    resolve_dataset_names,
    save_summary_csv,
    save_train_metrics_csv,
)

TRAIN_SUMMARY_COLUMNS = [
    "dataset",
    "source_rows",
    "dedup_rows",
    "train_rows",
    "ACC",
    "PREC",
    "TPR",
    "F1",
    "AUC",
    "TP",
    "TN",
    "FP",
    "FN",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the feature-based SQLi model on external datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "sajid576", "syedsaqlainhussain"],
        default="all",
        help="Which dataset to train on. Default: all.",
    )
    return parser.parse_args()


def _configure_train_module(model_root: Path) -> None:
    ours_train.MODEL_DIR = str(model_root)
    Path(ours_train.MODEL_DIR).mkdir(parents=True, exist_ok=True)


def _patched_read_data(data_path: str) -> pd.DataFrame:
    df = normalize_labels(load_source_dataset(Path(data_path)))

    print(f"[INFO] read CSV: {data_path}")
    print(f"[INFO] raw rows: {df.shape[0]}")
    print("\n[INFO] label distribution:")
    print(df["Label"].value_counts())

    df["Query"] = df["Query"].fillna("").astype(str).str.lower()
    df.drop_duplicates(subset=["Query", "Label"], inplace=True)
    print(f"[INFO] deduplicated rows: {df.shape[0]}")
    return df


def _load_latest_prediction_metrics(paths) -> dict:
    metric_keys = ["ACC", "PREC", "TPR", "F1", "AUC", "TP", "TN", "FP", "FN"]
    empty_metrics = {key: None for key in metric_keys}

    if not paths.prediction_metrics_csv.exists():
        return empty_metrics

    metric_df = pd.read_csv(paths.prediction_metrics_csv)
    if metric_df.empty:
        return empty_metrics

    latest_metric = metric_df.iloc[-1].to_dict()
    return {key: latest_metric.get(key) for key in metric_keys}


def train_one_dataset(dataset_name: str) -> dict:
    paths, metadata = prepare_dataset_splits(dataset_name)
    _configure_train_module(paths.model_root)
    ours_train.read_data = _patched_read_data

    print(f"\n[TRAIN] dataset={dataset_name}")
    print(f"[TRAIN] source={paths.source_path}")
    print(f"[TRAIN] preprocessed source => {paths.source_csv}")
    print(f"[TRAIN] split train/test => {paths.train_csv} / {paths.test_csv}")
    if "preprocessed_rows" in metadata:
        print(f"[TRAIN] raw/preprocessed rows => {metadata['raw_rows']} / {metadata['preprocessed_rows']}")

    data = ours_train.read_data(str(paths.train_csv))
    data["Query_preprocessed"] = data["Query"].apply(ours_train.advanced_preprocess)
    data = ours_train.extract_struct_features(data)
    data.to_csv(paths.feature_csv, index=False, encoding="utf-8", escapechar="\\")
    print(f"[TRAIN] saved feature csv => {paths.feature_csv}")

    features_dict = ours_train.train_test_split_and_featurize(data)
    train_feat, test_feat = features_dict["num_features"]
    model_results = ours_train.train_and_save_model(
        feature_name="numeric_features",
        train_x=train_feat,
        train_y=features_dict["y_train"],
        test_x=test_feat,
        test_y=features_dict["y_test"],
    )
    save_train_metrics_csv(paths, model_results)
    print(f"[TRAIN] saved train metrics => {paths.train_metrics_csv}")

    return {
        "dataset": dataset_name,
        "source_rows": metadata["source_rows"],
        "dedup_rows": metadata["dedup_rows"],
        "train_rows": metadata["train_rows"],
        **_load_latest_prediction_metrics(paths),
    }


def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    return [{column: row.get(column) for column in TRAIN_SUMMARY_COLUMNS} for row in rows]


def _merge_summary_rows(existing_path: Path, fresh_rows: list[dict]) -> list[dict]:
    if not existing_path.exists():
        return _normalize_summary_rows(fresh_rows)

    existing_df = pd.read_csv(existing_path)
    existing_records = existing_df.to_dict("records")
    valid_datasets = set(resolve_dataset_names("all"))
    replacement = {row["dataset"]: row for row in fresh_rows}
    merged = [
        row
        for row in existing_records
        if row.get("dataset") in valid_datasets and row.get("dataset") not in replacement
    ]
    merged.extend(fresh_rows)
    return _normalize_summary_rows(merged)


def main() -> None:
    args = _parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name in resolve_dataset_names(args.dataset):
        rows.append(train_one_dataset(dataset_name))

    summary_path = RUNS_DIR / "train_summary.csv"
    save_summary_csv(_merge_summary_rows(summary_path, rows), summary_path)
    print(f"\n[TRAIN] saved summary => {summary_path}")


if __name__ == "__main__":
    main()
