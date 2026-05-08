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

import ours.predict as ours_predict

from cross_dataset.common import (
    RUNS_DIR,
    dataset_paths,
    ensure_dataset_dirs,
    ensure_prediction_metrics_csv,
    load_source_dataset,
    load_metadata,
    normalize_labels,
    resolve_dataset_names,
    save_summary_csv,
)

PREDICTION_SUMMARY_COLUMNS = [
    "dataset",
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
    parser = argparse.ArgumentParser(description="Predict with the feature-based SQLi model on external datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "sajid576", "syedsaqlainhussain"],
        default="all",
        help="Which dataset to predict on. Default: all.",
    )
    return parser.parse_args()


def _configure_predict_module(paths) -> None:
    ours_predict.MODEL_DIR = str(paths.model_root)
    ours_predict.NUMERIC_FEATURES_DIR = str(paths.numeric_dir)
    ours_predict.CM_SAVE_PATH = str(paths.predict_dir / "confusion_matrix.png")
    ours_predict.PREDICTION_CSV = str(paths.prediction_csv)
    ours_predict.csv_file = str(paths.prediction_metrics_csv)
    ours_predict.SAVE_PLOTS = False


def _patched_read_data_for_predict(data_path: str) -> pd.DataFrame:
    df = normalize_labels(load_source_dataset(Path(data_path)))
    df["Query"] = df["Query"].fillna("").astype(str)
    return df


def predict_one_dataset(dataset_name: str) -> dict:
    paths = dataset_paths(dataset_name)
    ensure_dataset_dirs(paths)
    load_metadata(paths)
    ensure_prediction_metrics_csv(paths)
    _configure_predict_module(paths)
    ours_predict.read_data_for_predict = _patched_read_data_for_predict

    print(f"\n[PREDICT] dataset={dataset_name}")
    print(f"[PREDICT] test csv => {paths.test_csv}")
    ours_predict.evaluate_on_testset(str(paths.test_csv))

    metric_df = pd.read_csv(paths.prediction_metrics_csv)
    if metric_df.empty:
        raise ValueError(f"No prediction metrics were written for {dataset_name}")
    latest_metric = metric_df.iloc[-1].to_dict()

    return {
        "dataset": dataset_name,
        **{key: latest_metric.get(key) for key in PREDICTION_SUMMARY_COLUMNS if key != "dataset"},
    }


def _normalize_summary_rows(rows: list[dict]) -> list[dict]:
    return [{column: row.get(column) for column in PREDICTION_SUMMARY_COLUMNS} for row in rows]


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

    summary_rows = []
    for dataset_name in resolve_dataset_names(args.dataset):
        summary_row = predict_one_dataset(dataset_name)
        summary_rows.append(summary_row)

    summary_path = RUNS_DIR / "prediction_summary.csv"
    save_summary_csv(_merge_summary_rows(summary_path, summary_rows), summary_path)
    print(f"\n[PREDICT] saved summary => {summary_path}")


if __name__ == "__main__":
    main()
