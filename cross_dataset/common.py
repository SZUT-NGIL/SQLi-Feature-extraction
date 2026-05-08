from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

import chardet
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "cross_dataset" / "runs"

DATASET_PATHS = {
    "sajid576": ROOT_DIR / "Data" / "deep.csv",
    "syedsaqlainhussain": ROOT_DIR / "Data" / "UniEmbed_AE-NET",
}


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    source_path: Path
    run_dir: Path
    data_dir: Path
    train_dir: Path
    predict_dir: Path
    model_root: Path
    numeric_dir: Path
    source_csv: Path
    train_csv: Path
    test_csv: Path
    feature_csv: Path
    train_metrics_csv: Path
    prediction_csv: Path
    prediction_metrics_csv: Path
    metadata_json: Path


def dataset_paths(name: str) -> DatasetPaths:
    if name not in DATASET_PATHS:
        raise KeyError(f"Unknown dataset: {name}")

    run_dir = RUNS_DIR / name
    data_dir = run_dir / "data"
    train_dir = run_dir / "train"
    predict_dir = run_dir / "predict"
    model_root = run_dir / "model" / "1"
    numeric_dir = model_root / "numeric_features"
    return DatasetPaths(
        name=name,
        source_path=DATASET_PATHS[name],
        run_dir=run_dir,
        data_dir=data_dir,
        train_dir=train_dir,
        predict_dir=predict_dir,
        model_root=model_root,
        numeric_dir=numeric_dir,
        source_csv=data_dir / "source_preprocessed.csv",
        train_csv=data_dir / "train_set.csv",
        test_csv=data_dir / "test_set.csv",
        feature_csv=train_dir / "feature_extracted_final.csv",
        train_metrics_csv=train_dir / "train_metrics.csv",
        prediction_csv=predict_dir / "predictions.csv",
        prediction_metrics_csv=predict_dir / "prediction_metrics.csv",
        metadata_json=run_dir / "metadata.json",
    )


def ensure_dataset_dirs(paths: DatasetPaths) -> None:
    for path in (
        RUNS_DIR,
        paths.run_dir,
        paths.data_dir,
        paths.train_dir,
        paths.predict_dir,
        paths.model_root,
        paths.numeric_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_dataset_names(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(DATASET_PATHS.keys())
    if dataset_arg not in DATASET_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset_arg}")
    return [dataset_arg]


def _candidate_encodings(source_path: Path) -> list[str]:
    detected = chardet.detect(source_path.read_bytes()[:200000]).get("encoding")
    encodings = [detected, "utf-8", "utf-8-sig", "utf-16", "latin1"]
    unique: list[str] = []
    for encoding in encodings:
        if encoding and encoding not in unique:
            unique.append(encoding)
    return unique


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        text = str(column).replace("\ufeff", "").strip()
        lowered = text.lower()
        if lowered == "query":
            renamed[column] = "Query"
        elif lowered == "label":
            renamed[column] = "Label"
        else:
            renamed[column] = text
    return df.rename(columns=renamed)


def load_source_dataset(source_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in _candidate_encodings(source_path):
        try:
            df = pd.read_csv(source_path, encoding=encoding, on_bad_lines="skip")
            df = _canonicalize_columns(df)
            if {"Query", "Label"}.issubset(df.columns):
                return df[["Query", "Label"]].copy()
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc

    if last_error is not None:
        raise ValueError(f"Unable to read dataset {source_path}: {last_error}") from last_error
    raise ValueError(f"Unable to read dataset {source_path}")


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    def _map_label(value) -> int:
        if pd.isna(value):
            raise ValueError("Label contains missing values.")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(float(value) >= 0.5)

        text = str(value).strip().lower()
        if text in {"1", "1.0", "attack", "attacks", "malicious", "true", "yes", "positive"}:
            return 1
        if text in {"0", "0.0", "normal", "benign", "false", "no", "negative"}:
            return 0

        try:
            return int(float(text) >= 0.5)
        except ValueError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Unsupported label value: {value}") from exc

    out_df = df.copy()
    out_df["Query"] = out_df["Query"].fillna("")
    out_df["Label"] = out_df["Label"].map(_map_label).astype("int64")
    return out_df


def _load_ae_net_sqliv3(source_path: Path) -> pd.DataFrame:
    rows: list[tuple[str, str | int]] = []
    with source_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        next(reader, None)
        for row in reader:
            if not row:
                continue

            label = None
            for cell in row[1:]:
                text = str(cell).strip()
                if text in {"0", "1", "attack", "benign"}:
                    label = text
                    break

            if label is None:
                continue
            rows.append((row[0], label))

    return pd.DataFrame(rows, columns=["Query", "Label"])


def load_ae_net_merged_dataset(source_dir: Path) -> tuple[pd.DataFrame, dict]:
    sqli_path = source_dir / "sqli.csv"
    sqliv2_path = source_dir / "sqliv2.csv"
    sqliv3_path = source_dir / "SQLiV3.csv"

    sqli_df = pd.read_csv(sqli_path, encoding="utf-16", on_bad_lines="skip")[["Sentence", "Label"]].rename(
        columns={"Sentence": "Query"}
    )
    sqliv2_df = pd.read_csv(sqliv2_path, encoding="utf-16", on_bad_lines="skip")[["Query", "Label"]].copy()
    sqliv3_df = _load_ae_net_sqliv3(sqliv3_path)

    merged_df = pd.concat([sqli_df, sqliv2_df, sqliv3_df], ignore_index=True)
    merged_df = normalize_labels(merged_df)
    merged_df["Query"] = merged_df["Query"].astype(str).str.strip()

    empty_mask = merged_df["Query"].isin(["", "nan"])
    merged_df = merged_df.loc[~empty_mask].copy()
    merged_df["word_count"] = merged_df["Query"].str.split().map(len)
    two_word_mask = merged_df["word_count"] == 2
    prepared_df = merged_df.loc[~two_word_mask, ["Query", "Label"]].reset_index(drop=True)

    summary = {
        "raw_rows": int(len(sqli_df) + len(sqliv2_df) + len(sqliv3_df)),
        "empty_rows_removed": int(empty_mask.sum()),
        "two_word_rows_removed": int(two_word_mask.sum()),
        "preprocessed_rows": int(len(prepared_df)),
        "preprocessed_label_distribution": prepared_df["Label"].value_counts(normalize=True).sort_index().to_dict(),
    }
    return prepared_df, summary


def split_like_ours(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dedup_df = df.drop_duplicates(subset=["Query"]).reset_index(drop=True)
    train_df, test_df = train_test_split(
        dedup_df,
        test_size=test_size,
        stratify=dedup_df["Label"],
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    summary = {
        "source_rows": int(len(df)),
        "dedup_rows": int(len(dedup_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_distribution_source": df["Label"].value_counts(normalize=True).sort_index().to_dict(),
        "label_distribution_train": train_df["Label"].value_counts(normalize=True).sort_index().to_dict(),
        "label_distribution_test": test_df["Label"].value_counts(normalize=True).sort_index().to_dict(),
        "split_test_size": float(test_size),
        "split_random_state": int(random_state),
    }
    return train_df, test_df, summary


def split_without_dedup(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Label"],
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    summary = {
        "source_rows": int(len(df)),
        "dedup_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_distribution_source": df["Label"].value_counts(normalize=True).sort_index().to_dict(),
        "label_distribution_train": train_df["Label"].value_counts(normalize=True).sort_index().to_dict(),
        "label_distribution_test": test_df["Label"].value_counts(normalize=True).sort_index().to_dict(),
        "split_test_size": float(test_size),
        "split_random_state": int(random_state),
        "dedup_applied": False,
    }
    return train_df, test_df, summary


def prepare_dataset_splits(name: str) -> tuple[DatasetPaths, dict]:
    paths = dataset_paths(name)
    ensure_dataset_dirs(paths)

    if name == "syedsaqlainhussain":
        source_df, extra_summary = load_ae_net_merged_dataset(paths.source_path)
        source_df.to_csv(paths.source_csv, index=False, encoding="utf-8-sig")
        train_df, test_df, summary = split_without_dedup(source_df)
    else:
        source_df = normalize_labels(load_source_dataset(paths.source_path))
        source_df.to_csv(paths.source_csv, index=False, encoding="utf-8-sig")
        train_df, test_df, summary = split_like_ours(source_df)
        extra_summary = {"dedup_applied": True}

    train_df.to_csv(paths.train_csv, index=False, encoding="utf-8")
    test_df.to_csv(paths.test_csv, index=False, encoding="utf-8")

    metadata = {
        "dataset_name": name,
        "source_path": str(paths.source_path),
        "source_csv": str(paths.source_csv),
        "train_csv": str(paths.train_csv),
        "test_csv": str(paths.test_csv),
        **summary,
        **extra_summary,
    }
    paths.metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths, metadata


def save_train_metrics_csv(paths: DatasetPaths, metrics: dict[str, dict[str, float]]) -> None:
    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append({"model": model_name, **model_metrics})
    pd.DataFrame(rows).to_csv(paths.train_metrics_csv, index=False, encoding="utf-8-sig")


def ensure_prediction_metrics_csv(paths: DatasetPaths) -> None:
    if paths.prediction_metrics_csv.exists():
        return
    pd.DataFrame(columns=["Method", "ACC", "PREC", "TPR", "F1", "AUC", "TP", "TN", "FP", "FN"]).to_csv(
        paths.prediction_metrics_csv,
        index=False,
        encoding="utf-8-sig",
    )


def load_metadata(paths: DatasetPaths) -> dict:
    if not paths.metadata_json.exists():
        raise FileNotFoundError(f"Missing metadata file: {paths.metadata_json}")
    return json.loads(paths.metadata_json.read_text(encoding="utf-8-sig"))


def save_summary_csv(rows: list[dict], output_path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
