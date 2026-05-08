from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from experiments.ablation.config import FEATURE_COLUMNS, FEATURE_GROUPS, XGB_PARAMS
from experiments.common.bootstrap import bootstrap_repo_paths
from experiments.common.io import ensure_dir, read_dataset, resolve_train_test_paths
from experiments.common.metrics import evaluate_scores, scores_from_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation experiments for 12 SQLi features.")
    parser.add_argument("--train-path", default=None, help="Optional fixed training split.")
    parser.add_argument("--test-path", default=None, help="Optional fixed test split.")
    parser.add_argument("--data-path", default=None, help="Optional full dataset path for repeated re-splitting.")
    parser.add_argument(
        "--source-mode",
        choices=["auto", "fixed_split", "resplit_full"],
        default="auto",
        help="Dataset protocol. Default prefers existing fixed split for reproducible ablations.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test ratio when re-splitting a full dataset.")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.3,
        help="Validation ratio carved from the training split for threshold selection.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["artifact", "validation", "fixed"],
        default="artifact",
        help="How to choose the decision threshold. Default reuses the saved main-model threshold.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Decision threshold used when --threshold-mode fixed.",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="Comma-separated random seeds used for repeated runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory, default: experiments/ablation/outputs",
    )
    return parser.parse_args()


def _parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one random seed is required.")
    return seeds


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    bootstrap_repo_paths()
    from utils.hfes import extract_struct_features

    return extract_struct_features(df.copy()).reset_index(drop=True)


def _drop_duplicate_queries(df: pd.DataFrame) -> pd.DataFrame:
    dedup_df = df.copy()
    dedup_df["Query"] = dedup_df["Query"].astype(str).str.lower()
    dedup_df = dedup_df.drop_duplicates(subset=["Query"]).reset_index(drop=True)
    return dedup_df


def _load_artifact_threshold() -> float:
    threshold_path = ROOT_DIR / "ours" / "model" / "1" / "numeric_features" / "best_threshold.pkl"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold artifact not found: {threshold_path}")
    with open(threshold_path, "rb") as file_obj:
        return float(pickle.load(file_obj))


def _resolve_source(args: argparse.Namespace) -> tuple[str, pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]]:
    if args.source_mode == "fixed_split" or args.train_path or args.test_path:
        train_path, test_path = resolve_train_test_paths(args.train_path, args.test_path)
        if not (train_path and test_path):
            raise FileNotFoundError("Both --train-path and --test-path must exist.")
        print(f"[INFO] use fixed split: {train_path} / {test_path}")
        return "fixed_split", (read_dataset(train_path), read_dataset(test_path))

    train_path, test_path = resolve_train_test_paths(None, None)
    if args.source_mode == "auto" and train_path and test_path:
        print(f"[INFO] use fixed split by default: {train_path} / {test_path}")
        return "fixed_split", (read_dataset(train_path), read_dataset(test_path))

    full_path = Path(args.data_path) if args.data_path else ROOT_DIR / "Data" / "All_SQL_Dataset.csv"
    if full_path.exists():
        if args.source_mode == "auto":
            print("[WARN] fixed split not found, falling back to repeated re-splitting on full dataset.")
        print(f"[INFO] use repeated re-splitting on full dataset: {full_path}")
        return "resplit_full", _drop_duplicate_queries(read_dataset(full_path))

    raise FileNotFoundError("No valid dataset source found for ablation experiments.")


def _materialize_seed_split(
    source_mode: str,
    source_obj: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
    seed: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source_mode == "fixed_split":
        train_df, test_df = source_obj
        return train_df.copy(), test_df.copy()

    from sklearn.model_selection import train_test_split

    full_df = source_obj
    train_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df["Label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _split_train_validation(train_df: pd.DataFrame, seed: int, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    if not 0.0 < val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1.")

    fit_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["Label"],
        random_state=seed,
    )
    return fit_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _find_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -math.inf
    probs = np.asarray(scores, dtype=float).reshape(-1)
    for threshold in np.linspace(0.0, 1.0, 101):
        metrics = evaluate_scores(y_true, probs, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def _train_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_subset: list[str],
    seed: int,
    threshold_mode: str,
    fixed_threshold: float,
    artifact_threshold: float | None,
) -> dict[str, float | str]:
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_subset].values)
    x_val = scaler.transform(val_df[feature_subset].values)
    x_test = scaler.transform(test_df[feature_subset].values)
    y_train = train_df["Label"].to_numpy(dtype=int)
    y_val = val_df["Label"].to_numpy(dtype=int)
    y_test = test_df["Label"].to_numpy(dtype=int)

    params = dict(XGB_PARAMS)
    params["random_state"] = seed
    params["n_jobs"] = 1

    model = XGBClassifier(**params)
    model.fit(x_train, y_train)

    if threshold_mode == "artifact":
        if artifact_threshold is None:
            raise ValueError("artifact threshold requested but no artifact threshold is available.")
        threshold = artifact_threshold
        threshold_source = "artifact"
    elif threshold_mode == "fixed":
        threshold = fixed_threshold
        threshold_source = "fixed"
    else:
        val_scores = scores_from_model(model, x_val)
        threshold = _find_best_threshold(val_scores, y_val)
        threshold_source = "validation"

    test_scores = scores_from_model(model, x_test)
    metrics = evaluate_scores(y_test, test_scores, threshold=threshold)
    metrics["threshold"] = float(threshold)
    metrics["threshold_source"] = threshold_source
    return metrics


def _build_experiment_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "experiment": "full",
            "removed_type": "none",
            "removed_name": "none",
            "feature_subset": FEATURE_COLUMNS,
        }
    ]

    for group_name, group_features in FEATURE_GROUPS.items():
        specs.append(
            {
                "experiment": f"drop_group_{group_name}",
                "removed_type": "group",
                "removed_name": group_name,
                "feature_subset": [feature for feature in FEATURE_COLUMNS if feature not in group_features],
            }
        )
        specs.append(
            {
                "experiment": f"keep_group_{group_name}",
                "removed_type": "keep_group",
                "removed_name": group_name,
                "feature_subset": [feature for feature in FEATURE_COLUMNS if feature in group_features],
            }
        )

    for removed_feature in FEATURE_COLUMNS:
        specs.append(
            {
                "experiment": f"drop_feature_{removed_feature}",
                "removed_type": "feature",
                "removed_name": removed_feature,
                "feature_subset": [feature for feature in FEATURE_COLUMNS if feature != removed_feature],
            }
        )

    return specs


def _run_all_seeds(
    source_mode: str,
    source_obj: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
    seeds: list[int],
    test_size: float,
    val_size: float,
    threshold_mode: str,
    fixed_threshold: float,
) -> pd.DataFrame:
    raw_rows = []
    specs = _build_experiment_specs()
    artifact_threshold = _load_artifact_threshold() if threshold_mode == "artifact" else None

    if source_mode == "fixed_split":
        base_train_df, base_test_df = source_obj
        feature_source = (_extract_features(base_train_df), _extract_features(base_test_df))
    else:
        feature_source = _extract_features(source_obj)

    for seed in seeds:
        print(f"[INFO] running seed={seed}")
        if source_mode == "fixed_split":
            train_feat_df, test_feat_df = feature_source
            train_feat_df = train_feat_df.copy()
            test_feat_df = test_feat_df.copy()
        else:
            train_feat_df, test_feat_df = _materialize_seed_split(
                source_mode, feature_source, seed, test_size
            )

        fit_feat_df, val_feat_df = _split_train_validation(train_feat_df, seed, val_size)
        per_seed_rows = []
        for spec in specs:
            feature_subset = list(spec["feature_subset"])
            metrics = _train_eval(
                fit_feat_df,
                val_feat_df,
                test_feat_df,
                feature_subset,
                seed,
                threshold_mode,
                fixed_threshold,
                artifact_threshold,
            )
            per_seed_rows.append(
                {
                    "seed": seed,
                    "experiment": spec["experiment"],
                    "removed_type": spec["removed_type"],
                    "removed_name": spec["removed_name"],
                    "kept_features": ",".join(feature_subset),
                    "feature_count": len(feature_subset),
                    **metrics,
                }
            )

        seed_df = pd.DataFrame(per_seed_rows)
        full_row = seed_df[seed_df["experiment"] == "full"].iloc[0]
        seed_df["delta_f1"] = full_row["f1"] - seed_df["f1"]
        seed_df["delta_auc"] = full_row["auc"] - seed_df["auc"]
        raw_rows.extend(seed_df.to_dict("records"))

    return pd.DataFrame(raw_rows)


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "experiment",
        "removed_type",
        "removed_name",
        "kept_features",
        "feature_count",
        "threshold_source",
    ]
    metric_cols = ["acc", "prec", "rec", "f1", "auc", "delta_f1", "delta_auc", "threshold"]

    summary = df.groupby(group_cols).agg(
        n_seeds=("seed", "nunique"),
        **{f"{col}_mean": (col, "mean") for col in metric_cols},
        **{f"{col}_std": (col, "std") for col in metric_cols},
    ).reset_index()

    summary = summary.fillna(0.0)
    return summary


def _plot_ablation_worker(df: pd.DataFrame, title: str, save_path_str: str, y_label: str) -> None:
    save_path = Path(save_path_str)
    plot_df = df[df["removed_name"] != "none"].copy().sort_values("delta_f1_mean", ascending=True)
    if plot_df.empty:
        return

    fig = go.Figure(
        go.Bar(
            x=plot_df["delta_f1_mean"],
            y=plot_df["removed_name"],
            orientation="h",
            marker_color="#e15759",
            text=[f"{float(value):.4f}" for value in plot_df["delta_f1_mean"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="simple_white",
        width=1200,
        height=max(520, 90 * len(plot_df)),
        margin=dict(l=140, r=60, t=60, b=80),
        title=title,
        font=dict(family="Arial", size=16),
    )
    fig.update_xaxes(title_text="Mean F1 Drop", gridcolor="rgba(0, 0, 0, 0.12)")
    fig.update_yaxes(title_text=y_label, categoryorder="array", categoryarray=plot_df["removed_name"].tolist())
    fig.write_image(str(save_path), scale=2)


def _plot_ablation(df: pd.DataFrame, title: str, save_path: Path, y_label: str) -> None:
    try:
        _plot_ablation_worker(df, title, str(save_path), y_label)
    except Exception as exc:
        print(f"[WARN] failed to generate plot {save_path.name}: {exc}")


def _plot_single_group_f1_worker(df: pd.DataFrame, title: str, save_path_str: str) -> None:
    save_path = Path(save_path_str)
    plot_df = df[df["removed_type"] == "keep_group"].copy().sort_values("f1_mean", ascending=True)
    if plot_df.empty:
        return

    fig = go.Figure(
        go.Bar(
            x=plot_df["f1_mean"],
            y=plot_df["removed_name"],
            orientation="h",
            marker_color="#4e79a7",
            text=[f"{float(value):.3f}" for value in plot_df["f1_mean"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="simple_white",
        width=1200,
        height=max(520, 110 * len(plot_df)),
        margin=dict(l=140, r=60, t=60, b=80),
        title=title,
        font=dict(family="Arial", size=16),
    )
    fig.update_xaxes(title_text="Mean F1", range=[0.0, 1.05], gridcolor="rgba(0, 0, 0, 0.12)")
    fig.update_yaxes(title_text="Retained Group", categoryorder="array", categoryarray=plot_df["removed_name"].tolist())
    fig.write_image(str(save_path), scale=2)


def _plot_single_group_f1(df: pd.DataFrame, title: str, save_path: Path) -> None:
    try:
        _plot_single_group_f1_worker(df, title, str(save_path))
    except Exception as exc:
        print(f"[WARN] failed to generate plot {save_path.name}: {exc}")


def main() -> None:
    args = _parse_args()
    seeds = _parse_seed_list(args.seeds)
    output_dir = ensure_dir(args.output_dir or Path(__file__).resolve().parent / "outputs")

    source_mode, source_obj = _resolve_source(args)
    raw_df = _run_all_seeds(
        source_mode,
        source_obj,
        seeds,
        args.test_size,
        args.val_size,
        args.threshold_mode,
        args.fixed_threshold,
    )
    group_raw_df = raw_df[raw_df["removed_type"].isin(["none", "group"])].copy()
    single_group_raw_df = raw_df[raw_df["removed_type"].isin(["none", "keep_group"])].copy()
    feature_raw_df = raw_df[raw_df["removed_type"] == "feature"].copy()

    group_df = _summarize_results(group_raw_df)
    single_group_df = _summarize_results(single_group_raw_df).sort_values(
        "delta_f1_mean", ascending=False
    ).reset_index(drop=True)
    feature_df = _summarize_results(feature_raw_df).sort_values(
        "delta_f1_mean", ascending=False
    ).reset_index(drop=True)

    group_raw_path = output_dir / "group_ablation_raw.csv"
    single_group_raw_path = output_dir / "single_group_ablation_raw.csv"
    feature_raw_path = output_dir / "feature_ablation_raw.csv"
    group_path = output_dir / "group_ablation.csv"
    single_group_path = output_dir / "single_group_ablation.csv"
    feature_path = output_dir / "feature_ablation.csv"
    group_raw_df.to_csv(group_raw_path, index=False, encoding="utf-8-sig")
    single_group_raw_df.to_csv(single_group_raw_path, index=False, encoding="utf-8-sig")
    feature_raw_df.to_csv(feature_raw_path, index=False, encoding="utf-8-sig")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    single_group_df.to_csv(single_group_path, index=False, encoding="utf-8-sig")
    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] saved group raw csv: {group_raw_path}")
    print(f"[INFO] saved single-group raw csv: {single_group_raw_path}")
    print(f"[INFO] saved feature raw csv: {feature_raw_path}")
    print(f"[INFO] saved group summary csv: {group_path}")
    print(f"[INFO] saved single-group summary csv: {single_group_path}")
    print(f"[INFO] saved feature summary csv: {feature_path}")
    print(f"[INFO] saved outputs dir: {output_dir}")


if __name__ == "__main__":
    main()


