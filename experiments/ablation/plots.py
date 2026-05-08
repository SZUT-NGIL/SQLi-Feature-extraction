from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ABLATION_DIR / "outputs"

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

FEATURE_LABELS = {
    "qlen": "qlen",
    "wcount": "wcount",
    "sq": "sq",
    "dq": "dq",
    "puncts": "puncts",
    "comments": "comments",
    "spaces": "spaces",
    "logic": "logic",
    "arith": "arith",
    "alpha": "alpha",
    "sqlkw": "SQLkw",
    "sqlfunc": "SQLfunc",
}

GROUP_LABELS = {
    "structure": "Structure",
    "obfuscation": "Obfuscation",
    "semantic": "Semantics",
}

GROUP_COLORS = {
    "structure": "#2b6cb0",
    "obfuscation": "#dd6b20",
    "semantic": "#2f855a",
}


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font="Microsoft YaHei")
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_df = pd.read_csv(ROOT / "Data" / "feature_extracted_final.csv")
    group_df = pd.read_csv(OUTPUT_DIR / "group_ablation.csv")
    single_group_df = pd.read_csv(OUTPUT_DIR / "single_group_ablation.csv")
    return features_df, group_df, single_group_df


def make_heatmap(features_df: pd.DataFrame) -> Path:
    z_df = features_df.copy()
    for col in FEATURE_COLUMNS:
        std = z_df[col].std(ddof=0)
        z_df[col] = 0.0 if std == 0 else (z_df[col] - z_df[col].mean()) / std

    profile = (
        z_df.groupby("Label")[FEATURE_COLUMNS]
        .mean()
        .rename(index={0: "Benign", 1: "SQLi"})
        .rename(columns=FEATURE_LABELS)
    )

    out = OUTPUT_DIR / "Benign-SQLi_heatmap _heatmap .png"
    fig, ax = plt.subplots(figsize=(16, 4.6))
    sns.heatmap(
        profile,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("RdBu_r", as_cmap=True),
        center=0,
        linewidths=0.8,
        linecolor="#FFFFFF",
        cbar_kws={"label": "Z-score Normalization"},
        ax=ax,
    )
    ax.set_title("Class-wise Heatmap of Feature Profiles", fontsize=16, pad=26, weight="bold")
    ax.set_xlabel("Feature Dimensions", fontsize=12)
    ax.set_ylabel("Class Label", fontsize=12)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    for boundary in (4, 9):
        ax.vlines(boundary, *ax.get_ylim(), colors="#FFFFFF", linewidth=4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def make_dumbbell(group_df: pd.DataFrame, single_group_df: pd.DataFrame) -> Path:
    full_f1 = float(group_df.loc[group_df["experiment"] == "full", "f1_mean"].iloc[0])
    drop_df = (
        group_df[group_df["removed_type"] == "group"][["removed_name", "f1_mean", "delta_f1_mean"]]
        .rename(
            columns={
                "removed_name": "group",
                "f1_mean": "drop_group_f1",
                "delta_f1_mean": "drop_group_loss",
            }
        )
    )
    keep_df = (
        single_group_df[single_group_df["removed_type"] == "keep_group"][["removed_name", "f1_mean"]]
        .rename(columns={"removed_name": "group", "f1_mean": "keep_only_f1"})
    )
    merged = drop_df.merge(keep_df, on="group", how="inner")
    merged["group_label"] = merged["group"].map(GROUP_LABELS)
    merged = merged.sort_values("drop_group_loss", ascending=True).reset_index(drop=True)
    group_pos = {row["group"]: idx for idx, (_, row) in enumerate(merged.iterrows())}

    out = OUTPUT_DIR / "group-ablation.png"
    y = np.arange(len(merged))

    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    ax.set_facecolor("#FAFAFA")
    for idx, row in merged.iterrows():
        ax.plot(
            [row["keep_only_f1"], row["drop_group_f1"]],
            [y[idx], y[idx]],
            color="#CBD5E0",
            linewidth=4,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            row["keep_only_f1"],
            y[idx],
            s=170,
            color=GROUP_COLORS[row["group"]],
            edgecolor="white",
            linewidth=1.8,
            marker="o",
            zorder=3,
        )
        ax.scatter(
            row["drop_group_f1"],
            y[idx],
            s=180,
            color=GROUP_COLORS[row["group"]],
            edgecolor="white",
            linewidth=1.8,
            marker="D",
            zorder=4,
        )
        ax.text(row["keep_only_f1"] - 0.0008, y[idx] + 0.12, f"keep only: {fmt_pct(row['keep_only_f1'])}", ha="right", fontsize=10)
        ax.text(row["drop_group_f1"] + 0.0008, y[idx] + 0.12, f"drop_group: {fmt_pct(row['drop_group_f1'])}", ha="left", fontsize=10)
        ax.text(full_f1 + 0.00012, y[idx] - 0.18, f"drop_group_loss {fmt_pct(row['drop_group_loss'])}", ha="left", fontsize=10, color="#7B341E")

    label_pad = max(0.0012, (full_f1 - merged["keep_only_f1"].min()) * 0.04)
    x_min = merged["keep_only_f1"].min() - 0.0035
    label_y = (group_pos.get("semantic", 0) + group_pos.get("structure", 0)) / 2

    ax.axvline(full_f1, color="#C53030", linestyle="--", linewidth=1)
    ax.text(
        full_f1 - label_pad,
        label_y,
        f"Full Feature F1 = {fmt_pct(full_f1)}",
        ha="right",
        va="center",
        fontsize=11,
        color="#C53030",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#C53030", alpha=0.95),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(merged["group_label"])
    ax.set_xlabel("F1 (%)", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f}%"))
    ax.set_title("Role Differences of Feature Groups in Terms of Necessity and Independence", fontsize=17, pad=16, weight="bold")
    ax.set_xlim(x_min, full_f1 + 0.0032)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_style()
    features_df, group_df, single_group_df = load_inputs()
    heatmap = make_heatmap(features_df)
    dumbbell = make_dumbbell(group_df, single_group_df)
    print(f"Generated figures: {heatmap.name}, {dumbbell.name}")



if __name__ == "__main__":
    main()
