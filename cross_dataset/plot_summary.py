from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
INPUT_CSV = ROOT_DIR / "summary.csv"
OUTPUT_DIR = ROOT_DIR / "plots"
OUTPUT_PNG = OUTPUT_DIR / "summary_comparison.png"


def _load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"dataset", "ACC", "TPR", "F1", "Method"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")

    for metric in ("ACC", "TPR", "F1"):
        df[metric] = pd.to_numeric(df[metric], errors="raise") * 100.0
    return df


def _build_heatmap_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    dataset_order = list(dict.fromkeys(df["dataset"].tolist()))
    rows: list[dict] = []
    split_lines: list[int] = []

    for dataset in dataset_order:
        dataset_rows = df.loc[df["dataset"] == dataset].copy()
        baseline_rows = dataset_rows.loc[dataset_rows["Method"].str.lower() != "ours"]
        ours_rows = dataset_rows.loc[dataset_rows["Method"].str.lower() == "ours"]

        if len(baseline_rows) != 1 or len(ours_rows) != 1:
            raise ValueError(
                f"Dataset '{dataset}' must contain exactly one baseline row and one Ours row."
            )

        baseline_row = baseline_rows.iloc[0]
        ours_row = ours_rows.iloc[0]

        rows.append(
            {
                "RowLabel": f"{dataset} | {baseline_row['Method']}",
                "ACC": float(baseline_row["ACC"]),
                "TPR": float(baseline_row["TPR"]),
                "F1": float(baseline_row["F1"]),
                "Role": "Baseline",
            }
        )
        rows.append(
            {
                "RowLabel": f"{dataset} | Ours",
                "ACC": float(ours_row["ACC"]),
                "TPR": float(ours_row["TPR"]),
                "F1": float(ours_row["F1"]),
                "Role": "Ours",
            }
        )
        split_lines.append(len(rows))

    heatmap_df = pd.DataFrame(rows)
    return heatmap_df, split_lines[:-1]


def draw_summary_plot(
    input_csv: Path = INPUT_CSV,
    output_png: Path = OUTPUT_PNG,
) -> Path:
    df = _load_summary(input_csv)
    heatmap_df, split_lines = _build_heatmap_table(df)

    metrics = ["ACC", "TPR", "F1"]
    values = heatmap_df[metrics].to_numpy(dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    image = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(heatmap_df)))
    ax.set_yticklabels(heatmap_df["RowLabel"], fontsize=11)

    for idx, tick in enumerate(ax.get_yticklabels()):
        if heatmap_df.iloc[idx]["Role"] == "Ours":
            tick.set_color("#b33a2f")
            tick.set_fontweight("bold")
        else:
            tick.set_color("#47515c")

    midpoint = (vmin + vmax) / 2.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "white" if value >= midpoint else "#222222"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    for split_line in split_lines:
        ax.axhline(split_line - 0.5, color="white", linewidth=3)

    ax.set_title("Literature Comparison", fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dataset | Method", fontsize=12, fontweight="bold")
    ax.tick_params(top=False, bottom=True, left=True, right=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Score (%)", fontsize=11, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_png


if __name__ == "__main__":
    saved_path = draw_summary_plot()
    print(f"saved plot: {saved_path}")
