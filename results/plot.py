import os

import pandas as pd
import plotly.graph_objects as go


class AcademicPlotter:
    def __init__(self, input_csv, output_dir="results/plots"):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.method_order = ["BoW", "TF-IDF", "Word2Vec", "FastText", "BERT", "Ours"]
        self.colors_acc = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        self.colors_pre = ["#1f77b4", "#ff7f0e"]
        self.colors_err = ["#d62728", "#9467bd"]
        self.marker_map = {
            "BoW": "circle",
            "TF-IDF": "square",
            "Word2Vec": "triangle-up",
            "FastText": "diamond",
            "BERT": "triangle-down",
            "Ours": "circle",
        }

    def _load_data(self):
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"找不到文件: {self.input_csv}")

        df = pd.read_csv(self.input_csv)
        metrics_to_convert = ["ACC", "PREC", "TPR", "F1", "FNR", "FPR", "AUC"]
        for metric in metrics_to_convert:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce") * 100.0

        df["Method"] = pd.Categorical(df["Method"], categories=self.method_order, ordered=True)
        df = df.sort_values("Method").reset_index(drop=True)
        return df

    def _ensure_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _base_layout(self, fig, ylabel):
        fig.update_layout(
            template="simple_white",
            width=1200,
            height=720,
            font=dict(family="Arial", size=16),
            legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
            margin=dict(l=90, r=40, t=40, b=110),
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", tickangle=0)
        fig.update_yaxes(
            title_text=ylabel,
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="rgba(0, 0, 0, 0.12)",
        )

    def _save_plot(self, fig, filename):
        full_path = os.path.join(self.output_dir, filename)
        fig.write_image(full_path, scale=2)
        print(f"saved plot: {full_path}")

    def _plot_grouped_bar(self, df, metrics, colors, ylabel, filename, focus_high=True):
        fig = go.Figure()
        for metric, color in zip(metrics, colors):
            fig.add_trace(
                go.Bar(
                    x=df["Method"],
                    y=df[metric],
                    name=metric,
                    marker_color=color,
                    text=[f"{value:.2f}%" for value in df[metric]],
                    textposition="outside",
                )
            )

        self._base_layout(fig, ylabel)
        fig.update_layout(barmode="group")

        values = df[metrics].to_numpy(dtype=float)
        if focus_high:
            min_val = float(values.min())
            lower_bound = max(90.0, min_val - 1.0) if min_val > 90 else max(0.0, min_val - 5.0)
            if min_val > 98:
                lower_bound = min_val - 0.2
            fig.update_yaxes(range=[lower_bound, 100.05])
        else:
            fig.update_yaxes(range=[0.0, float(values.max()) * 1.25])

        self._save_plot(fig, filename)

    def _plot_scatter_tradeoff(self, df, filename):
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b", "#e377c2", "#17becf"]
        label_shift_map = {
            "BoW": {"xshift": 5, "yshift": 10},
            "TF-IDF": {"xshift": -80, "yshift": 20},
            "Word2Vec": {"xshift": 18, "yshift": 12},
            "FastText": {"xshift": 14, "yshift": 6},
            "BERT": {"xshift": 12, "yshift": -18},
            "Ours": {"xshift": -80, "yshift": -4},
        }

        fig = go.Figure()
        annotations = []
        for index, row in df.iterrows():
            method = row["Method"]
            color = colors[index % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=[row["FPR"]],
                    y=[row["FNR"]],
                    mode="markers",
                    name=str(method),
                    hovertemplate=f"{method}<br>FPR=%{{x:.2f}}%<br>FNR=%{{y:.2f}}%<extra></extra>",
                    marker=dict(
                        symbol=self.marker_map.get(str(method), "circle"),
                        size=22 if str(method) == "Ours" else 18,
                        color=color,
                        line=dict(color="white", width=1.5),
                    ),
                )
            )
            shift = label_shift_map.get(str(method), {"xshift": 16, "yshift": 8})
            annotations.append(
                dict(
                    x=row["FPR"],
                    y=row["FNR"],
                    xref="x",
                    yref="y",
                    text=f"{method} ({row['FPR']:.2f}%, {row['FNR']:.2f}%)",
                    showarrow=False,
                    xshift=shift["xshift"],
                    yshift=shift["yshift"],
                    xanchor="left",
                    align="left",
                    font=dict(size=10, color="#333333"),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers",
                name="Ideal (0,0)",

                hovertemplate="Ideal (0,0)<extra></extra>",
                marker=dict(symbol="star", size=22, color="#d62728", line=dict(color="black", width=1)),
            )
        )
        annotations.append(
            dict(
                x=0.0,
                y=0.0,
                xref="x",
                yref="y",
                text="Ideal (0,0)",
                showarrow=False,
                xshift=10,
                yshift=16,
                xanchor="left",
                align="left",
                font=dict(size=10, color="#333333"),
            )
        )

        max_fpr = float(df["FPR"].max())
        max_fnr = float(df["FNR"].max())
        fig.update_layout(
            template="simple_white",
            width=1200,
            height=820,
            font=dict(family="Arial", size=16),
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
            margin=dict(l=90, r=40, t=40, b=90),
            annotations=[
                dict(
                    x=0.02,
                    y=0.02,
                    ax=0.18,
                    ay=0.18,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    text="Optimization Direction",
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=1.5,
                    arrowcolor="green",
                    font=dict(color="green", size=14),
                )
            ] + annotations,
        )
        fig.update_xaxes(
            title_text="False Positive Rate (FPR, %) - Lower is Better",
            range=[-0.05, max_fpr * 1.35 if max_fpr > 0 else 1.0],
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="rgba(0, 0, 0, 0.12)",
        )
        fig.update_yaxes(
            title_text="False Negative Rate (FNR, %) - Lower is Better",
            range=[-0.05, max_fnr * 1.25 if max_fnr > 0 else 1.0],
            showline=True,
            linewidth=1,
            linecolor="black",
            gridcolor="rgba(0, 0, 0, 0.12)",
        )

        self._save_plot(fig, filename)

    def run(self):
        self._ensure_dir()
        df = self._load_data()
        print(f"loaded {len(df)} rows from {self.input_csv}")

        self._plot_grouped_bar(df, ["ACC", "F1", "AUC"], self.colors_acc, "Metric Value (%)", "model_acc_f1_auc.png", focus_high=True)
        self._plot_grouped_bar(df, ["PREC", "TPR"], self.colors_pre, "Metric Value (%)", "pre_tpr.png", focus_high=True)
        self._plot_grouped_bar(df, ["FNR", "FPR"], self.colors_err, "Error Rate (%)", "fnr_fpr.png", focus_high=False)
        self._plot_scatter_tradeoff(df, "scatter_fnr_fpr.png")


def draw_all_plots(csv_path="results/csv/results.csv"):
    plotter = AcademicPlotter(csv_path)
    plotter.run()


if __name__ == "__main__":
    draw_all_plots()
