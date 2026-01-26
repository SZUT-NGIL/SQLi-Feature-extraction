import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class AcademicPlotter:
    def __init__(self, input_csv, output_dir='results/plots'):
        self.input_csv = input_csv
        self.output_dir = output_dir
        
        # 字体配置（支持中文）
        self.fonts = ['Songti SC', 'PingFang SC', 'Heiti TC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['font.sans-serif'] = self.fonts
        plt.rcParams['axes.unicode_minus'] = False
        
        # 颜色板
        self.colors_acc = ['#1f77b4', '#ff7f0e', '#2ca02c']
        self.colors_pre = ['#1f77b4', '#ff7f0e']
        self.colors_err = ['#d62728', '#9467bd']
        
        # 散点图形状映射：OurCN 使用元组(5,1)表示五边形
        self.marker_map = {
            'BoW': 'o',
            'TF-IDF': 's',
            'Word2Vec': '^',
            'FastText': 'D',
            'Bert': 'v',
            'Ours': 'o'    # 我们的方法
        }

    def _load_data(self):
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"错误：找不到文件 {self.input_csv}")
        df = pd.read_csv(self.input_csv)
        # 统一乘100转百分比
        metrics_to_convert = ['ACC', 'PREC', 'TPR', 'F1', 'FNR', 'FPR', 'AUC']
        for metric in metrics_to_convert:
            if metric in df.columns:
                df[metric] = df[metric] * 100
        return df

    def _ensure_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _apply_academic_style(self, ax, ylabel, title, x_labels):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.tick_params(axis='y', labelsize=10)

    def _plot_grouped_bar(self, df, metrics, colors, title, ylabel, filename, focus_high=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        models = df['Method']
        values = df[metrics].values
        x = np.arange(len(models))
        bar_width = 0.25 if len(metrics) > 2 else 0.35
        group_spacing = 0.02 if len(metrics) > 2 else 0.0

        for i, metric in enumerate(metrics):
            offset = (i - (len(metrics) - 1) / 2) * (bar_width + group_spacing)
            rects = ax.bar(x + offset, values[:, i], bar_width, label=metric,
                           color=colors[i], alpha=0.9)
            ax.bar_label(rects, padding=3, labels=[f'{v:.2f}%' for v in values[:, i]],
                         fontsize=9, color='black')

        self._apply_academic_style(ax, ylabel, title, models)

        if focus_high:
            min_val = values.min()
            lower_bound = max(90.0, min_val - 1.0) if min_val > 90 else max(0, min_val - 5)
            if min_val > 98:
                lower_bound = min_val - 0.2
            ax.set_ylim(lower_bound, 100.05)
            if min_val > 98:
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        else:
            ax.set_ylim(0, values.max() * 1.25)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=len(metrics), frameon=False, fontsize=10)

        self._save_plot(plt, filename)

    def _plot_scatter_tradeoff(self, df, filename):
        fig, ax = plt.subplots(figsize=(10, 8))

        fpr_vals = df['FPR'].values
        fnr_vals = df['FNR'].values
        model_names = df['Method'].values
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

        scatter_points = []
        for i, name in enumerate(model_names):
            marker = self.marker_map.get(name, 'o')
            color = colors[i]
            s = 200 if name == 'OurCN' else 180
            sc = ax.scatter(fpr_vals[i], fnr_vals[i], marker=marker, s=s,
                            color=color, edgecolors='white', linewidth=1.5,
                            alpha=0.9, zorder=5)
            scatter_points.append((sc, name))

            coord_label = f"{name}\n({fpr_vals[i]:.2f}%, {fnr_vals[i]:.2f}%)"
            xytext = (10, 5)
            fontweight = 'normal'
            if name == 'OurCN':
                xytext = (10, -25)
                fontweight = 'bold'
            elif name == 'Word2Vec':
                xytext = (10, 12)
            elif name == 'Bert':
                xytext = (10, 0)

            ax.annotate(coord_label, (fpr_vals[i], fnr_vals[i]),
                        xytext=xytext, textcoords='offset points',
                        fontsize=11, fontweight=fontweight, color='#333333')

        ideal_sc = ax.scatter(0, 0, marker='*', s=250, color='#d62728', label='Ideal (0,0)', zorder=10, edgecolors='black')

        ax.annotate("Optimization Direction", xy=(0.02, 0.02), xytext=(0.15, 0.15),
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.5, alpha=0.6),
                    fontsize=10, color="green", alpha=0.6)

        ax.set_xlabel('False Positive Rate (FPR, %) — Lower is Better', fontsize=13)
        ax.set_ylabel('False Negative Rate (FNR, %) — Lower is Better', fontsize=13)
        #ax.set_title('Security Performance Trade-off (FNR vs. FPR)', fontsize=15, pad=20)

        ax.set_xlim(-0.05, max(fpr_vals) * 1.35)
        ax.set_ylim(-0.05, max(fnr_vals) * 1.25)

        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles = [sc for sc, _ in scatter_points]
        labels = [name for _, name in scatter_points]

        handles.append(ideal_sc)
        labels.append('Ideal (0,0)')

        ax.legend(handles, labels, loc='lower right', frameon=True, fontsize=11, title="Methods", labelspacing=1.2, borderpad=1.0)

        self._save_plot(plt, filename)

    def _save_plot(self, plt_obj, filename):
        full_path = os.path.join(self.output_dir, filename)
        plt_obj.tight_layout()
        plt_obj.savefig(full_path, dpi=300, bbox_inches='tight')
        plt_obj.close()
        print(f"✅ 图表已保存: {full_path}")

    def run(self):
        try:
            self._ensure_dir()
            df = self._load_data()
            print(f"数据加载成功，共 {len(df)} 行。开始绘图...")

            self._plot_grouped_bar(
                df, ['ACC', 'F1', 'AUC'], self.colors_acc,
                '',#Performance Comparison of Different Methods (ACC, F1, AUC)
                'Metric Value (%)',
                'model_acc_f1_auc.png', focus_high=True
            )

            self._plot_grouped_bar(
                df, ['PREC', 'TPR'], self.colors_pre,
                '', #Security Detection Capability：PREC and TPR
                'Metric Value  (%)',
                'pre_rec.png', focus_high=True
            )

            self._plot_grouped_bar(
                df, ['FNR', 'FPR'], self.colors_err,
                '', #Security Risk Metrics: FNR and FPR
                'Error Rate (%)',
                'fnr_fpr.png', focus_high=False
            )

            self._plot_scatter_tradeoff(df, 'scatter_fnr_fpr.png')

            print("\n🎉 所有图表绘制完成！")
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def draw_all_plots(csv_path='results/csv/results.csv'):
    plotter = AcademicPlotter(csv_path)
    plotter.run()

if __name__ == "__main__":
    draw_all_plots()
