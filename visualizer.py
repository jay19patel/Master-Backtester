"""Chart rendering helpers. Runs headless (no display) and saves straight to image
files, since this project is driven from the console."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class HeatmapVisualizer:
    """Renders a small labeled percentage heatmap (e.g. direction vs signal) to a
    JPG file.

    Usage:
        HeatmapVisualizer("direction_signal_heatmap.jpg").plot_percentage_heatmap(
            row_pct, title="Direction vs Signal", x_label="Signal", y_label="Direction"
        )
    """

    def __init__(self, output_path):
        self.output_path = output_path

    def plot_percentage_heatmap(self, data, title, x_label, y_label):
        """data: a pandas DataFrame of percentages (0-100), rows x columns."""
        n_rows, n_cols = data.shape
        fig, ax = plt.subplots(figsize=(1.8 * n_cols + 2.5, 1.4 * n_rows + 2))

        image = ax.imshow(data.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(data.columns, fontsize=12)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(data.index, fontsize=12)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

        for i in range(n_rows):
            for j in range(n_cols):
                value = data.values[i, j]
                text_color = "white" if value >= 60 or value <= 15 else "black"
                ax.text(
                    j, i, f"{value:.1f}%", ha="center", va="center",
                    color=text_color, fontsize=13, fontweight="bold",
                )

        fig.colorbar(image, ax=ax, label="%")
        fig.tight_layout()
        fig.savefig(self.output_path, format="jpg", dpi=150)
        plt.close(fig)

        print(f"[HeatmapVisualizer] Saved heatmap to {self.output_path}")
        return self.output_path
