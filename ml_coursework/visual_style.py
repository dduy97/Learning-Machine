from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


BACKGROUND = "#f8fafc"
AXIS_BACKGROUND = "#ffffff"
GRID_COLOR = "#e2e8f0"
TEXT_COLOR = "#0f172a"
MUTED_TEXT = "#475569"
BLUE = "#2563eb"
TEAL = "#0f766e"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
INK = "#111827"

QUALITATIVE_PALETTE = [
    "#2563eb",
    "#0f766e",
    "#dc2626",
    "#d97706",
    "#7c3aed",
    "#475569",
]
SEQUENTIAL_PALETTE = ["#dbeafe", "#93c5fd", "#60a5fa", "#2563eb", "#1e40af"]
WARM_PALETTE = ["#fff7ed", "#fed7aa", "#fb923c", "#ea580c", "#9a3412"]


def apply_project_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=QUALITATIVE_PALETTE,
        rc={
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": AXIS_BACKGROUND,
            "font.family": ["Segoe UI", "DejaVu Sans", "Arial"],
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": MUTED_TEXT,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.9,
            "grid.alpha": 0.8,
            "legend.frameon": True,
            "legend.facecolor": AXIS_BACKGROUND,
            "legend.edgecolor": "#cbd5e1",
            "legend.title_fontsize": 10,
            "legend.fontsize": 10,
            "savefig.facecolor": BACKGROUND,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        },
    )


def polish_axes(
    ax: plt.Axes,
    rotate_x: int | None = None,
    percent_x: bool = False,
    percent_y: bool = False,
) -> None:
    ax.title.set_position((0.0, 1.03))
    ax.title.set_horizontalalignment("left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", length=0)

    if rotate_x is not None:
        ax.tick_params(axis="x", rotation=rotate_x)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    if percent_x:
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    if percent_y:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def add_bar_labels(ax: plt.Axes, fmt: str = "{:.2f}", percent: bool = False) -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if percent:
                labels.append(f"{value:.1%}")
            else:
                labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, color=MUTED_TEXT)


def save_chart(fig: plt.Figure, output_path: Path) -> str:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return str(output_path)
