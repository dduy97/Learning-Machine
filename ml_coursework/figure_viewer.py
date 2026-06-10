"""Interactive viewer for generated project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def show_saved_figures(figures: dict[str, str], title_prefix: str = "") -> None:
    """Show saved PNG figures in a single Matplotlib window."""
    if not figures:
        print("No figures to show.")
        return

    loaded_figures = []
    for figure_name, figure_path in figures.items():
        path = Path(figure_path)
        if path.exists():
            loaded_figures.append((figure_name, path, plt.imread(path)))
        else:
            print(f"Missing figure: {path}")

    if not loaded_figures:
        print("No existing figures to show.")
        return

    current_index = {"value": 0}
    fig, ax = plt.subplots(figsize=(11, 7))
    plt.subplots_adjust(bottom=0.14)

    def render() -> None:
        figure_name, path, image = loaded_figures[current_index["value"]]
        title = figure_name.replace("_", " ").title()
        if title_prefix:
            title = f"{title_prefix}: {title}"

        ax.clear()
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{title} ({current_index['value'] + 1}/{len(loaded_figures)})")
        fig.canvas.manager.set_window_title(path.name)
        fig.canvas.draw_idle()

    def previous(_: object) -> None:
        current_index["value"] = (current_index["value"] - 1) % len(loaded_figures)
        render()

    def next_figure(_: object) -> None:
        current_index["value"] = (current_index["value"] + 1) % len(loaded_figures)
        render()

    previous_button = Button(fig.add_axes([0.34, 0.035, 0.14, 0.055]), "Previous")
    next_button = Button(fig.add_axes([0.52, 0.035, 0.14, 0.055]), "Next")
    previous_button.on_clicked(previous)
    next_button.on_clicked(next_figure)

    # Keep button objects alive while the Matplotlib window is open.
    fig._navigation_buttons = (previous_button, next_button)  # type: ignore[attr-defined]

    render()
    plt.show()
