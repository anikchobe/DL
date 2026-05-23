"""Shared look for the solution graphs.

Only styling lives here — no maths, no training.  Each solutionN_viz.py imports
these and gives the colours names that mean something in that stage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Three hues that stay apart for colourblind readers as well.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

# For quantities that are *ordered* (step number, learning rate): one hue from
# light to dark, so bigger reads as darker rather than as an unrelated colour.
BLUE_RAMP_4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BLUE_RAMP_5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

SURFACE = "#fcfcfb"  # paper behind the plots
INK = "#0b0b0b"      # titles, and the data itself
MUTED = "#898781"    # axis labels, reference lines, notes
GRID = "#e1e0d9"     # hairline gridlines
AXIS = "#c3c2b7"     # the zero rule


def style_axes(axes, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive chrome, so the data is the loudest thing on the panel."""
    axes.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=10)
    axes.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, labelsize=8, length=0)
    for side, spine in axes.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(AXIS)
        spine.set_linewidth(1)


def draw_line(axes, xs, ys, color, label=None, marker="o"):
    """A thin line whose markers are ringed in the paper colour, so they never merge."""
    return axes.plot(
        xs, ys,
        color=color, linewidth=2, label=label,
        marker=marker, markersize=4.5,
        markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
    )


def note(axes, text: str, x: float, y: float, color: str = MUTED, **options) -> None:
    """A comment placed in the panel, positioned in axes fractions not data units."""
    axes.annotate(text, xy=(x, y), xycoords="axes fraction",
                  color=color, fontsize=8.5, **options)


def legend(axes, loc: str = "best", **options):
    """Identity is never carried by colour alone, so every panel with two or
    more series gets one of these."""
    options.setdefault("frameon", False)
    options.setdefault("fontsize", 8.5)
    options.setdefault("labelcolor", MUTED)
    return axes.legend(loc=loc, **options)


def finish(figure, output_path: Path) -> Path:
    """Save the figure, and open a window too when there is one to open."""
    figure.savefig(output_path, dpi=140, facecolor=SURFACE)
    if matplotlib.get_backend().lower() != "agg":  # "agg" means there is no window
        plt.show()
    return output_path
