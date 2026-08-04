"""Plotly-based plotting utilities: single-trace 1D/2D plots, multi-panel
subplots, and distribution plots, with optional journal/paper style presets.
"""

from plotter.paper_presets import PAPER_PRESETS
from plotter.plot import (
    plot_1d,
    plot_1d_multi,
    plot_2d,
    plot_3d,
    plot_box,
    plot_confusion_matrix,
    plot_multi,
    plot_violin,
)

__all__ = [
    "PAPER_PRESETS",
    "plot_1d",
    "plot_1d_multi",
    "plot_2d",
    "plot_3d",
    "plot_box",
    "plot_confusion_matrix",
    "plot_multi",
    "plot_violin",
]
