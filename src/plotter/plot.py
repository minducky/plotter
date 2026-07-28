"""Plotly-based plotting utilities: single-trace 1D/2D plots, multi-panel
subplots, and distribution plots, with optional journal/paper style presets.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotter.paper_presets import resolve_style
from plotter.utils import (
    apply_layout,
    build_heatmap_trace,
    build_line_trace,
    download_figure,
    tickfont,
    to_numpy,
)

# %% Core 1D / 2D plot builders


def plot_1d(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    title_fontcolor: str = "black",
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    xaxis_fontcolor: str = "black",
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    yaxis_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    download: bool = False,
    download_fpath: str | None = None,
) -> None:
    """Plot a single 1D line trace.

    Args:
        x: X-axis values (numpy array or torch tensor).
        y: Y-axis values (numpy array or torch tensor).
        name: Trace name shown in the legend/hover.
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size.
        title_fontcolor: Title font color.
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size.
        xaxis_fontcolor: X-axis label font color.
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size.
        yaxis_fontcolor: Y-axis label font color.
        tick_fontsize: Tick label font size, or None for Plotly's default.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS (e.g. "TASLP_single"),
            or None to use the explicit style args above as-is. When set,
            the preset's font/size/figure-size values override the
            corresponding explicit args.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True. Format is inferred
            from the extension (".html" for interactive HTML, otherwise a
            static image via Kaleido, e.g. ".pdf"/".png"/".svg").
    """
    style = resolve_style(
        paper,
        title_font=title_font,
        title_fontsize=title_fontsize,
        xaxis_font=xaxis_font,
        xaxis_fontsize=xaxis_fontsize,
        yaxis_font=yaxis_font,
        yaxis_fontsize=yaxis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )

    fig = go.Figure()
    fig.add_trace(build_line_trace(x, y, name))
    apply_layout(
        fig,
        title,
        style,
        title_fontcolor=title_fontcolor,
        xaxis_title=xaxis_title,
        xaxis_fontcolor=xaxis_fontcolor,
        yaxis_title=yaxis_title,
        yaxis_fontcolor=yaxis_fontcolor,
        zeroline_color="lightgray",
    )

    fig.show()
    if download:
        download_figure(fig, download_fpath)


def plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    title_fontcolor: str = "black",
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    xaxis_fontcolor: str = "black",
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    yaxis_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    download: bool = False,
    download_fpath: str | None = None,
) -> None:
    """Plot a single 2D heatmap.

    Args:
        x: X-axis coordinates (numpy array or torch tensor).
        y: Y-axis coordinates (numpy array or torch tensor).
        z: 2D array of values to color-map (numpy array or torch tensor).
        name: Trace name shown in hover.
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size.
        title_fontcolor: Title font color.
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size.
        xaxis_fontcolor: X-axis label font color.
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size.
        yaxis_fontcolor: Y-axis label font color.
        tick_fontsize: Tick label font size, or None for Plotly's default.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True. Format is inferred
            from the extension (".html" for interactive HTML, otherwise a
            static image via Kaleido, e.g. ".pdf"/".png"/".svg").
    """
    style = resolve_style(
        paper,
        title_font=title_font,
        title_fontsize=title_fontsize,
        xaxis_font=xaxis_font,
        xaxis_fontsize=xaxis_fontsize,
        yaxis_font=yaxis_font,
        yaxis_fontsize=yaxis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )

    fig = go.Figure()
    fig.add_trace(build_heatmap_trace(x, y, z, name))
    apply_layout(
        fig,
        title,
        style,
        title_fontcolor=title_fontcolor,
        xaxis_title=xaxis_title,
        xaxis_fontcolor=xaxis_fontcolor,
        yaxis_title=yaxis_title,
        yaxis_fontcolor=yaxis_fontcolor,
    )

    fig.show()
    if download:
        download_figure(fig, download_fpath)


# %% Multi-panel and distribution plots


def plot_multi(
    panels: list[dict],
    rows: int,
    cols: int,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    title_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    download: bool = False,
    download_fpath: str | None = None,
) -> None:
    """Plot a grid of 1D and/or 2D panels as subplots.

    Args:
        panels: List of panel dicts, one per subplot, filled row-major into
            the (rows, cols) grid. Each dict has a "kind" of "1d" or "2d"
            plus that kind's data/labels:
                1d: {"kind": "1d", "x", "y", "name", "xaxis_title",
                     "yaxis_title"}
                2d: {"kind": "2d", "x", "y", "z", "name", "xaxis_title",
                     "yaxis_title"}
        rows: Number of subplot rows.
        cols: Number of subplot columns.
        title: Overall figure title.
        title_font: Title font family.
        title_fontsize: Title font size.
        title_fontcolor: Title font color.
        tick_fontsize: Tick label font size for every panel, or None for
            Plotly's default.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True. Format is inferred
            from the extension (".html" for interactive HTML, otherwise a
            static image via Kaleido, e.g. ".pdf"/".png"/".svg").
    """
    style = resolve_style(
        paper,
        title_font=title_font,
        title_fontsize=title_fontsize,
        xaxis_font=title_font,
        xaxis_fontsize=16,
        yaxis_font=title_font,
        yaxis_fontsize=16,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[p.get("name", "") for p in panels]
    )

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, cols)
        row, col = row + 1, col + 1
        if panel["kind"] == "1d":
            trace = build_line_trace(panel["x"], panel["y"], panel.get("name", ""))
        else:
            trace = build_heatmap_trace(
                panel["x"], panel["y"], panel["z"], panel.get("name", "")
            )
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(
            title_text=panel.get("xaxis_title"),
            tickfont=tickfont(style),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=panel.get("yaxis_title"),
            tickfont=tickfont(style),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                family=style["title_font"],
                size=style["title_fontsize"],
                color=title_fontcolor,
            ),
            x=0.5,
            xanchor="center",
        ),
        plot_bgcolor="white",
        width=style["width"],
        height=style["height"],
    )

    fig.show()
    if download:
        download_figure(fig, download_fpath)


def plot_1d_multi(
    series: list[tuple[np.ndarray, np.ndarray, str]],
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    title_fontcolor: str = "black",
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    xaxis_fontcolor: str = "black",
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    yaxis_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    download: bool = False,
    download_fpath: str | None = None,
) -> None:
    """Plot multiple 1D lines together on one shared figure.

    Args:
        series: List of (x, y, name) tuples, one per line; x/y may be numpy
            arrays or torch tensors. Each line gets a
            distinct color from Plotly's default color cycle.
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size.
        title_fontcolor: Title font color.
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size.
        xaxis_fontcolor: X-axis label font color.
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size.
        yaxis_fontcolor: Y-axis label font color.
        tick_fontsize: Tick label font size, or None for Plotly's default.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True. Format is inferred
            from the extension (".html" for interactive HTML, otherwise a
            static image via Kaleido, e.g. ".pdf"/".png"/".svg").
    """
    style = resolve_style(
        paper,
        title_font=title_font,
        title_fontsize=title_fontsize,
        xaxis_font=xaxis_font,
        xaxis_fontsize=xaxis_fontsize,
        yaxis_font=yaxis_font,
        yaxis_fontsize=yaxis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )

    fig = go.Figure()
    for x, y, name in series:
        fig.add_trace(build_line_trace(x, y, name, color=None))
    apply_layout(
        fig,
        title,
        style,
        title_fontcolor=title_fontcolor,
        xaxis_title=xaxis_title,
        xaxis_fontcolor=xaxis_fontcolor,
        yaxis_title=yaxis_title,
        yaxis_fontcolor=yaxis_fontcolor,
        zeroline_color="lightgray",
    )

    fig.show()
    if download:
        download_figure(fig, download_fpath)


def plot_violin(
    data: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    title_fontcolor: str = "black",
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    yaxis_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    download: bool = False,
    download_fpath: str | None = None,
) -> None:
    """Plot one or more distributions as violin plots.

    Args:
        data: List of 1D arrays (numpy or torch), one distribution per violin.
        labels: Name for each violin, one per entry in `data`. Defaults to
            "Group 1", "Group 2", etc.
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size.
        title_fontcolor: Title font color.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size.
        yaxis_fontcolor: Y-axis label font color.
        tick_fontsize: Tick label font size, or None for Plotly's default.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True. Format is inferred
            from the extension (".html" for interactive HTML, otherwise a
            static image via Kaleido, e.g. ".pdf"/".png"/".svg").
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
    style = resolve_style(
        paper,
        title_font=title_font,
        title_fontsize=title_fontsize,
        xaxis_font=title_font,
        xaxis_fontsize=16,
        yaxis_font=yaxis_font,
        yaxis_fontsize=yaxis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )

    fig = go.Figure()
    for values, label in zip(data, labels, strict=True):
        fig.add_trace(
            go.Violin(
                y=to_numpy(values), name=label, box_visible=True, meanline_visible=True
            )
        )
    apply_layout(
        fig,
        title,
        style,
        title_fontcolor=title_fontcolor,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_fontcolor=yaxis_fontcolor,
    )

    fig.show()
    if download:
        download_figure(fig, download_fpath)


def plot_box(
    data: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    title_fontcolor: str = "black",
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    yaxis_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    download: bool = False,
    download_fpath: str | None = None,
) -> None:
    """Plot one or more distributions as box plots.

    Args:
        data: List of 1D arrays (numpy or torch), one distribution per box.
        labels: Name for each box, one per entry in `data`. Defaults to
            "Group 1", "Group 2", etc.
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size.
        title_fontcolor: Title font color.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size.
        yaxis_fontcolor: Y-axis label font color.
        tick_fontsize: Tick label font size, or None for Plotly's default.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True. Format is inferred
            from the extension (".html" for interactive HTML, otherwise a
            static image via Kaleido, e.g. ".pdf"/".png"/".svg").
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
    style = resolve_style(
        paper,
        title_font=title_font,
        title_fontsize=title_fontsize,
        xaxis_font=title_font,
        xaxis_fontsize=16,
        yaxis_font=yaxis_font,
        yaxis_fontsize=yaxis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )

    fig = go.Figure()
    for values, label in zip(data, labels, strict=True):
        fig.add_trace(go.Box(y=to_numpy(values), name=label))
    apply_layout(
        fig,
        title,
        style,
        title_fontcolor=title_fontcolor,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_fontcolor=yaxis_fontcolor,
    )

    fig.show()
    if download:
        download_figure(fig, download_fpath)
