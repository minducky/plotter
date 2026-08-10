"""Plotly-based plotting utilities: single-trace 1D/2D/3D plots, multi-panel
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
    build_surface_trace,
    finalise_figure,
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
    color: str | None = "black",
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
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
        color: Line color (any Plotly-recognized color string, e.g.
            "#1f77b4" or "blue").
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS (e.g. "TASLP_single"),
            or None to use the explicit style args above as-is. When set,
            the preset's font/size/figure-size values override the
            corresponding explicit args.
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
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
    fig.add_trace(build_line_trace(x, y, name, color=color))
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

    finalise_figure(fig, show, download, download_fpath)


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
    colorscale: str = "Cividis",
    zmin: float | None = None,
    zmax: float | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
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
        colorscale: Plotly heatmap colorscale name (e.g. "Cividis" for
            sequential data, or a diverging one like "PuOr" for data centered
            at zero — pair a diverging colorscale with fixed zmin/zmax).
        zmin: Lower bound of the color range, or None for Plotly's
            auto-range. Set explicitly (e.g. -1) alongside a diverging
            colorscale so zero always maps to the same color.
        zmax: Upper bound of the color range, or None for Plotly's
            auto-range.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
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
    fig.add_trace(
        build_heatmap_trace(x, y, z, name, colorscale=colorscale, zmin=zmin, zmax=zmax)
    )
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

    finalise_figure(fig, show, download, download_fpath)


# %% 3D plot builder


def plot_3d(
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
    zaxis_title: str | None = None,
    zaxis_font: str = "Arial",
    zaxis_fontsize: int = 16,
    zaxis_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    showscale: bool = True,
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
) -> None:
    """Plot a single 3D surface.

    Args:
        x: X-axis coordinates (numpy array or torch tensor).
        y: Y-axis coordinates (numpy array or torch tensor).
        z: 2D array of surface heights (numpy array or torch tensor).
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
        zaxis_title: Z-axis label.
        zaxis_font: Z-axis label font family.
        zaxis_fontsize: Z-axis label font size.
        zaxis_fontcolor: Z-axis label font color.
        tick_fontsize: Tick label font size for all three axes, or None for
            Plotly's default.
        width: Figure width in pixels.
        height: Figure height in pixels.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
            Paper presets only define x/y-axis font/size; zaxis_font/
            zaxis_fontsize are never overridden by a preset.
        showscale: Whether to show the surface's color scale bar.
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
    """
    style = resolve_style(
        paper,
        title_font=title_font,
        title_fontsize=title_fontsize,
        xaxis_font=xaxis_font,
        xaxis_fontsize=xaxis_fontsize,
        yaxis_font=yaxis_font,
        yaxis_fontsize=yaxis_fontsize,
        zaxis_font=zaxis_font,
        zaxis_fontsize=zaxis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )

    fig = go.Figure()
    fig.add_trace(build_surface_trace(x, y, z, name, showscale=showscale))
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
        scene=dict(
            xaxis_title=dict(
                text=xaxis_title,
                font=dict(
                    family=style["xaxis_font"],
                    size=style["xaxis_fontsize"],
                    color=xaxis_fontcolor,
                ),
            ),
            yaxis_title=dict(
                text=yaxis_title,
                font=dict(
                    family=style["yaxis_font"],
                    size=style["yaxis_fontsize"],
                    color=yaxis_fontcolor,
                ),
            ),
            zaxis_title=dict(
                text=zaxis_title,
                font=dict(
                    family=style["zaxis_font"],
                    size=style["zaxis_fontsize"],
                    color=zaxis_fontcolor,
                ),
            ),
            xaxis_tickfont=tickfont(style),
            yaxis_tickfont=tickfont(style),
            zaxis_tickfont=tickfont(style),
        ),
        width=style["width"],
        height=style["height"],
    )

    finalise_figure(fig, show, download, download_fpath)


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
    link_x: bool = False,
    link_y: bool = False,
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    return_fig: bool = False,
) -> go.Figure | None:
    """Plot a grid of 1D, 2D, and/or 3D panels as subplots.

    Args:
        panels: List of panel dicts, one per subplot, filled row-major into
            the (rows, cols) grid. Each dict has a "kind" of "1d", "2d", or
            "3d" plus that kind's data/labels:
                1d: {"kind": "1d", "x", "y", "name", "xaxis_title",
                     "yaxis_title", "color"}. "color" defaults to "black"
                     (matching build_line_trace) if omitted.
                2d: {"kind": "2d", "x", "y", "z", "name", "xaxis_title",
                     "yaxis_title", "colorscale", "zmin", "zmax"}.
                     "colorscale" defaults to "Cividis"; "zmin"/"zmax"
                     default to None (Plotly auto-ranges) if omitted,
                     matching build_heatmap_trace's defaults. Set an explicit
                     "colorscale" (e.g. a colorblind-safe diverging one, see
                     paper_presets.DIVERGING_COLORSCALE_COLORBLIND_SAFE) plus
                     fixed "zmin"/"zmax" per-panel to mix e.g. a sequential
                     heatmap and a diverging heatmap in the same grid.
                3d: {"kind": "3d", "x", "y", "z", "name", "xaxis_title",
                     "yaxis_title", "zaxis_title", "showscale"}. "showscale"
                     defaults to False since a full-size colorbar per 3D
                     panel tends to crowd a multi-panel grid; set it to True
                     per-panel to opt back in.
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
        link_x: If True, zooming/panning the x-axis of any 1D/2D panel
            applies the same x-range to every 1D/2D panel. Has no effect on
            3D panels (Plotly scenes don't share axes across subplots).
        link_y: If True, zooming/panning the y-axis of any 1D/2D panel
            applies the same y-range to every 1D/2D panel. Combine with
            link_x=True to fully sync 2D panels (e.g. heatmaps) across the
            grid.
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
        return_fig: If True, skip fig.show()/download and return the built
            go.Figure instead, so the caller can post-process it (e.g. add
            row-label annotations via fig.add_annotation, or per-panel tick
            overrides via fig.update_yaxes(row=,col=,tickvals=,ticktext=))
            before saving it themselves. Defaults to False, matching every
            other plot_* function's auto show/save behavior.

    Returns:
        The built go.Figure if return_fig=True, else None.
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

    has_2d = any(panel["kind"] == "2d" for panel in panels)
    has_3d = any(panel["kind"] == "3d" for panel in panels)
    specs = None
    if has_3d:
        specs = [
            [
                {"type": "scene"}
                if idx < len(panels) and panels[idx]["kind"] == "3d"
                else {"type": "xy"}
                for idx in range(r * cols, r * cols + cols)
            ]
            for r in range(rows)
        ]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[p.get("name", "") for p in panels],
        specs=specs,
        # Leave room between columns for each 2D panel's own colorbar so it
        # doesn't overlap the tick labels of the panel to its right.
        horizontal_spacing=(0.18 if has_2d else None),
    )

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, cols)
        row, col = row + 1, col + 1
        kind = panel["kind"]
        if kind == "1d":
            trace = build_line_trace(
                panel["x"],
                panel["y"],
                panel.get("name", ""),
                color=panel.get("color", "black"),
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
        elif kind == "2d":
            subplot = fig.get_subplot(row, col)
            x0, x1 = subplot.xaxis.domain
            y0, y1 = subplot.yaxis.domain
            colorbar = dict(
                x=x1 + 0.015, y=(y0 + y1) / 2, len=y1 - y0, thickness=15
            )
            trace = build_heatmap_trace(
                panel["x"],
                panel["y"],
                panel["z"],
                panel.get("name", ""),
                colorscale=panel.get("colorscale", "Cividis"),
                colorbar=colorbar,
                zmin=panel.get("zmin"),
                zmax=panel.get("zmax"),
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
        else:  # "3d"
            trace = build_surface_trace(
                panel["x"],
                panel["y"],
                panel["z"],
                panel.get("name", ""),
                showscale=panel.get("showscale", False),
            )
            fig.add_trace(trace, row=row, col=col)
            fig.update_scenes(
                xaxis_title=panel.get("xaxis_title"),
                yaxis_title=panel.get("yaxis_title"),
                zaxis_title=panel.get("zaxis_title"),
                row=row,
                col=col,
            )

    if link_x:
        fig.update_xaxes(matches="x")
    if link_y:
        fig.update_yaxes(matches="y")

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

    if return_fig:
        return fig

    finalise_figure(fig, show, download, download_fpath)
    return None


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
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
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
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
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

    finalise_figure(fig, show, download, download_fpath)


def plot_violin(
    data: list[np.ndarray],
    labels: list[str] | None = None,
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
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
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
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
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
        xaxis_fontcolor=xaxis_fontcolor,
        yaxis_title=yaxis_title,
        yaxis_fontcolor=yaxis_fontcolor,
    )

    finalise_figure(fig, show, download, download_fpath)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str | None = None,
    normalize: bool = True,
    reversed_axis: bool = True,
    colorscale: str = "Blues",
    title_font: str | None = None,
    title_fontsize: int = 16,
    title_fontcolor: str = "black",
    xaxis_title: str | None = "Predicted",
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    xaxis_fontcolor: str = "black",
    yaxis_title: str | None = "True",
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    yaxis_fontcolor: str = "black",
    tick_fontsize: int | None = None,
    width: int | None = None,
    height: int | None = None,
    paper: str | None = None,
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
) -> None:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes); rows are true
            classes, columns are predicted classes. Numpy array or torch
            tensor.
        labels: Class labels, length n_classes, shared by both axes.
        title: Figure title.
        normalize: If True (default), row-normalize so each cell shows
            recall (fraction of the true class predicted as each class);
            colorbar is fixed to [0, 1] and formatted as a percentage. If
            False, plots raw counts.
        reversed_axis: If True (default), reverses the y-axis so the
            diagonal runs from top-left to bottom-right.
        colorscale: Plotly heatmap colorscale name.
        title_font: Title font family. Defaults to None (Plotly's own
            default font, unstyled) rather than "Arial" like the other
            plot_* functions, so a bare call reproduces a plain, unbolded
            title. Pass an explicit family (or a `paper` preset) to style
            it.
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
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
    """
    cm = to_numpy(cm).astype(float)
    labels = [str(l) for l in labels]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        z = np.where(row_sums > 0, cm / row_sums, 0.0)
        zmin, zmax = 0, 1
        colorbar = dict(title="Recall", tickformat=".0%")
        hovertemplate = "True: %{y}<br>Pred: %{x}<br>Recall: %{z:.1%}<extra></extra>"
    else:
        z = cm
        zmin, zmax = None, None
        colorbar = dict(title="Count")
        hovertemplate = "True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>"

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

    title_font_dict = dict(size=style["title_fontsize"], color=title_fontcolor)
    if style["title_font"] is not None:
        title_font_dict["family"] = style["title_font"]

    fig = go.Figure(
        build_heatmap_trace(
            labels,
            labels,
            z,
            "",
            colorscale=colorscale,
            colorbar=colorbar,
            zmin=zmin,
            zmax=zmax,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=title_font_dict),
        xaxis=dict(
            title=dict(
                text=xaxis_title,
                font=dict(
                    family=style["xaxis_font"],
                    size=style["xaxis_fontsize"],
                    color=xaxis_fontcolor,
                ),
            ),
            tickfont=tickfont(style),
            categoryorder="array",
            categoryarray=labels,
        ),
        yaxis=dict(
            title=dict(
                text=yaxis_title,
                font=dict(
                    family=style["yaxis_font"],
                    size=style["yaxis_fontsize"],
                    color=yaxis_fontcolor,
                ),
            ),
            tickfont=tickfont(style),
            categoryorder="array",
            categoryarray=labels,
            autorange="reversed" if reversed_axis else True,
        ),
        width=style["width"],
        height=style["height"],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    finalise_figure(fig, show, download, download_fpath)


def plot_box(
    data: list[np.ndarray],
    labels: list[str] | None = None,
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
    show: bool = True,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
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
        show: Whether to call fig.show(). Defaults to True for interactive/
            notebook use; set False when calling from an unattended/headless
            context (e.g. an HPC training job), where fig.show() may error
            or hang with no display available.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once (e.g. one
            ".pdf" and one ".html"). Format is inferred per-path from its
            extension.
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
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
    for values, label in zip(data, labels, strict=True):
        fig.add_trace(go.Box(y=to_numpy(values), name=label))
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

    finalise_figure(fig, show, download, download_fpath)
