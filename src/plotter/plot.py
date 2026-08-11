"""matplotlib-based plotting utilities: single-trace 1D/2D/3D plots,
multi-panel subplots, and distribution plots, with optional journal/paper
style presets.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from plotter.paper_presets import resolve_style
from plotter.utils import (
    apply_paper_rcparams,
    apply_ticks,
    draw_1d,
    draw_2d,
    draw_3d,
    finalise_figure,
    resolve_figsize,
    style_axes,
    to_numpy,
)

# %% Core 1D / 2D plot builders


def plot_1d(
    x: np.ndarray,
    y: np.ndarray,
    title: str | None = None,
    font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    color: str | None = "black",
    line_width: float = 0.3,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
) -> None:
    """Plot a single 1D line.

    Args:
        x: X-axis values (numpy array or torch tensor).
        y: Y-axis values (numpy array or torch tensor).
        title: Figure title.
        font: Font family for the title and axis labels.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        axis_fontsize: X/Y axis label font size (points), shared by both.
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
        xticks: X-axis tick positions, or None for matplotlib's default.
        xticklabels: X-axis tick labels (same length as xticks), or None to
            show the tick values themselves.
        yticks: Y-axis tick positions, or None for matplotlib's default.
        yticklabels: Y-axis tick labels (same length as yticks), or None to
            show the tick values themselves.
        xlim: (min, max) x-axis range, or None for matplotlib's auto-range.
        ylim: (min, max) y-axis range, or None for matplotlib's auto-range.
        color: Line color.
        line_width: Line width (points).
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS (e.g. "TASLP_single"),
            or None to use the explicit style args above as-is. When set,
            the preset's font/size/figure-width values override the
            corresponding explicit args.
        interactive: Whether to call `plt.show()`. `plotter` never selects a
            backend itself; this assumes an interactive backend is already
            active when True. Leave False in headless/HPC contexts.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        dpi: Override for the package's default save DPI on this call only,
            or None to use it (`plotter.utils.SAVE_DPI`, 1200). Lower this
            for very large raster content (e.g. a heatmap with hundreds of
            cells per side) where the default would produce an unreasonably
            large file.
    """
    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    draw_1d(
        ax, style, x, y,
        color=color, line_width=line_width,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        xticks=xticks, xticklabels=xticklabels, yticks=yticks, yticklabels=yticklabels,
        xlim=xlim, ylim=ylim,
    )
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["font"])

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)


def plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str | None = None,
    font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    colorscale: str = "cividis",
    zmin: float | None = None,
    zmax: float | None = None,
    show_colorbar: bool = True,
    colorbar_showticklabels: bool = True,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
) -> None:
    """Plot a single 2D heatmap.

    Args:
        x: X-axis coordinates (numpy array or torch tensor).
        y: Y-axis coordinates (numpy array or torch tensor).
        z: 2D array of values to color-map (numpy array or torch tensor).
        title: Figure title.
        font: Font family for the title and axis labels.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        axis_fontsize: X/Y axis label font size (points), shared by both.
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
        xticks: X-axis tick positions, or None for matplotlib's default.
        xticklabels: X-axis tick labels (same length as xticks), or None to
            show the tick values themselves.
        yticks: Y-axis tick positions, or None for matplotlib's default.
        yticklabels: Y-axis tick labels (same length as yticks), or None to
            show the tick values themselves.
        xlim: (min, max) x-axis range, or None for matplotlib's auto-range.
        ylim: (min, max) y-axis range, or None for matplotlib's auto-range.
        colorscale: matplotlib colormap name (e.g. "cividis" for sequential
            data, or a diverging one like "PuOr" for data centered at zero
            -- pair a diverging colormap with fixed zmin/zmax).
        zmin: Lower bound of the color range, or None for matplotlib's
            auto-range.
        zmax: Upper bound of the color range, or None for matplotlib's
            auto-range.
        show_colorbar: Whether to draw a colorbar.
        colorbar_showticklabels: Whether the colorbar shows numeric tick
            labels (the gradient itself is always shown).
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        dpi: Override for the package's default save DPI on this call only,
            or None to use it (`plotter.utils.SAVE_DPI`, 1200). Lower this
            for very large raster content (e.g. a heatmap with hundreds of
            cells per side) where the default would produce an unreasonably
            large file.
    """
    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    draw_2d(
        fig, ax, style, z, x=x, y=y,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        show_colorbar=show_colorbar, colorbar_showticklabels=colorbar_showticklabels,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        xticks=xticks, xticklabels=xticklabels, yticks=yticks, yticklabels=yticklabels,
        xlim=xlim, ylim=ylim,
    )
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["font"])

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)


# %% 3D plot builder


def plot_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str | None = None,
    font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    zaxis_title: str | None = None,
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    elev: float | None = None,
    azim: float | None = None,
    roll: float | None = None,
    colorscale: str = "cividis",
    show_colorbar: bool = True,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
) -> None:
    """Plot a single 3D surface.

    Args:
        x: X-axis coordinates, shape (Nx,) (numpy array or torch tensor).
        y: Y-axis coordinates, shape (Ny,).
        z: 2D array of surface heights, shape (Ny, Nx).
        title: Figure title.
        font: Font family for the title and axis labels.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        zaxis_title: Z-axis label.
        axis_fontsize: X/Y/Z axis label font size (points), shared by all
            three.
        tick_fontsize: Tick label font size for all three axes (points), or
            None for matplotlib's default.
        xlim: (min, max) x-axis range, or None for matplotlib's auto-range.
        ylim: (min, max) y-axis range, or None for matplotlib's auto-range.
        zlim: (min, max) z-axis range, or None for matplotlib's auto-range.
        elev: Camera elevation angle in degrees, or None for matplotlib's
            default (30).
        azim: Camera azimuth angle in degrees, or None for matplotlib's
            default (-60). Controls which axis reads left-to-right --
            e.g. azim=-90 puts x front-to-back and y left-to-right.
        roll: Camera roll angle in degrees, or None for matplotlib's
            default (0).
        colorscale: matplotlib colormap name.
        show_colorbar: Whether to draw the surface's color scale bar.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        dpi: Override for the package's default save DPI on this call only,
            or None to use it (`plotter.utils.SAVE_DPI`, 1200). Lower this
            for very large raster content (e.g. a heatmap with hundreds of
            cells per side) where the default would produce an unreasonably
            large file.
    """
    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig = plt.figure(figsize=resolve_figsize(style), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    draw_3d(
        fig, ax, style, x, y, z,
        colorscale=colorscale, show_colorbar=show_colorbar,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=zaxis_title,
        xlim=xlim, ylim=ylim, zlim=zlim,
        elev=elev, azim=azim, roll=roll,
    )
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["font"])

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)


# %% Multi-panel and distribution plots


def plot_multi(
    panels: list[dict],
    rows: int,
    cols: int,
    title: str | None = None,
    font: str = "Arial",
    title_fontsize: int = 24,
    subplot_title_fontsize: int = 16,
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
    return_fig: bool = False,
):
    """Plot a grid of 1D, 2D, and/or 3D panels as subplots.

    Draws each panel with the same `draw_1d`/`draw_2d`/`draw_3d` helpers
    used by the single-panel `plot_1d`/`plot_2d`/`plot_3d` functions, so
    panel behavior (ticks, limits, colorbar sizing, axis fonts) stays
    consistent with the single-plot versions.

    Args:
        panels: List of panel dicts, one per subplot, filled row-major into
            the (rows, cols) grid. Each dict has a "kind" of "1d", "2d", or
            "3d" plus that kind's data/labels:
                1d: {"kind": "1d", "x", "y", "name", "xaxis_title",
                     "yaxis_title", "color", "line_width", "xticks",
                     "xticklabels", "yticks", "yticklabels", "xlim", "ylim"}.
                     "color" defaults to "black" and "line_width" to 0.3 if
                     omitted. "name" (if truthy) becomes that panel's own
                     title, drawn above it at `subplot_title_fontsize`.
                2d: {"kind": "2d", "x", "y", "z", "name", "xaxis_title",
                     "yaxis_title", "colorscale", "zmin", "zmax",
                     "show_colorbar", "colorbar_showticklabels", "xticks",
                     "xticklabels", "yticks", "yticklabels", "xlim", "ylim"}.
                     "colorscale" defaults to "cividis"; "zmin"/"zmax"
                     default to None (matplotlib auto-ranges).
                     "show_colorbar" defaults to True; set False to omit
                     that panel's colorbar (e.g. when an adjacent panel
                     already shows the same scale). "colorbar_showticklabels"
                     defaults to True; set False to show the gradient
                     without numeric tick labels.
                3d: {"kind": "3d", "x", "y", "z", "name", "xaxis_title",
                     "yaxis_title", "zaxis_title", "colorscale",
                     "show_colorbar", "xlim", "ylim", "zlim", "elev",
                     "azim", "roll"}.
                     "show_colorbar" defaults to False since a full-size
                     colorbar per 3D panel tends to crowd a multi-panel
                     grid; set it to True per-panel to opt back in.
                     "elev"/"azim"/"roll" set the camera angle (degrees),
                     each defaulting to matplotlib's own default (30/-60/0)
                     if omitted -- see `plot_3d`'s docstring.
        rows: Number of subplot rows.
        cols: Number of subplot columns.
        title: Overall figure title.
        font: Font family for the title and axis labels.
        title_fontsize: Overall figure title font size (points).
        subplot_title_fontsize: Per-panel title font size (points), for
            panels with a truthy "name". Separate from `title_fontsize` so
            panel titles can be sized independently from the overall figure
            title.
        axis_fontsize: Axis label font size (points), shared by every
            panel's x/y (and z, for 3D panels) labels.
        tick_fontsize: Tick label font size for every panel (points), or
            None for matplotlib's default.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        return_fig: If True, skip show/download and return the built
            `Figure` instead, so the caller can post-process it (e.g. tweak
            an axes' ticks directly via `fig.axes[i]`) before saving it
            themselves. Defaults to False (auto show/save, matching every
            other plot_* function's behavior).

    Returns:
        The built matplotlib Figure if return_fig=True, else None.
    """
    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig, axes = plt.subplots(
        rows, cols, figsize=resolve_figsize(style),
        squeeze=False, constrained_layout=True,
    )

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        kind = panel["kind"]

        if kind == "1d":
            draw_1d(
                ax, style, panel["x"], panel["y"],
                color=panel.get("color", "black"),
                line_width=panel.get("line_width", 0.3),
                xaxis_title=panel.get("xaxis_title"),
                yaxis_title=panel.get("yaxis_title"),
                xticks=panel.get("xticks"), xticklabels=panel.get("xticklabels"),
                yticks=panel.get("yticks"), yticklabels=panel.get("yticklabels"),
                xlim=panel.get("xlim"), ylim=panel.get("ylim"),
            )
        elif kind == "2d":
            draw_2d(
                fig, ax, style, panel["z"], x=panel["x"], y=panel["y"],
                colorscale=panel.get("colorscale", "cividis"),
                zmin=panel.get("zmin"), zmax=panel.get("zmax"),
                show_colorbar=panel.get("show_colorbar", True),
                colorbar_showticklabels=panel.get("colorbar_showticklabels", True),
                xaxis_title=panel.get("xaxis_title"),
                yaxis_title=panel.get("yaxis_title"),
                xticks=panel.get("xticks"), xticklabels=panel.get("xticklabels"),
                yticks=panel.get("yticks"), yticklabels=panel.get("yticklabels"),
                xlim=panel.get("xlim"), ylim=panel.get("ylim"),
            )
        else:  # "3d"
            fig.delaxes(ax)
            ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            draw_3d(
                fig, ax, style, panel["x"], panel["y"], panel["z"],
                colorscale=panel.get("colorscale", "cividis"),
                show_colorbar=panel.get("show_colorbar", False),
                xaxis_title=panel.get("xaxis_title"),
                yaxis_title=panel.get("yaxis_title"),
                zaxis_title=panel.get("zaxis_title"),
                xlim=panel.get("xlim"), ylim=panel.get("ylim"), zlim=panel.get("zlim"),
                elev=panel.get("elev"), azim=panel.get("azim"), roll=panel.get("roll"),
            )

        name = panel.get("name")
        if name:
            ax.set_title(
                name, fontsize=subplot_title_fontsize, fontfamily=style["font"]
            )

    if title:
        fig.suptitle(title, fontsize=style["title_fontsize"], fontfamily=style["font"])

    if return_fig:
        return fig

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)
    return None


def plot_1d_multi(
    series: list[tuple[np.ndarray, np.ndarray, str]],
    title: str | None = None,
    font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
) -> None:
    """Plot multiple 1D lines together on one shared figure.

    Args:
        series: List of (x, y, name) tuples, one per line; x/y may be numpy
            arrays or torch tensors. Each line gets a distinct color from
            matplotlib's default color cycle, and `name` becomes its legend
            label.
        title: Figure title.
        font: Font family for the title and axis labels.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        axis_fontsize: X/Y axis label font size (points), shared by both.
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
        xticks: X-axis tick positions, or None for matplotlib's default.
        xticklabels: X-axis tick labels (same length as xticks), or None to
            show the tick values themselves.
        yticks: Y-axis tick positions, or None for matplotlib's default.
        yticklabels: Y-axis tick labels (same length as yticks), or None to
            show the tick values themselves.
        xlim: (min, max) x-axis range, or None for matplotlib's auto-range.
        ylim: (min, max) y-axis range, or None for matplotlib's auto-range.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        dpi: Override for the package's default save DPI on this call only,
            or None to use it (`plotter.utils.SAVE_DPI`, 1200). Lower this
            for very large raster content (e.g. a heatmap with hundreds of
            cells per side) where the default would produce an unreasonably
            large file.
    """
    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    for x, y, name in series:
        draw_1d(ax, style, x, y, color=None, label=name)
    if series:
        ax.legend(fontsize=style.get("tick_fontsize"))
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    apply_ticks(ax, xticks, xticklabels, yticks, yticklabels)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)


def plot_violin(
    data: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
) -> None:
    """Plot one or more distributions as violin plots.

    Args:
        data: List of 1D arrays (numpy or torch), one distribution per violin.
        labels: Name for each violin, one per entry in `data`. Defaults to
            "Group 1", "Group 2", etc.
        title: Figure title.
        font: Font family for the title and axis labels.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        axis_fontsize: X/Y axis label font size (points), shared by both.
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        dpi: Override for the package's default save DPI on this call only,
            or None to use it (`plotter.utils.SAVE_DPI`, 1200). Lower this
            for very large raster content (e.g. a heatmap with hundreds of
            cells per side) where the default would produce an unreasonably
            large file.
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    ax.violinplot([to_numpy(values) for values in data], showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)


def plot_box(
    data: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
) -> None:
    """Plot one or more distributions as box plots.

    Args:
        data: List of 1D arrays (numpy or torch), one distribution per box.
        labels: Name for each box, one per entry in `data`. Defaults to
            "Group 1", "Group 2", etc.
        title: Figure title.
        font: Font family for the title and axis labels.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        axis_fontsize: X/Y axis label font size (points), shared by both.
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        dpi: Override for the package's default save DPI on this call only,
            or None to use it (`plotter.utils.SAVE_DPI`, 1200). Lower this
            for very large raster content (e.g. a heatmap with hundreds of
            cells per side) where the default would produce an unreasonably
            large file.
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    ax.boxplot([to_numpy(values) for values in data], tick_labels=labels)
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str | None = None,
    normalize: bool = True,
    reversed_axis: bool = True,
    colorscale: str = "Blues",
    font: str | None = None,
    title_fontsize: int = 16,
    xaxis_title: str | None = "Predicted",
    yaxis_title: str | None = "True",
    axis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    dpi: int | None = None,
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
        reversed_axis: If True (default), the true-class axis runs top
            (first class) to bottom, so the diagonal runs from top-left to
            bottom-right -- the conventional confusion-matrix layout.
        colorscale: matplotlib colormap name.
        font: Font family for the title and axis labels. Defaults to None
            (matplotlib's own default font, unstyled) rather than "Arial"
            like the other plot_* functions, so a bare call reproduces a
            plain figure. Pass an explicit family (or a `paper` preset) to
            style it.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        axis_fontsize: X/Y axis label font size (points), shared by both.
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        dpi: Override for the package's default save DPI on this call only,
            or None to use it (`plotter.utils.SAVE_DPI`, 1200). Lower this
            for very large raster content (e.g. a heatmap with hundreds of
            cells per side) where the default would produce an unreasonably
            large file.
    """
    cm = to_numpy(cm).astype(float)
    labels = [str(label) for label in labels]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        z = np.where(row_sums > 0, cm / row_sums, 0.0)
        zmin, zmax = 0, 1
    else:
        z = cm
        zmin, zmax = None, None

    style = resolve_style(
        paper,
        font=font,
        title_fontsize=title_fontsize,
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        width=width,
        height=height,
    )
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    origin = "upper" if reversed_axis else "lower"
    im, cbar = draw_2d(
        fig, ax, style, z, colorscale=colorscale, zmin=zmin, zmax=zmax, origin=origin,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        xticks=range(len(labels)), yticks=range(len(labels)), yticklabels=labels,
    )
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if normalize:
        cbar.set_label("Recall")
        cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    else:
        cbar.set_label("Count")

    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["font"])

    finalise_figure(fig, interactive, download, download_fpath, dpi=dpi)
