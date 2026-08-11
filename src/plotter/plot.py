"""matplotlib-based plotting utilities: single-trace 1D/2D/3D plots,
multi-panel subplots, and distribution plots, with optional journal/paper
style presets.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from plotter.paper_presets import resolve_style
from plotter.utils import (
    add_colorbar,
    apply_paper_rcparams,
    finalise_figure,
    resolve_figsize,
    style_axes,
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
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    color: str | None = "black",
    line_width: float = 1,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
) -> None:
    """Plot a single 1D line.

    Args:
        x: X-axis values (numpy array or torch tensor).
        y: Y-axis values (numpy array or torch tensor).
        name: Line label (used for a legend if one is ever added).
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size (points).
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size (points).
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
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
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    ax.plot(to_numpy(x), to_numpy(y), color=color, linewidth=line_width, label=name)
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["title_font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    finalise_figure(fig, interactive, download, download_fpath)


def plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    colorscale: str = "cividis",
    zmin: float | None = None,
    zmax: float | None = None,
    showscale: bool = True,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
) -> None:
    """Plot a single 2D heatmap.

    Args:
        x: X-axis coordinates (numpy array or torch tensor).
        y: Y-axis coordinates (numpy array or torch tensor).
        z: 2D array of values to color-map (numpy array or torch tensor).
        name: Unused label kept for API symmetry with the other plot_*
            functions (matplotlib heatmaps have no hover label).
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size (points).
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size (points).
        tick_fontsize: Tick label font size (points), or None for
            matplotlib's default.
        colorscale: matplotlib colormap name (e.g. "cividis" for sequential
            data, or a diverging one like "PuOr" for data centered at zero
            -- pair a diverging colormap with fixed zmin/zmax).
        zmin: Lower bound of the color range, or None for matplotlib's
            auto-range.
        zmax: Upper bound of the color range, or None for matplotlib's
            auto-range.
        showscale: Whether to draw a colorbar.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
    """
    del name  # kept for signature symmetry with plot_1d; unused here
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
    apply_paper_rcparams(style)

    x_arr, y_arr, z_arr = to_numpy(x), to_numpy(y), to_numpy(z)
    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    im = ax.imshow(
        z_arr, extent=[x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]],
        aspect="auto", origin="lower", cmap=colorscale, vmin=zmin, vmax=zmax,
    )
    if showscale:
        add_colorbar(fig, im, ax, tick_fontsize=style.get("tick_fontsize"))
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["title_font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title, grid=False)

    finalise_figure(fig, interactive, download, download_fpath)


# %% 3D plot builder


def plot_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    zaxis_title: str | None = None,
    zaxis_font: str = "Arial",
    zaxis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    colorscale: str = "cividis",
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    showscale: bool = True,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
) -> None:
    """Plot a single 3D surface.

    Args:
        x: X-axis coordinates, shape (Nx,) (numpy array or torch tensor).
        y: Y-axis coordinates, shape (Ny,).
        z: 2D array of surface heights, shape (Ny, Nx).
        name: Unused label kept for API symmetry with the other plot_*
            functions.
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size (points).
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size (points).
        zaxis_title: Z-axis label.
        zaxis_font: Z-axis label font family.
        zaxis_fontsize: Z-axis label font size (points).
        tick_fontsize: Tick label font size for all three axes (points), or
            None for matplotlib's default.
        colorscale: matplotlib colormap name.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
            Paper presets only define x/y-axis font/size; zaxis_font/
            zaxis_fontsize are never overridden by a preset.
        showscale: Whether to draw the surface's color scale bar.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
    """
    del name
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
    apply_paper_rcparams(style)

    x_arr, y_arr, z_arr = to_numpy(x), to_numpy(y), to_numpy(z)
    xx, yy = np.meshgrid(x_arr, y_arr)
    fig = plt.figure(figsize=resolve_figsize(style), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(xx, yy, z_arr, cmap=colorscale)
    if showscale:
        add_colorbar(fig, surf, ax, tick_fontsize=style.get("tick_fontsize"))
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["title_font"])
    if xaxis_title:
        ax.set_xlabel(xaxis_title, fontsize=style["xaxis_fontsize"], fontfamily=style["xaxis_font"])
    if yaxis_title:
        ax.set_ylabel(yaxis_title, fontsize=style["yaxis_fontsize"], fontfamily=style["yaxis_font"])
    if zaxis_title:
        ax.set_zlabel(zaxis_title, fontsize=zaxis_fontsize, fontfamily=zaxis_font)
    if style.get("tick_fontsize") is not None:
        ax.tick_params(labelsize=style["tick_fontsize"])

    finalise_figure(fig, interactive, download, download_fpath)


# %% Multi-panel and distribution plots


def plot_multi(
    panels: list[dict],
    rows: int,
    cols: int,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    link_x: bool = False,
    link_y: bool = False,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
    return_fig: bool = False,
):
    """Plot a grid of 1D, 2D, and/or 3D panels as subplots.

    Args:
        panels: List of panel dicts, one per subplot, filled row-major into
            the (rows, cols) grid. Each dict has a "kind" of "1d", "2d", or
            "3d" plus that kind's data/labels:
                1d: {"kind": "1d", "x", "y", "name", "xaxis_title",
                     "yaxis_title", "color", "line_width"}. "color" defaults
                     to "black" and "line_width" to 1 if omitted. "name" (if
                     truthy) becomes that panel's own title, drawn above it.
                2d: {"kind": "2d", "x", "y", "z", "name", "xaxis_title",
                     "yaxis_title", "colorscale", "zmin", "zmax",
                     "showscale", "colorbar_showticklabels"}. "colorscale"
                     defaults to "cividis"; "zmin"/"zmax" default to None
                     (matplotlib auto-ranges). "showscale" defaults to True;
                     set False to omit that panel's colorbar (e.g. when an
                     adjacent panel already shows the same scale).
                     "colorbar_showticklabels" defaults to True; set False
                     to show the gradient without numeric tick labels.
                3d: {"kind": "3d", "x", "y", "z", "name", "xaxis_title",
                     "yaxis_title", "zaxis_title", "colorscale",
                     "showscale"}. "showscale" defaults to False since a
                     full-size colorbar per 3D panel tends to crowd a
                     multi-panel grid; set it to True per-panel to opt back
                     in.
        rows: Number of subplot rows.
        cols: Number of subplot columns.
        title: Overall figure title.
        title_font: Title font family.
        title_fontsize: Title font size (points).
        tick_fontsize: Tick label font size for every panel (points), or
            None for matplotlib's default.
        width: Figure width in inches.
        height: Figure height in inches.
        paper: Journal preset key from PAPER_PRESETS, or None. See plot_1d
            for how paper presets interact with the explicit style args.
        link_x: If True, all 1D/2D panels share the same x-axis limits
            (based on the last panel drawn). Has no effect on 3D panels.
        link_y: If True, all 1D/2D panels share the same y-axis limits.
        interactive: Whether to call `plt.show()`. See plot_1d.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
        return_fig: If True, skip show/download and return the built
            `Figure` instead, so the caller can post-process it (e.g. add
            row-label text via `utils.add_row_label`, or tweak an axes'
            ticks directly via `fig.axes[i]`) before saving it themselves.
            Defaults to False (auto show/save, matching every other plot_*
            function's behavior).

    Returns:
        The built matplotlib Figure if return_fig=True, else None.
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
    apply_paper_rcparams(style)

    fig, axes = plt.subplots(
        rows, cols, figsize=resolve_figsize(style), squeeze=False, constrained_layout=True,
    )

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        kind = panel["kind"]

        if kind == "1d":
            ax.plot(
                to_numpy(panel["x"]), to_numpy(panel["y"]),
                color=panel.get("color", "black"), linewidth=panel.get("line_width", 1),
            )
            style_axes(ax, style, xaxis_title=panel.get("xaxis_title"), yaxis_title=panel.get("yaxis_title"))
        elif kind == "2d":
            x_arr, y_arr, z_arr = to_numpy(panel["x"]), to_numpy(panel["y"]), to_numpy(panel["z"])
            im = ax.imshow(
                z_arr, extent=[x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]],
                aspect="auto", origin="lower",
                cmap=panel.get("colorscale", "cividis"),
                vmin=panel.get("zmin"), vmax=panel.get("zmax"),
            )
            if panel.get("showscale", True):
                add_colorbar(
                    fig, im, ax, tick_fontsize=style.get("tick_fontsize"),
                    show_ticklabels=panel.get("colorbar_showticklabels", True),
                )
            style_axes(ax, style, xaxis_title=panel.get("xaxis_title"), yaxis_title=panel.get("yaxis_title"), grid=False)
        else:  # "3d"
            fig.delaxes(ax)
            ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            xx, yy = np.meshgrid(to_numpy(panel["x"]), to_numpy(panel["y"]))
            surf = ax.plot_surface(xx, yy, to_numpy(panel["z"]), cmap=panel.get("colorscale", "cividis"))
            if panel.get("showscale", False):
                add_colorbar(fig, surf, ax, tick_fontsize=style.get("tick_fontsize"))
            if panel.get("xaxis_title"):
                ax.set_xlabel(panel["xaxis_title"], fontsize=style.get("yaxis_fontsize"))
            if panel.get("yaxis_title"):
                ax.set_ylabel(panel["yaxis_title"], fontsize=style.get("yaxis_fontsize"))
            if panel.get("zaxis_title"):
                ax.set_zlabel(panel["zaxis_title"], fontsize=style.get("yaxis_fontsize"))

        name = panel.get("name")
        if name:
            ax.set_title(name, fontsize=style["title_fontsize"], fontfamily=style["title_font"])

    if link_x:
        ref = axes[0][0]
        for ax in fig.axes:
            if ax is not ref and ax.name != "3d":
                ax.sharex(ref)
    if link_y:
        ref = axes[0][0]
        for ax in fig.axes:
            if ax is not ref and ax.name != "3d":
                ax.sharey(ref)

    if title:
        fig.suptitle(title, fontsize=style["title_fontsize"], fontfamily=style["title_font"])

    if return_fig:
        return fig

    finalise_figure(fig, interactive, download, download_fpath)
    return None


def plot_1d_multi(
    series: list[tuple[np.ndarray, np.ndarray, str]],
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
    download: bool = False,
    download_fpath: str | list[str] | None = None,
) -> None:
    """Plot multiple 1D lines together on one shared figure.

    Args:
        series: List of (x, y, name) tuples, one per line; x/y may be numpy
            arrays or torch tensors. Each line gets a distinct color from
            matplotlib's default color cycle.
        title: Figure title.
        title_font: Title font family.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size (points).
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size (points).
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
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    for x, y, name in series:
        ax.plot(to_numpy(x), to_numpy(y), label=name)
    if series:
        ax.legend(fontsize=style.get("tick_fontsize"))
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["title_font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    finalise_figure(fig, interactive, download, download_fpath)


def plot_violin(
    data: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
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
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size (points).
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size (points).
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
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    ax.violinplot([to_numpy(values) for values in data], showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["title_font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    finalise_figure(fig, interactive, download, download_fpath)


def plot_box(
    data: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    title_font: str = "Arial",
    title_fontsize: int = 24,
    xaxis_title: str | None = None,
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    yaxis_title: str | None = None,
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
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
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size (points).
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size (points).
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
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    ax.boxplot([to_numpy(values) for values in data], tick_labels=labels)
    if title:
        ax.set_title(title, fontsize=style["title_fontsize"], fontfamily=style["title_font"])
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    finalise_figure(fig, interactive, download, download_fpath)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str | None = None,
    normalize: bool = True,
    reversed_axis: bool = True,
    colorscale: str = "Blues",
    title_font: str | None = None,
    title_fontsize: int = 16,
    xaxis_title: str | None = "Predicted",
    xaxis_font: str = "Arial",
    xaxis_fontsize: int = 16,
    yaxis_title: str | None = "True",
    yaxis_font: str = "Arial",
    yaxis_fontsize: int = 16,
    tick_fontsize: int | None = None,
    width: float | None = None,
    height: float | None = None,
    paper: str | None = None,
    interactive: bool = False,
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
        reversed_axis: If True (default), the true-class axis runs top
            (first class) to bottom, so the diagonal runs from top-left to
            bottom-right -- the conventional confusion-matrix layout.
        colorscale: matplotlib colormap name.
        title_font: Title font family. Defaults to None (matplotlib's own
            default font, unstyled) rather than "Arial" like the other
            plot_* functions, so a bare call reproduces a plain title. Pass
            an explicit family (or a `paper` preset) to style it.
        title_fontsize: Title font size (points).
        xaxis_title: X-axis label.
        xaxis_font: X-axis label font family.
        xaxis_fontsize: X-axis label font size (points).
        yaxis_title: Y-axis label.
        yaxis_font: Y-axis label font family.
        yaxis_fontsize: Y-axis label font size (points).
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
    """
    cm = to_numpy(cm).astype(float)
    labels = [str(l) for l in labels]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        z = np.where(row_sums > 0, cm / row_sums, 0.0)
        zmin, zmax = 0, 1
    else:
        z = cm
        zmin, zmax = None, None

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
    apply_paper_rcparams(style)

    fig, ax = plt.subplots(figsize=resolve_figsize(style), constrained_layout=True)
    origin = "upper" if reversed_axis else "lower"
    im = ax.imshow(z, cmap=colorscale, vmin=zmin, vmax=zmax, origin=origin, aspect="auto")
    cbar = add_colorbar(fig, im, ax, tick_fontsize=style.get("tick_fontsize"))
    if normalize:
        cbar.set_label("Recall")
        cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    else:
        cbar.set_label("Count")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(labelsize=style.get("tick_fontsize"))

    if title:
        title_kwargs = dict(fontsize=style["title_fontsize"])
        if style.get("title_font"):
            title_kwargs["fontfamily"] = style["title_font"]
        ax.set_title(title, **title_kwargs)
    if xaxis_title:
        ax.set_xlabel(xaxis_title, fontsize=style["xaxis_fontsize"], fontfamily=style["xaxis_font"])
    if yaxis_title:
        ax.set_ylabel(yaxis_title, fontsize=style["yaxis_fontsize"], fontfamily=style["yaxis_font"])

    finalise_figure(fig, interactive, download, download_fpath)
