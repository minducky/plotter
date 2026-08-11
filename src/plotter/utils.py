"""Shared matplotlib figure-building and styling helpers for `plotter`."""

import os
import warnings

import matplotlib.figure
import numpy as np


def to_numpy(data) -> np.ndarray:
    """Convert a torch tensor to a squeezed numpy array; pass array-likes through.

    Args:
        data: A numpy array, torch tensor, or other array-like.

    Returns:
        A squeezed numpy array.
    """
    if hasattr(
        data, "detach"
    ):  # duck-types torch.Tensor without a hard torch dependency
        data = data.detach().cpu().numpy()
    return np.squeeze(np.asarray(data))


def apply_paper_rcparams(style: dict) -> None:
    """Sets global font rcParams from a resolved style, before building a figure.

    matplotlib resolves fonts from `plt.rcParams` at draw time, so this must
    be called before `plt.subplots(...)`. Simpler than styling every text
    object individually (which Plotly required).

    Args:
        style: A dict from `resolve_style`, at least containing `font`.
    """
    import matplotlib.pyplot as plt

    family = style.get("font")
    if family is None:
        return
    plt.rcParams["font.family"] = "serif" if "Times" in family else "sans-serif"
    plt.rcParams["font.serif"] = [family, "Times", "DejaVu Serif"]
    plt.rcParams["font.sans-serif"] = [family, "Arial", "DejaVu Sans"]


def style_axes(
    ax,
    style: dict,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    grid: bool = True,
) -> None:
    """Applies title/label fonts, tick styling, grid, and a visible border to an Axes.

    matplotlib draws all four spines by default, so -- unlike the Plotly
    implementation -- no separate "add a border" step is needed here; this
    only has to *style* what's already there.

    Args:
        ax: Axes to style.
        style: A dict from `resolve_style`.
        xaxis_title: X-axis label text, or None to leave unset.
        yaxis_title: Y-axis label text, or None to leave unset.
        grid: Whether to draw a light gridline behind the data.
    """
    font = style.get("font")
    axis_fontsize = style.get("axis_fontsize")
    if xaxis_title:
        ax.set_xlabel(xaxis_title, fontsize=axis_fontsize, fontfamily=font)
    if yaxis_title:
        ax.set_ylabel(yaxis_title, fontsize=axis_fontsize, fontfamily=font)
    tick_fontsize = style.get("tick_fontsize")
    ax.tick_params(direction="out", length=2, width=0.5, labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if grid:
        ax.grid(True, color="lightgray", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


DEFAULT_FIGSIZE = (6.4, 4.8)  # matplotlib's own default, used when neither paper= nor width/height is given


def resolve_figsize(style: dict) -> tuple[float, float]:
    """Returns (width_in, height_in) from a resolved style, falling back to
    matplotlib's default for whichever of width/height wasn't set."""
    width = style.get("width") or DEFAULT_FIGSIZE[0]
    height = style.get("height") or DEFAULT_FIGSIZE[1]
    return width, height


def add_colorbar(fig, im, ax, tick_fontsize: float | None = None, show_ticklabels: bool = True):
    """Adds a colorbar sized to match its target Axes' actual rendered height.

    This is the direct replacement for the whole colorbar position/length
    dance the Plotly implementation needed: `fig.colorbar(im, ax=ax, ...)`
    reads `ax`'s real bounding box, so the colorbar always matches it.

    Args:
        fig: Figure the colorbar belongs to.
        im: The image/mappable returned by imshow/pcolormesh/etc.
        ax: The Axes the colorbar is attached to.
        tick_fontsize: Font size for the colorbar's tick labels, or None.
        show_ticklabels: Whether to show numeric tick labels on the colorbar
            (the gradient itself is always shown).

    Returns:
        The created Colorbar.
    """
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.5)
    if tick_fontsize is not None:
        cbar.ax.tick_params(labelsize=tick_fontsize)
    if not show_ticklabels:
        cbar.set_ticks([])
    return cbar


def apply_ticks(
    ax,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
) -> None:
    """Sets explicit tick positions/labels on an Axes, leaving matplotlib's
    auto-generated ticks alone for whichever of the four args is None.

    Args:
        ax: Axes to set ticks on.
        xticks: X-axis tick positions, or None to leave matplotlib's default.
        xticklabels: X-axis tick labels (same length as xticks), or None to
            show the tick values themselves.
        yticks: Y-axis tick positions, or None to leave matplotlib's default.
        yticklabels: Y-axis tick labels (same length as yticks), or None to
            show the tick values themselves.
    """
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)


def draw_1d(
    ax,
    style: dict,
    x,
    y,
    color: str | None = "black",
    line_width: float = 1,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    label: str | None = None,
):
    """Draws a single 1D line onto an existing Axes.

    Shared by `plot_1d` (its own Axes) and `plot_multi`/`plot_1d_multi`
    (an Axes from a grid/shared figure), so tick/limit/labeling behavior
    stays identical everywhere a line gets drawn.

    Args:
        ax: Axes to draw onto.
        style: A dict from `resolve_style`.
        x: X-axis values (numpy array or torch tensor).
        y: Y-axis values (numpy array or torch tensor).
        color: Line color.
        line_width: Line width (points).
        xaxis_title: X-axis label, or None to leave unset.
        yaxis_title: Y-axis label, or None to leave unset.
        xticks: X-axis tick positions, or None for matplotlib's default.
        xticklabels: X-axis tick labels, or None to show tick values.
        yticks: Y-axis tick positions, or None for matplotlib's default.
        yticklabels: Y-axis tick labels, or None to show tick values.
        xlim: (min, max) x-axis range, or None for matplotlib's auto-range.
        ylim: (min, max) y-axis range, or None for matplotlib's auto-range.
        label: Legend label for this line, or None. Caller is responsible
            for calling `ax.legend()` if a legend should actually be shown.

    Returns:
        The Line2D created by `ax.plot`.
    """
    (line,) = ax.plot(
        to_numpy(x), to_numpy(y), color=color, linewidth=line_width, label=label
    )
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    apply_ticks(ax, xticks, xticklabels, yticks, yticklabels)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return line


def draw_2d(
    fig,
    ax,
    style: dict,
    z,
    x=None,
    y=None,
    colorscale: str = "cividis",
    zmin: float | None = None,
    zmax: float | None = None,
    show_colorbar: bool = True,
    colorbar_showticklabels: bool = True,
    origin: str = "lower",
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Draws a single 2D heatmap (+ optional colorbar) onto an existing Axes.

    Shared by `plot_2d` (its own Axes), `plot_multi` (an Axes from a grid),
    and `plot_confusion_matrix` (no x/y coordinates, just pixel indices).

    `interpolation="nearest"` is deliberate: imshow's default resampling
    (rcParams `image.interpolation`, "antialiased" on recent matplotlib)
    blends adjacent data cells together when the rendered raster resolution
    doesn't line up with `z`'s own resolution -- which for a cochleagram
    (tens of thousands of time samples squeezed into a couple inches) it
    essentially never does. That blending is what made zoomed-in PDFs look
    soft/merged instead of showing crisp per-cell structure the way the
    package's previous Plotly backend did. "nearest" keeps whatever cells
    do get rendered sharp, and pairs with `save_figure`'s higher DPI to
    retain as much of that per-cell detail as practical.

    Args:
        fig: Figure `ax` belongs to (needed to attach the colorbar).
        ax: Axes to draw onto.
        style: A dict from `resolve_style`.
        z: 2D array of values to color-map (numpy array or torch tensor).
        x: X-axis coordinates, or None to imshow by pixel index (no
            `extent`) -- used by `plot_confusion_matrix`.
        y: Y-axis coordinates, or None. Must be given iff `x` is given.
        colorscale: matplotlib colormap name.
        zmin: Lower bound of the color range, or None for auto-range.
        zmax: Upper bound of the color range, or None for auto-range.
        show_colorbar: Whether to draw a colorbar.
        colorbar_showticklabels: Whether the colorbar shows numeric tick
            labels (the gradient itself is always shown).
        origin: "lower" (default) or "upper" -- passed straight to imshow.
        xaxis_title: X-axis label, or None to leave unset.
        yaxis_title: Y-axis label, or None to leave unset.
        xticks: X-axis tick positions, or None for matplotlib's default.
        xticklabels: X-axis tick labels, or None to show tick values.
        yticks: Y-axis tick positions, or None for matplotlib's default.
        yticklabels: Y-axis tick labels, or None to show tick values.
        xlim: (min, max) x-axis range, or None for matplotlib's auto-range.
        ylim: (min, max) y-axis range, or None for matplotlib's auto-range.

    Returns:
        A (im, cbar) tuple -- `im` is the AxesImage from imshow; `cbar` is
        the created Colorbar, or None if show_colorbar=False.
    """
    z_arr = to_numpy(z)
    extent = None
    if x is not None:
        x_arr, y_arr = to_numpy(x), to_numpy(y)
        extent = [x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]]
    im = ax.imshow(
        z_arr, extent=extent, cmap=colorscale, vmin=zmin, vmax=zmax,
        origin=origin, aspect="auto", interpolation="nearest",
    )
    cbar = None
    if show_colorbar:
        cbar = add_colorbar(
            fig, im, ax,
            tick_fontsize=style.get("tick_fontsize"),
            show_ticklabels=colorbar_showticklabels,
        )
    style_axes(ax, style, xaxis_title=xaxis_title, yaxis_title=yaxis_title, grid=False)
    apply_ticks(ax, xticks, xticklabels, yticks, yticklabels)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return im, cbar


def draw_3d(
    fig,
    ax,
    style: dict,
    x,
    y,
    z,
    colorscale: str = "cividis",
    show_colorbar: bool = True,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    zaxis_title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    elev: float | None = None,
    azim: float | None = None,
    roll: float | None = None,
):
    """Draws a single 3D surface (+ optional colorbar) onto an existing 3D Axes.

    Shared by `plot_3d` (its own Axes) and `plot_multi` (an Axes from a
    grid). All three axis labels share one `style["axis_fontsize"]`, same as
    `style_axes` does for the x/y axes of 1D/2D plots.

    Args:
        fig: Figure `ax` belongs to (needed to attach the colorbar).
        ax: 3D Axes to draw onto.
        style: A dict from `resolve_style`; `font`, `axis_fontsize`, and
            `tick_fontsize` are used here.
        x: X-axis coordinates, shape (Nx,).
        y: Y-axis coordinates, shape (Ny,).
        z: 2D array of surface heights, shape (Ny, Nx).
        colorscale: matplotlib colormap name.
        show_colorbar: Whether to draw the surface's color scale bar.
        xaxis_title: X-axis label, or None to leave unset.
        yaxis_title: Y-axis label, or None to leave unset.
        zaxis_title: Z-axis label, or None to leave unset.
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

    Returns:
        A (surf, cbar) tuple -- `surf` is the Poly3DCollection from
        plot_surface; `cbar` is the created Colorbar, or None if
        show_colorbar=False.
    """
    xx, yy = np.meshgrid(to_numpy(x), to_numpy(y))
    surf = ax.plot_surface(xx, yy, to_numpy(z), cmap=colorscale)
    cbar = None
    if show_colorbar:
        cbar = add_colorbar(fig, surf, ax, tick_fontsize=style.get("tick_fontsize"))
    font = style.get("font")
    axis_fontsize = style.get("axis_fontsize")
    if xaxis_title:
        ax.set_xlabel(xaxis_title, fontsize=axis_fontsize, fontfamily=font)
    if yaxis_title:
        ax.set_ylabel(yaxis_title, fontsize=axis_fontsize, fontfamily=font)
    if zaxis_title:
        ax.set_zlabel(zaxis_title, fontsize=axis_fontsize, fontfamily=font)
    if style.get("tick_fontsize") is not None:
        ax.tick_params(labelsize=style["tick_fontsize"])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    if elev is not None or azim is not None or roll is not None:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    return surf, cbar


def reserve_left_margin(fig: matplotlib.figure.Figure, margin: float = 0.08) -> None:
    """Reserves figure-fraction space on the left, inside the figure's own
    bounds, for row labels added via `add_row_label`.

    Without this, a row label placed at a negative figure-fraction x sits
    outside the figure's own canvas -- fine as long as you save with
    `bbox_inches='tight'`, except tight-bbox then *expands the saved page*
    to include it, silently breaking the exact physical page size a `paper=`
    preset promised. Reserving margin up front keeps the label (and the
    save) inside the original `figsize`, so `bbox_inches='tight'` never
    needs to grow the canvas.

    Only has an effect when the figure uses matplotlib's constrained-layout
    engine (i.e. was created with `constrained_layout=True`); harmless
    no-op otherwise. Safe to call more than once (idempotent for the same
    `margin`).

    Args:
        fig: Figure to reserve margin on.
        margin: Fraction of the figure width to reserve on the left.
    """
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(rect=(margin, 0, 1 - margin, 1))


def add_row_label(
    fig: matplotlib.figure.Figure,
    ax,
    label: str,
    x: float = 0.02,
    fontsize: float = 9,
    fontfamily: str | None = None,
) -> None:
    """Adds a vertically-centered row-label (e.g. "Left"/"Right"), aligned
    to `ax`'s actual rendered vertical extent.

    Forces a layout pass first (harmless if one already ran, e.g. via
    `constrained_layout=True`) so `ax.get_position()` reflects where the
    Axes really ends up -- unlike a domain fraction read before layout,
    this can't drift out of sync with the final render. Call
    `reserve_left_margin` first so `x` (a small positive fraction) lands in
    space actually set aside for it, rather than needing a tight-bbox
    canvas expansion to avoid clipping.

    Args:
        fig: The figure `ax` belongs to.
        ax: The Axes to vertically center the label on.
        label: Text to show, rotated 90 degrees.
        x: Figure-fraction x position. Small and positive, meant to sit
            inside the margin reserved by `reserve_left_margin`.
        fontsize: Label font size (points).
        fontfamily: Label font family, or None for matplotlib's current
            rcParams default.
    """
    fig.canvas.draw()
    pos = ax.get_position()
    y_center = (pos.y0 + pos.y1) / 2
    kwargs = dict(fontsize=fontsize)
    if fontfamily:
        kwargs["fontfamily"] = fontfamily
    fig.text(x, y_center, label, rotation=90, va="center", ha="center", fontweight="bold", **kwargs)


SAVE_DPI = 600  # resolution for rasterized content (imshow heatmaps) embedded
# in a saved PDF/PNG/SVG. Vector content (lines, text, axes) is unaffected
# by this -- it's only the raster images that get sampled at this density.
# 300 looked visibly softer/blockier than the old Plotly backend when
# zoomed in on a heatmap panel; 600 is a practical middle ground between
# that and the (impractically large) DPI needed to embed a cochleagram's
# full native sample-rate resolution.


def save_figure(fig: matplotlib.figure.Figure, download_fpath: str) -> None:
    """Save a figure to disk, choosing the writer by file extension.

    ".pdf"/".png"/".svg" are written via `fig.savefig(..., dpi=SAVE_DPI)`
    with NO `bbox_inches='tight'` -- every figure this package builds uses
    `constrained_layout=True`, which already keeps titles/labels/colorbars/
    row-label text (see `reserve_left_margin`) within the figure's own
    bounds, so the saved page is always *exactly* `figsize` with no
    renderer-specific DPI conversion to get wrong (unlike the old Plotly/
    Kaleido implementation) and no tight-bbox cropping/expansion to make it
    inexact. ".html" has no first-class matplotlib equivalent to Plotly's
    interactive export, so it's skipped with a warning rather than silently
    doing nothing or raising.

    Args:
        fig: Figure to save.
        download_fpath: Output path; its extension determines the format.
    """
    ext = os.path.splitext(download_fpath)[1].lower()
    if ext == ".html":
        warnings.warn(
            f"save_figure: skipping {download_fpath!r} -- matplotlib has no "
            "interactive HTML export; use interactive=True to view instead.",
            stacklevel=2,
        )
        return
    fig.savefig(download_fpath, dpi=SAVE_DPI)


def save_figure_multi(fig: matplotlib.figure.Figure, download_fpaths: list[str]) -> None:
    """Save a figure to multiple paths at once, format inferred per-path.

    Args:
        fig: Figure to save.
        download_fpaths: Output paths (e.g. one ".pdf" and one ".png").
            Each path's format is inferred from its extension, same as
            `save_figure`.
    """
    for fpath in download_fpaths:
        save_figure(fig, fpath)


def finalise_figure(
    fig: matplotlib.figure.Figure,
    interactive: bool,
    download: bool,
    download_fpath: str | list[str] | None,
) -> None:
    """Show and/or save a figure, per the interactive/download/download_fpath flags.

    Args:
        fig: Figure to show/save.
        interactive: Whether to call `plt.show()`. `plotter` never selects a
            backend itself (no `matplotlib.use(...)` call) -- this assumes
            the caller's environment already has an interactive backend
            active (e.g. `%matplotlib widget` in Jupyter, or `Qt5Agg`/
            `TkAgg` in a script) when True. Leave False in headless/HPC
            contexts, where showing a figure can error or hang.
        download: Whether to also save the figure to disk.
        download_fpath: Output path when download=True, or a list of paths
            to save the same figure in multiple formats at once. Format is
            inferred per-path from its extension.
    """
    if interactive:
        import matplotlib.pyplot as plt

        plt.show()
    if download:
        if download_fpath is None:
            raise ValueError("download_fpath must be set when download=True")
        fpaths = (
            download_fpath if isinstance(download_fpath, list) else [download_fpath]
        )
        save_figure_multi(fig, fpaths)
