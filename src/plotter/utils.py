"""Shared matplotlib figure-building and styling helpers for `plotter`."""

import os
import warnings

import matplotlib.figure
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


# 4.6%/4% isn't a matplotlib default (that's fraction=0.15, pad~0.05) -- it's
# a widely-used community recipe for a slim colorbar matched to its plot's
# height.
COLORBAR_SIZE = "4.6%"
COLORBAR_PAD = "4%"


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
    # `fig.colorbar(im, ax=ax, fraction=..., pad=...)` puts the colorbar's
    # Axes under constrained_layout's own colorbar handling, which
    # re-applies a fixed height:width box_aspect (default 20) on every
    # draw/savefig pass -- so it silently overrides any position/aspect we
    # set right after creation, and binds before `ax`'s actual height
    # whenever `ax` is narrow relative to its height (e.g. a many-column
    # grid), leaving the colorbar visibly shorter than `ax`. A divider-based
    # cax instead gets its own locator that reads `ax`'s real rendered bbox
    # on every draw, independent of constrained_layout's colorbar-specific
    # aspect logic, so it tracks `ax`'s height exactly no matter how many
    # more layout passes happen afterward.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(0.5)
    if tick_fontsize is not None:
        cbar.ax.tick_params(labelsize=tick_fontsize)
    if not show_ticklabels:
        cbar.set_ticks([])
    return cbar


def reserve_colorbar_space(ax, size: str = COLORBAR_SIZE, pad: str = COLORBAR_PAD) -> None:
    """Shrinks `ax` by the same amount `add_colorbar` would, without drawing
    a colorbar there.

    For a multi-row grid where only some rows' panels get a colorbar (e.g.
    one shared scale shown once), the panels that skip it would otherwise
    stay full-width while their siblings shrink to make room -- misaligning
    anything meant to share a scale, like a time axis. Call this on the
    panels that don't get their own colorbar, so every panel in the group
    ends up the same width.

    Args:
        ax: The Axes to reserve (and hide) colorbar-shaped space on.
        size: Width to reserve, as a percentage string. Defaults to
            `COLORBAR_SIZE`, matching `add_colorbar`.
        pad: Gap before the reserved space, as a percentage string. Defaults
            to `COLORBAR_PAD`, matching `add_colorbar`.
    """
    make_axes_locatable(ax).append_axes("right", size=size, pad=pad).axis("off")


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


SAVE_DPI = 1200  # resolution for rasterized content (imshow heatmaps) embedded
# in a saved PDF/PNG/SVG. Vector content (lines, text, axes) is unaffected
# by this -- it's only the raster images that get sampled at this density.
# A cochleagram panel can hold tens of thousands of time samples in under an
# inch of physical width, so no practical DPI embeds its full native
# resolution (at this panel size, 1200 dpi still only captures ~1-2% of a
# 44.1kHz signal's sample count -- confirmed by inspecting a rendered PDF's
# embedded image XObject dimensions). 1200 is a user-chosen practical
# ceiling: roughly 2x sharper than 600 without file size/render time
# growing unreasonably. Pair with draw_2d's interpolation="nearest" (no
# antialiasing) so whatever resolution IS embedded stays crisp rather than
# blending into a soft/merged look.


def save_figure(fig: matplotlib.figure.Figure, download_fpath: str, dpi: int | None = None) -> None:
    """Save a figure to disk, choosing the writer by file extension.

    ".pdf"/".png"/".svg" are written via `fig.savefig(..., dpi=dpi or
    SAVE_DPI)` with NO `bbox_inches='tight'` -- every figure this package
    builds uses `constrained_layout=True`, which already keeps titles/
    labels/colorbars within the figure's own bounds, so the saved page is
    always *exactly* `figsize`
    with no renderer-specific DPI conversion to get wrong (unlike the old
    Plotly/Kaleido implementation) and no tight-bbox cropping/expansion to
    make it inexact. ".html" has no first-class matplotlib equivalent to
    Plotly's interactive export, so it's skipped with a warning rather than
    silently doing nothing or raising.

    Args:
        fig: Figure to save.
        download_fpath: Output path; its extension determines the format.
        dpi: Override for `SAVE_DPI` on this save only, or None to use the
            module default. Large raster content (e.g. a heatmap with
            hundreds of cells per side, or a figure sized well beyond a
            typical print page) can make `SAVE_DPI`'s 1200 produce an
            enormous file -- pass a lower value in that case; see
            `result_plotter.py`'s confusion-matrix sizing for a worked
            example of scaling `dpi` down as content grows.
    """
    ext = os.path.splitext(download_fpath)[1].lower()
    if ext == ".html":
        warnings.warn(
            f"save_figure: skipping {download_fpath!r} -- matplotlib has no "
            "interactive HTML export; use interactive=True to view instead.",
            stacklevel=2,
        )
        return
    fig.savefig(download_fpath, dpi=dpi or SAVE_DPI)


def save_figure_multi(
    fig: matplotlib.figure.Figure, download_fpaths: list[str], dpi: int | None = None
) -> None:
    """Save a figure to multiple paths at once, format inferred per-path.

    Args:
        fig: Figure to save.
        download_fpaths: Output paths (e.g. one ".pdf" and one ".png").
            Each path's format is inferred from its extension, same as
            `save_figure`.
        dpi: See `save_figure`.
    """
    for fpath in download_fpaths:
        save_figure(fig, fpath, dpi=dpi)


def finalise_figure(
    fig: matplotlib.figure.Figure,
    interactive: bool,
    download: bool,
    download_fpath: str | list[str] | None,
    dpi: int | None = None,
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
        dpi: See `save_figure`.
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
        save_figure_multi(fig, fpaths, dpi=dpi)
