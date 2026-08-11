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
        style: A dict from `resolve_style`, at least containing `title_font`.
    """
    import matplotlib.pyplot as plt

    family = style.get("title_font")
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
    if xaxis_title:
        ax.set_xlabel(xaxis_title, fontsize=style.get("xaxis_fontsize"), fontfamily=style.get("xaxis_font"))
    if yaxis_title:
        ax.set_ylabel(yaxis_title, fontsize=style.get("yaxis_fontsize"), fontfamily=style.get("yaxis_font"))
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


def save_figure(fig: matplotlib.figure.Figure, download_fpath: str) -> None:
    """Save a figure to disk, choosing the writer by file extension.

    ".pdf"/".png"/".svg" are written via `fig.savefig(..., dpi=300)` with NO
    `bbox_inches='tight'` -- every figure this package builds uses
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
    fig.savefig(download_fpath, dpi=300)


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
