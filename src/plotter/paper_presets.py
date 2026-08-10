"""Journal/paper style presets for the `plotter` package.

Edit the values in PAPER_PRESETS to change how each `paper=...` option
formats a figure (fonts, sizes, figure width). Height is intentionally not
part of a preset — journals constrain column width, not figure height, so
`height` always comes from the caller's explicit arg.
"""

# Kaleido (the headless-Chrome renderer behind `write_image`) always treats a
# figure's `width`/`height` as CSS pixels at 96 px/inch when laying out the
# exported page (a PDF/SVG page's physical size, or a PNG's pixel dimensions
# at scale=1). This is fixed by the renderer, not something `plotter`
# controls, so `_mm_to_px` must convert at 96 dpi for a `width_mm` preset to
# produce a PDF whose actual physical page width matches it — converting at
# 300 dpi instead makes the page ~3.1x too large. Confirmed by inspecting a
# rendered PDF's /MediaBox.
#
# Kaleido's `scale=` kwarg on `write_image` does NOT give a way around this:
# it uniformly enlarges the whole exported page (not just embedded raster
# resolution), confirmed empirically by comparing a PDF's /MediaBox at
# scale=1 vs scale=3.125 for the same width/height — the page grew by
# exactly 3.125x, not just the raster content. So there's no free way to
# raise embedded-raster fidelity without also growing the physical page
# size; `plotter` deliberately leaves `scale` at its default of 1 and
# prioritizes correct physical page sizing.
_KALEIDO_REFERENCE_DPI = 96


def _mm_to_px(mm: float, dpi: int = _KALEIDO_REFERENCE_DPI) -> int:
    """Convert a length in millimeters to the pixel width/height Kaleido
    needs to produce that physical size in an exported PDF/SVG/PNG.

    Args:
        mm: Length in millimeters.
        dpi: Reference resolution; defaults to Kaleido's fixed 96 px/inch
            page-layout assumption. Only override this for a different
            renderer with a different pixel-to-physical-size assumption.

    Returns:
        The equivalent length in pixels, rounded to the nearest integer.
    """
    return round(mm / 25.4 * dpi)


PAPER_PRESETS: dict[str, dict] = {
    "TASLP_single": {
        "font_family": "Times New Roman, Times, DejaVu Serif",
        "tick_fontsize": 7,
        "label_fontsize": 8,
        "title_fontsize": 9,
        "width_mm": 89,
    },
    "TASLP_double": {
        "font_family": "Times New Roman, Times, DejaVu Serif",
        "tick_fontsize": 7,
        "label_fontsize": 8,
        "title_fontsize": 9,
        "width_mm": 182,
    },
    "Nature": {
        "font_family": "Arial",
        "tick_fontsize": 7,
        "label_fontsize": 8,
        "title_fontsize": 9,
        "width_mm": 89,
    },
    "NeurIPS": {
        "font_family": "Times New Roman, Times, DejaVu Serif",
        "tick_fontsize": 8,
        "label_fontsize": 9,
        "title_fontsize": 10,
        "width_mm": 88,
    },
}


def resolve_style(paper: str | None, **explicit) -> dict:
    """Merge paper-preset styling with explicit style kwargs.

    If `paper` is given, its values take precedence over the matching
    explicit kwargs — paper presets are meant to be the single source of
    truth for a submission's required formatting. `height` is never
    touched by a preset; it always passes through from `explicit`.

    Args:
        paper: Key into PAPER_PRESETS, or None to use `explicit` as-is.
        **explicit: title_font, title_fontsize, xaxis_font, xaxis_fontsize,
            yaxis_font, yaxis_fontsize, tick_fontsize, width, height.

    Returns:
        Dict with the same keys as `explicit`, resolved.
    """
    resolved = dict(explicit)
    if paper is not None:
        preset = PAPER_PRESETS[paper]
        resolved.update(
            title_font=preset["font_family"],
            title_fontsize=preset["title_fontsize"],
            xaxis_font=preset["font_family"],
            xaxis_fontsize=preset["label_fontsize"],
            yaxis_font=preset["font_family"],
            yaxis_fontsize=preset["label_fontsize"],
            tick_fontsize=preset["tick_fontsize"],
            width=_mm_to_px(preset["width_mm"]),
        )
    return resolved
