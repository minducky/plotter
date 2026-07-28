"""Journal/paper style presets for the `plotter` package.

Edit the values in PAPER_PRESETS to change how each `paper=...` option
formats a figure (fonts, sizes, figure width). Height is intentionally not
part of a preset — journals constrain column width, not figure height, so
`height` always comes from the caller's explicit arg.
"""

_PAPER_DPI = 300


def _mm_to_px(mm: float, dpi: int = _PAPER_DPI) -> int:
    """Convert a length in millimeters to pixels at the given DPI.

    Args:
        mm: Length in millimeters.
        dpi: Resolution to convert at.

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
        "width_mm": 161,
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
