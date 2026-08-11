"""Journal/paper style presets for the `plotter` package.

Edit the values in PAPER_PRESETS to change how each `paper=...` option
formats a figure (fonts, sizes, figure width). Height is intentionally not
part of a preset — journals constrain column width, not figure height, so
`height` always comes from the caller's explicit arg.

All sizes resolve to matplotlib's native units: figure width/height in
inches (matplotlib `figsize`), font sizes in points. There's no
renderer-specific DPI conversion to get wrong here — `fig.savefig(path,
dpi=300)` with a given `figsize` produces an exactly-sized page, always.
"""

MM_PER_INCH = 25.4


def mm_to_inches(mm: float) -> float:
    """Convert a length in millimeters to inches (matplotlib's figsize unit).

    Args:
        mm: Length in millimeters.

    Returns:
        The equivalent length in inches.
    """
    return mm / MM_PER_INCH


PAPER_PRESETS: dict[str, dict] = {
    "TASLP_single": {
        "font_family": "Times New Roman",
        "tick_fontsize": 7,
        "label_fontsize": 8,
        "title_fontsize": 9,
        "width_mm": 89,
    },
    "TASLP_double": {
        "font_family": "Times New Roman",
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
        "font_family": "Times New Roman",
        "tick_fontsize": 8,
        "label_fontsize": 9,
        "title_fontsize": 10,
        "width_mm": 89,
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
        **explicit: font, title_fontsize, xaxis_fontsize, yaxis_fontsize,
            tick_fontsize, width, height. `width` and `height` are in
            inches (matplotlib figsize units).

    Returns:
        Dict with the same keys as `explicit`, resolved.
    """
    resolved = dict(explicit)
    if paper is not None:
        preset = PAPER_PRESETS[paper]
        resolved.update(
            font=preset["font_family"],
            title_fontsize=preset["title_fontsize"],
            xaxis_fontsize=preset["label_fontsize"],
            yaxis_fontsize=preset["label_fontsize"],
            tick_fontsize=preset["tick_fontsize"],
            width=mm_to_inches(preset["width_mm"]),
        )
    return resolved
