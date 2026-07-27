"""Plotly-based plotting utilities for waveforms, spectral analyses, and
general-purpose figures, with optional journal/paper style presets.
"""

import numpy as np
import plotly.graph_objects as go
from acoustic_signal_processing import cal_fft, cal_psd, cal_stft
from plotly.subplots import make_subplots

# %% Paper presets

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
        "height_mm": 112,
    },
    # TODO: placeholder values, not verified against the real Nature figure spec.
    "Nature": {
        "font_family": "Arial",
        "tick_fontsize": 7,
        "label_fontsize": 8,
        "title_fontsize": 9,
        "width_mm": 89,
        "height_mm": 89,
    },
    # TODO: placeholder values, not verified against the real NeurIPS figure spec.
    "NeurIPS": {
        "font_family": "Times New Roman, Times, DejaVu Serif",
        "tick_fontsize": 8,
        "label_fontsize": 9,
        "title_fontsize": 10,
        "width_mm": 88,
        "height_mm": 66,
    },
}


def _resolve_style(paper: str | None, **explicit) -> dict:
    """Merge paper-preset styling with explicit style kwargs.

    If `paper` is given, its values take precedence over the matching
    explicit kwargs — paper presets are meant to be the single source of
    truth for a submission's required formatting.

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
            height=_mm_to_px(preset["height_mm"]),
        )
    return resolved


# %% Shared trace/layout helpers


def _build_line_trace(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    color: str | None = "black",
    line_width: int = 1,
) -> go.Scatter:
    """Build a Plotly Scatter line trace."""
    return go.Scatter(
        x=x, y=y, mode="lines", name=name, line=dict(color=color, width=line_width)
    )


def _build_heatmap_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
    colorscale: str = "Cividis",
) -> go.Heatmap:
    """Build a Plotly Heatmap trace."""
    return go.Heatmap(x=x, y=y, z=z, colorscale=colorscale, name=name)


def _tickfont(style: dict) -> dict | None:
    """Build a Plotly tickfont dict from a resolved style, or None."""
    if style.get("tick_fontsize") is None:
        return None
    return dict(size=style["tick_fontsize"])


def _apply_layout(
    fig: go.Figure,
    title: str | None,
    style: dict,
    title_fontcolor: str = "black",
    xaxis_title: str | None = None,
    xaxis_fontcolor: str = "black",
    yaxis_title: str | None = None,
    yaxis_fontcolor: str = "black",
    zeroline_color: str | None = None,
) -> None:
    """Apply shared title/axis/gridline styling to a figure's layout."""
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
        xaxis=dict(
            title=dict(
                text=f"<b>{xaxis_title}</b>",
                font=dict(
                    family=style["xaxis_font"],
                    size=style["xaxis_fontsize"],
                    color=xaxis_fontcolor,
                ),
            ),
            showgrid=True,
            gridcolor="lightgray",
            tickfont=_tickfont(style),
        ),
        yaxis=dict(
            title=dict(
                text=f"<b>{yaxis_title}</b>",
                font=dict(
                    family=style["yaxis_font"],
                    size=style["yaxis_fontsize"],
                    color=yaxis_fontcolor,
                ),
            ),
            showgrid=True,
            gridcolor="lightgray",
            zerolinecolor=zeroline_color,
            tickfont=_tickfont(style),
        ),
        plot_bgcolor="white",
        width=style["width"],
        height=style["height"],
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
        x: X-axis values.
        y: Y-axis values.
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
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
    """
    style = _resolve_style(
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
    fig.add_trace(_build_line_trace(x, y, name))
    _apply_layout(
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
        fig.write_html(download_fpath)


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
        x: X-axis coordinates.
        y: Y-axis coordinates.
        z: 2D array of values to color-map.
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
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
    """
    style = _resolve_style(
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
    fig.add_trace(_build_heatmap_trace(x, y, z, name))
    _apply_layout(
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
        fig.write_html(download_fpath)


# %% Domain-specific wrappers (wave, fft, psd, stft)


def plot_wave(
    sig: np.ndarray,
    sr: int,
    name: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    width: int,
    height: int,
    download: bool = False,
    download_fpath: str | None = None,
    paper: str | None = None,
) -> None:
    """Plot a time-domain waveform.

    Args:
        sig: Signal samples.
        sr: Sample rate in Hz.
        name: Trace name shown in the legend/hover.
        title: Figure title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        width: Figure width in pixels.
        height: Figure height in pixels.
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
        paper: Journal preset key from PAPER_PRESETS, or None.
    """
    x, y = np.linspace(0, len(sig) / sr, len(sig)), sig
    plot_1d(
        x,
        y,
        name=name,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        paper=paper,
        download=download,
        download_fpath=download_fpath,
    )


# TODO: remove — superseded by the more general plot functions below.
def plot_fft(
    sig: np.ndarray,
    sr: int,
    name: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    width: int,
    height: int,
    download: bool = False,
    download_fpath: str | None = None,
    db: bool = False,
    paper: str | None = None,
) -> None:
    """Plot an FFT magnitude spectrum.

    Args:
        sig: Signal samples.
        sr: Sample rate in Hz.
        name: Trace name shown in the legend/hover.
        title: Figure title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        width: Figure width in pixels.
        height: Figure height in pixels.
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
        db: Whether to compute the spectrum in dB.
        paper: Journal preset key from PAPER_PRESETS, or None.
    """
    x, y = cal_fft(sig, sr, db)
    plot_1d(
        x,
        y,
        name=name,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        paper=paper,
        download=download,
        download_fpath=download_fpath,
    )


# TODO: remove — superseded by the more general plot functions below.
def plot_psd(
    sig: np.ndarray,
    sr: int,
    name: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    width: int,
    height: int,
    download: bool = False,
    download_fpath: str | None = None,
    db: bool = False,
    n_fft: int | None = None,
    paper: str | None = None,
) -> None:
    """Plot a power spectral density estimate.

    Args:
        sig: Signal samples.
        sr: Sample rate in Hz.
        name: Trace name shown in the legend/hover.
        title: Figure title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        width: Figure width in pixels.
        height: Figure height in pixels.
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
        db: Whether to compute the PSD in dB.
        n_fft: FFT length used for the PSD estimate.
        paper: Journal preset key from PAPER_PRESETS, or None.
    """
    x, y = cal_psd(sig, sr, db, n_fft)
    plot_1d(
        x,
        y,
        name=name,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        paper=paper,
        download=download,
        download_fpath=download_fpath,
    )


# TODO: remove — superseded by the more general plot functions below.
def plot_stft(
    sig: np.ndarray,
    sr: int,
    name: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    width: int,
    height: int,
    download: bool = False,
    download_fpath: str | None = None,
    db: bool = False,
    n_fft: int = 2048,
    win_length: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    paper: str | None = None,
) -> None:
    """Plot a short-time Fourier transform spectrogram.

    Args:
        sig: Signal samples.
        sr: Sample rate in Hz.
        name: Trace name shown in hover.
        title: Figure title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        width: Figure width in pixels.
        height: Figure height in pixels.
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
        db: Whether to compute the STFT magnitude in dB.
        n_fft: FFT length.
        win_length: Analysis window length in samples.
        hop_length: Hop size between analysis windows in samples.
        window: Window function name.
        paper: Journal preset key from PAPER_PRESETS, or None.
    """
    x, y, z = cal_stft(
        sig,
        sr,
        db=db,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )
    # `z` is already real-valued dB when db=True (and may be negative) --
    # only take magnitude for the linear (complex) case, not the dB one.
    z = z if db else np.abs(z)
    plot_2d(
        x=x,
        y=y,
        z=z,
        name=name,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        paper=paper,
        download=download,
        download_fpath=download_fpath,
    )


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
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
    """
    style = _resolve_style(
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
            trace = _build_line_trace(panel["x"], panel["y"], panel.get("name", ""))
        else:
            trace = _build_heatmap_trace(
                panel["x"], panel["y"], panel["z"], panel.get("name", "")
            )
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(
            title_text=panel.get("xaxis_title"),
            tickfont=_tickfont(style),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=panel.get("yaxis_title"),
            tickfont=_tickfont(style),
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
        fig.write_html(download_fpath)


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
        series: List of (x, y, name) tuples, one per line. Each line gets a
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
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
    """
    style = _resolve_style(
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
        fig.add_trace(_build_line_trace(x, y, name, color=None))
    _apply_layout(
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
        fig.write_html(download_fpath)


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
        data: List of 1D arrays, one distribution per violin.
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
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
    style = _resolve_style(
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
            go.Violin(y=values, name=label, box_visible=True, meanline_visible=True)
        )
    _apply_layout(
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
        fig.write_html(download_fpath)


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
        data: List of 1D arrays, one distribution per box.
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
        download: Whether to also save the figure as an HTML file.
        download_fpath: Output path for the HTML file when download=True.
    """
    labels = (
        labels if labels is not None else [f"Group {i + 1}" for i in range(len(data))]
    )
    style = _resolve_style(
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
        fig.add_trace(go.Box(y=values, name=label))
    _apply_layout(
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
        fig.write_html(download_fpath)


# %% Main function

if __name__ == "__main__":
    sr = 16000
    freq = 500
    length = 0.5
    t = np.linspace(0, length, int(sr * length))
    sig = np.sin(2 * np.pi * freq * t)

    plot_wave(
        sig=sig,
        sr=sr,
        name="Wave",
        title=f"sin {freq}Hz {length}sec Wave",
        xaxis_title="Time (sec)",
        yaxis_title="Amplitude",
        width=1200,
        height=500,
        download=True,
        download_fpath="test.html",
    )

    plot_fft(
        sig=sig,
        sr=sr,
        name="FFT",
        title=f"sin {freq}Hz {length}sec FFT",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Half Amplitude (dB)",
        width=1200,
        height=500,
        download=True,
        download_fpath="test.html",
        db=True,
    )

    plot_psd(
        sig=sig,
        sr=sr,
        name="PSD",
        title=f"sin {freq}Hz {length}sec PSD",
        xaxis_title="Frequency (Hz)",
        yaxis_title=r"PSD (dB/Hz)",
        width=1200,
        height=500,
        download=True,
        download_fpath="test.html",
        db=True,
        n_fft=512,
    )

    plot_stft(
        sig=sig,
        sr=sr,
        name="STFT",
        title=f"sin {freq}Hz {length}sec STFT",
        xaxis_title="Time (sec)",
        yaxis_title="Frequency (Hz)",
        width=800,
        height=800,
        download=False,
        download_fpath=None,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        window="hann",
    )
