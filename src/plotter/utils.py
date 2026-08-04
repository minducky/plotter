"""Shared Plotly trace-building and layout-styling helpers for `plotter`."""

import os

import numpy as np
import plotly.graph_objects as go


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


def download_figure(fig: go.Figure, download_fpath: str) -> None:
    """Save a figure to disk, choosing the writer by file extension.

    ".html" writes an interactive HTML file; any other extension (e.g.
    ".pdf", ".png", ".svg") is written as a static image via Kaleido.

    Args:
        fig: Figure to save.
        download_fpath: Output path; its extension determines the format.
    """
    ext = os.path.splitext(download_fpath)[1].lower()
    if ext == ".html":
        fig.write_html(download_fpath)
    else:
        fig.write_image(download_fpath)


def download_figure_multi(fig: go.Figure, download_fpaths: list[str]) -> None:
    """Save a figure to multiple paths at once, format inferred per-path.

    Args:
        fig: Figure to save.
        download_fpaths: Output paths (e.g. one ".pdf" and one ".html").
            Each path's format is inferred from its extension, same as
            `download_figure`.
    """
    for fpath in download_fpaths:
        download_figure(fig, fpath)


def build_line_trace(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    color: str | None = "black",
    line_width: int = 1,
) -> go.Scatter:
    """Build a Plotly Scatter line trace. Accepts numpy arrays or torch tensors."""
    return go.Scatter(
        x=to_numpy(x),
        y=to_numpy(y),
        mode="lines",
        name=name,
        line=dict(color=color, width=line_width),
    )


def build_heatmap_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
    colorscale: str = "Cividis",
    colorbar: dict | None = None,
) -> go.Heatmap:
    """Build a Plotly Heatmap trace. Accepts numpy arrays or torch tensors."""
    return go.Heatmap(
        x=to_numpy(x),
        y=to_numpy(y),
        z=to_numpy(z),
        colorscale=colorscale,
        name=name,
        colorbar=colorbar,
    )


def build_surface_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
    colorscale: str = "Cividis",
    showscale: bool = True,
    colorbar: dict | None = None,
) -> go.Surface:
    """Build a Plotly Surface trace. Accepts numpy arrays or torch tensors."""
    return go.Surface(
        x=to_numpy(x),
        y=to_numpy(y),
        z=to_numpy(z),
        colorscale=colorscale,
        name=name,
        showscale=showscale,
        colorbar=colorbar,
    )


def tickfont(style: dict) -> dict | None:
    """Build a Plotly tickfont dict from a resolved style, or None."""
    if style.get("tick_fontsize") is None:
        return None
    return dict(size=style["tick_fontsize"])


def apply_layout(
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
            tickfont=tickfont(style),
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
            tickfont=tickfont(style),
        ),
        plot_bgcolor="white",
        width=style["width"],
        height=style["height"],
    )
