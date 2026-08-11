# plotter

A Plotly-based plotting utility package. It provides single-trace 1D/2D plots,
multi-panel/multi-line plots, and distribution (violin/box) plots, with a
`paper=` argument to apply journal-submission font/size/width presets.

## Usage

```bash
git clone https://github.com/minducky/plotter.git
cd plotter
pip install -e .
```

```python
import numpy as np
from plotter import plot_1d

x = np.linspace(0, 10, 200)
y = np.sin(x)
plot_1d(x, y, title="Example", xaxis_title="x", yaxis_title="y")
```

To pick up later changes:

```bash
git pull
pip install -e .
```
