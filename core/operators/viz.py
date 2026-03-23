import io
from typing import Any, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None


def histogram_svg(hist_data: Tuple[list, list], title: str = "", logger: Optional[Any] = None) -> str:
    """Generates an SVG string of a histogram plot."""
    if not plt:
        return """<svg width="100" height="20" xmlns="http://www.w3.org/2000/svg"><text x="5" y="15" font-family="sans-serif" font-size="10" fill="orange">Matplotlib missing</text></svg>"""
    if not hist_data or not any(len(d) > 0 for d in hist_data if hasattr(d, "__len__")):
        return ""
    try:
        counts, bins = hist_data
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=120)
            ax.bar(bins[:-1], counts, width=np.diff(bins), color="#7aa2ff", alpha=0.85, align="edge")
            ax.grid(axis="y", alpha=0.2)
            ax.margins(x=0)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_title(title)
            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            plt.close(fig)
        return buf.getvalue()
    except Exception:
        if logger:
            logger.error("Failed to generate histogram SVG.", exc_info=True)
        return """<svg width="100" height="20" xmlns="http://www.w3.org/2000/svg"><text x="5" y="15" font-family="sans-serif" font-size="10" fill="red">Plotting failed</text></svg>"""
