"""Save notebook and course-required EPS figure variants."""

from pathlib import Path


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(suffix) for suffix in (".png", ".eps")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=180, facecolor="white", transparent=False)
    return paths
