"""Shared palette and save helper for the review figure set.

One fixed colour per data source/policy, used by every figure in
validation/figures/review/ so a policy is recognisable across the whole set.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = ROOT / "validation" / "figures" / "review"

COLORS = {
    "EDGAR": "#7f7f7f",                # grey - real data, always
    "noop": "#9467bd",                 # purple
    "random": "#d62728",               # red
    "heuristic": "#2ca02c",            # green
    "boardroom": "#1f77b4",            # blue
    "oracle_v1": "#fdae61",            # light orange  (oracle family = oranges)
    "oracle_v3": "#d95f02",            # dark orange
    "oracle_v3_no_memory": "#8c510a",  # brown
}

LABELS = {
    "EDGAR": "EDGAR panel",
    "noop": "no-action",
    "random": "random",
    "heuristic": "heuristic",
    "boardroom": "boardroom",
    "oracle_v1": "oracle_v1",
    "oracle_v3": "oracle_v3",
    "oracle_v3_no_memory": "oracle_v3 (no memory)",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 12,
})


def save(fig, name: str) -> None:
    """Write PNG (200 dpi) + SVG into validation/figures/review/."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REVIEW_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(REVIEW_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.svg")


def footnote(fig, text: str) -> None:
    fig.text(0.01, -0.01, text, fontsize=7, color="#555555", ha="left", va="top")
