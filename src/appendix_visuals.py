"""Rebuild Appendix B figures from all saved development seeds; no training."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def load_paired_results():
    """Require the complete, paired cohort before drawing any result."""
    with (ROOT / "results/corrected/noise_runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    seeds = list(range(42001, 42011))
    methods = ["unclipped", "clipped", "noise_simulation"]
    assert len(rows) == len(seeds) * len(methods)
    values = np.empty((len(seeds), len(methods)))
    for i, seed in enumerate(seeds):
        for j, method in enumerate(methods):
            selected = [row for row in rows
                        if int(row["seed"]) == seed and row["method"] == method]
            assert len(selected) == 1, "Missing or duplicate paired endpoint"
            values[i, j] = 100 * float(selected[0]["development_macro_f1"])
    assert np.isfinite(values).all()
    return seeds, values


def plot_noise_pairs():
    """Show every endpoint and difference; the interval is over seeds, not people."""
    seeds, values = load_paired_results()
    difference = values[:, 2] - values[:, 1]
    mean = difference.mean()
    half_width = t.ppf(0.975, len(seeds) - 1) * difference.std(ddof=1) / np.sqrt(len(seeds))
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
        for row in values:
            axes[0].plot(range(3), row, color="0.65", marker="o", markersize=3,
                         linewidth=0.8, alpha=0.8)
        axes[0].plot(range(3), values.mean(axis=0), color="black", marker="D",
                     linewidth=1.5, markersize=5, label="Mean")
        axes[0].set_xticks(range(3), ["Unclipped", "Clipped", "Noise\nsimulation"])
        axes[0].set(ylabel="Development macro-F1 (%)", ylim=(0, 55),
                    title="(a) All ten paired seeds")
        axes[0].legend(frameon=False, fontsize=9)
        for x, y in enumerate(values.mean(axis=0)):
            axes[0].annotate(f"{y:.2f}%", (x, y), xytext=(12 if x == 0 else 0, 12),
                             textcoords="offset points", ha="left" if x == 0 else "center", fontsize=9)
        axes[1].plot(difference, range(1, 11), "o", color="0.4", markersize=4)
        axes[1].errorbar(mean, 12, xerr=half_width, fmt="D", color="black", capsize=4)
        axes[1].set_yticks(list(range(1, 11)) + [12],
                          [str(seed) for seed in seeds] + ["Mean / CI"])
        axes[1].set(xlabel="Noise minus clipped (percentage points)",
                    xlim=(-49, -41), ylim=(13, 0), title="(b) Paired differences")
        axes[1].grid(axis="x", color="0.9", linewidth=0.6)
        fig.tight_layout(w_pad=2)
    return fig


def export_figures():
    output = ROOT / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    figure = plot_noise_pairs()
    figure.savefig(output / "noise_paired_results.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "noise_paired_results.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    export_figures()
