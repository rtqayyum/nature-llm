"""Plot the full-cohort local comparison and the separate released-checkpoint comparison."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results/heldout/3111109/domains.csv"


def main():
    with SCORES.open() as stream:
        rows = list(csv.DictReader(stream))
    labels = [row["domain"] for row in rows]
    gains = [100 * (1 - float(row["weighted_margin_mae"]) / float(row["local_mae"])) for row in rows]
    ratios = [float(row["weighted_margin_common_mae"]) / float(row["release_common_mae"]) for row in rows]
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "heldout-3111109"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax in (left, right):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e3e8eb")
    y = list(range(len(rows)))
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_title("Full test rows", loc="left", fontweight="bold")
    left.set_xlabel("MAE reduction relative to local (%)")
    left.set_xlim(-.14, 3.85)
    left.set_xticks([0, 1, 2, 3])
    left.axvline(0, color="#8b9397", linewidth=.8)
    left.hlines(y, 0, gains, color="#419c8c", lw=2)
    left.scatter(gains, y, s=30, color="#126d60", zorder=3)
    for i, gain in enumerate(gains):
        left.annotate(f"{gain:.2f}%", (gain, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    right.set_title("Common test rows only", loc="left", fontweight="bold")
    right.set_xlabel("Our MAE / released checkpoint MAE")
    right.set_xlim(.82, 5.25)
    right.set_xticks([1, 2, 3, 4, 5])
    right.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    right.axvline(1, color="#8b9397", linewidth=.8)
    right.hlines(y, 1, ratios, color="#dd8f64", lw=2)
    right.scatter(ratios, y, s=30, color="#a6532e", zorder=3)
    for i, ratio in enumerate(ratios):
        right.annotate(f"{ratio:.2f}×", (ratio, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    fig.subplots_adjust(left=.145, right=.975, top=.84, bottom=.20, wspace=.18)
    output = ROOT / "paper/figures/heldout_domain_results.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date":None})
    plt.close(fig)


if __name__ == "__main__":
    main()
