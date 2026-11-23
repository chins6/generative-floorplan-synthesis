"""Compare the two label sets on the test set: 'ink' (mid-review) vs 'walls' (SVG labels).

    python compare_labels.py      # needs results/ and results/walls/ from evaluate.py

Writes results/labels_comparison.csv and figures/fig_labels_comparison.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from floorplan import config as C  # noqa: E402
from floorplan.plots import COLORS  # noqa: E402

MODELS = ["copy_input", "DAE", "VAE", "GNN"]


def main():
    rows = []
    for labels, folder in (("ink", C.PROJECT_DIR / "results"), ("walls", C.PROJECT_DIR / "results" / "walls")):
        comp = pd.read_csv(folder / "final_comparison.csv")
        err = pd.read_csv(folder / "error_analysis.csv").set_index("model")
        gen = pd.read_csv(folder / "generation.csv").set_index("source")
        for r in comp[comp.split == "test"].itertuples():
            if r.model not in MODELS:
                continue
            row = {"labels": labels, "model": r.model, "wall_precision": r.wall_precision,
                   "wall_recall": r.wall_recall, "wall_f1": r.wall_f1, "wall_iou": r.wall_iou,
                   "roc_auc": r.roc_auc}
            if r.model in err.index:
                row["fp_thickening"] = err.loc[r.model, "fp_thickening"]
                row["fn_erased"] = err.loc[r.model, "fn_erased"]
            rows.append(row)
        for src in gen.index:
            if "prior" in src or "perturbed" in src or src.startswith("real training"):
                rows.append({"labels": labels, "model": "gen: " + src,
                             "wall_fraction": gen.loc[src, "wall_fraction"],
                             "line_continuity": gen.loc[src, "line_continuity"],
                             "diversity": gen.loc[src, "diversity"]})
    df = pd.DataFrame(rows)
    df.to_csv(C.PROJECT_DIR / "results" / "labels_comparison.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for ax, metric, title in ((axes[0], "wall_f1", "Test wall-F1"), (axes[1], "wall_iou", "Test wall-IoU")):
        x = np.arange(len(MODELS))
        for k, (labels, hatch) in enumerate((("ink", ""), ("walls", "//"))):
            vals = [df[(df.labels == labels) & (df.model == m)][metric].iloc[0] for m in MODELS]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=[COLORS[m] for m in MODELS],
                          alpha=0.45 if labels == "ink" else 1.0, hatch=hatch, edgecolor="white",
                          label=("all dark ink (mid-review)" if labels == "ink" else "official wall polygons (SVG)"))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(x, ["copy input", "DAE", "VAE", "GNN"])
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Same models, same protocol, two label sets (80% of pixels erased)", fontsize=9)
    fig.tight_layout()
    out = C.PROJECT_DIR / "figures" / "fig_labels_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
