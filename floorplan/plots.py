"""Every figure used in the report and the slides (saved to figures/)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import config as C  # noqa: E402

COLORS = {"DAE": "#2a6fdb", "VAE": "#d9480f", "GNN": "#2b8a3e", "copy_input": "#868e96"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 200, "image.cmap": "gray_r"})


def _save(fig, name):
    C.ensure_dirs()
    path = C.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _img(ax, flat, title=None):
    ax.imshow(flat.reshape(C.SIDE, C.SIDE), vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#ced4da")
    if title:
        ax.set_title(title, fontsize=8)


# ---- architecture diagram (original) ----------------------------------------------
def architecture_diagram(gnn_layers=C.GNN_LAYERS):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc=color, ec="#343a40", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#343a40"))

    # shared input / output
    box(0.5, 18, 10, 10, "Damaged\nplan\n1x32x32\n(80% erased)", "#f1f3f5", fs=7)
    box(88, 18, 11.5, 10, "Repaired plan\n1x32x32\np(wall) per pixel\n-> threshold\n(tuned on val)",
        "#f1f3f5", fs=7)
    rows = [
        (35, "DAE", "#dbe4ff", ["Conv 3x3/2\n16x16x16", "Conv 3x3/2\n32x8x8", "Conv 3x3/2\nz: 16x4x4\n(ReLU)",
                                "3 x [Conv 3x3\n+ upsample x2]", "Conv 1x1\n+ sigmoid"]),
        (19, "VAE", "#ffe8cc", ["Conv 3x3/2\n16x16x16", "Conv 3x3/2\n32x8x8", "mu, log s^2\n16x4x4\nz = mu + s*eps",
                                "3 x [Conv 3x3\n+ upsample x2]", "Conv 1x1\n+ sigmoid"]),
        (3, "GNN", "#d3f9d8", ["Image -> 8x8 graph\n64 nodes (4x4 px)\n8-neighbour edges", "Linear\n18 -> 64",
                               f"{gnn_layers} x GraphSAGE\nh + ReLU(Ws h +\nWn mean(h_N))",
                               "Linear 64 -> 16\nper node", "nodes -> image\n+ sigmoid"]),
    ]
    xs = [15, 29, 43, 58, 73]
    for y, label, color, blocks in rows:
        ax.text(xs[0], y + 9.9, label, ha="left", va="bottom", fontsize=9.5, fontweight="bold",
                color=COLORS[label])
        for i, (x, text) in enumerate(zip(xs, blocks)):
            box(x, y, 12.5, 8.5, text, color, fs=6.8)
            if i:
                arrow(xs[i - 1] + 12.8, y + 4.25, x - 0.3, y + 4.25)
        arrow(11.0, 23, xs[0] - 0.3, y + 4.25)
        arrow(xs[-1] + 12.8, y + 4.25, 87.7, 23)
    ax.text(50, 49.8, "Three models, one task: restore the damaged wall map "
            "(weighted BCE, Adam, best checkpoint on validation)",
            ha="center", va="top", fontsize=9, style="italic")
    ax.text(xs[0] + 5, 29.0, "VAE only: + beta * KL(q(z|x) || N(0, I));  test time z = mu;  "
            "generation: z ~ N(0, I) -> decoder", ha="left", va="bottom", fontsize=6.8,
            color=COLORS["VAE"])
    return _save(fig, "fig_architecture.png")


# ---- training curves ----------------------------------------------------------------
def training_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h["epoch"], h["train_recon"], color=COLORS[name], alpha=0.55, label="train (recon.)")
        ax.plot(h["epoch"], h["val_loss"], color=COLORS[name], label="validation")
        best = int(h["epoch"][int(np.argmin(h["val_loss"]))])
        ax.axvline(best, color="k", ls="--", lw=0.8)
        ax.scatter([best], [min(h["val_loss"])], color="k", zorder=5, s=18,
                   label=f"best checkpoint (epoch {best})")
        ax.set_title(name, color=COLORS[name], fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        lo = min(min(h["val_loss"]), min(h["train_recon"]))
        ax.set_ylim(lo - 0.01, lo + 0.12)
    axes[0].set_ylabel("weighted BCE")
    fig.suptitle("Loss curves (train vs validation); the checkpoint with the lowest validation "
                 "loss is the one benchmarked on test", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig_training_curves.png")


# ---- ROC --------------------------------------------------------------------------
def roc_curves(test_probs, X_test, copy_row):
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    for name, prob in test_probs.items():
        fpr, tpr, _ = roc_curve(X_test.reshape(-1) > 0.5, prob.reshape(-1))
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, color=COLORS[name], label=f"{name}  AUC {auc(fpr, tpr):.3f}")
    fpr_c = copy_row["fp"] / (copy_row["fp"] + copy_row["tn"])
    tpr_c = copy_row["tp"] / (copy_row["tp"] + copy_row["fn"])
    ax.scatter([fpr_c], [tpr_c], marker="s", color=COLORS["copy_input"], zorder=5,
               label="copy-input baseline")
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate (wall recall)")
    ax.set_title("Pixel ROC on the held-out test set")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_roc.png")


# ---- confusion matrices ---------------------------------------------------------
def confusion_matrices(rows):
    fig, axes = plt.subplots(1, len(rows), figsize=(3.0 * len(rows), 2.8))
    for ax, (name, r) in zip(axes, rows.items()):
        m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        norm = m / m.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({int(m[i, j]):,})", ha="center", va="center",
                        fontsize=7.5, color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1], ["bg", "wall"])
        ax.set_yticks([0, 1], ["bg", "wall"])
        ax.set_xlabel("predicted")
        ax.set_title(f"{name} (t={r['threshold']:.2f})", color=COLORS.get(name, "k"), fontsize=9)
    axes[0].set_ylabel("true")
    fig.suptitle("Test-set pixel confusion matrices (row-normalised; counts in brackets)", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig_confusion.png")


# ---- qualitative -----------------------------------------------------------------
def qualitative(X_test, test_noisy, test_probs, thresholds, idx):
    cols = ["damaged input", "ground truth"] + list(test_probs)
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(1.45 * len(cols), 1.5 * len(idx)))
    for r, i in enumerate(idx):
        _img(axes[r, 0], test_noisy[i], cols[0] if r == 0 else None)
        _img(axes[r, 1], X_test[i], cols[1] if r == 0 else None)
        for c, name in enumerate(test_probs, start=2):
            _img(axes[r, c], (test_probs[name][i] > thresholds[name]).astype(float),
                 name if r == 0 else None)
    fig.suptitle("Held-out test plans (random, fixed seed)", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig_qualitative.png")


# ---- error breakdown -------------------------------------------------------------
def error_breakdown_bars(breakdowns):
    keys = [("fp_thickening", "FP: stroke thickening", "#ffa94d"),
            ("fp_hallucination", "FP: hallucinated wall", "#e03131"),
            ("fn_erased", "FN: erased wall not restored", "#74c0fc"),
            ("fn_ignored", "FN: surviving wall removed", "#1c7ed6")]
    names = list(breakdowns)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    left = np.zeros(len(names))
    for key, label, color in keys:
        vals = np.array([breakdowns[n][key] for n in names])
        ax.barh(names, vals, left=left, color=color, label=label)
        for y, (l, v) in enumerate(zip(left, vals)):
            if v > 0.06:
                ax.text(l + v / 2, y, f"{v:.2f}", ha="center", va="center", fontsize=7)
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("errors per true wall pixel (test set)")
    ax.set_title("What kind of mistakes does each model make?")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    fig.tight_layout()
    return _save(fig, "fig_error_breakdown.png")


# ---- damage sweep ------------------------------------------------------------------
def damage_sweep_plot(sweeps):
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    first = next(iter(sweeps.values()))
    labels = [r["damage"] for r in first]
    xs = np.arange(len(labels))
    ax.plot(xs, [r["copy_f1"] for r in first], "s--", color=COLORS["copy_input"], label="copy input")
    for name, rows in sweeps.items():
        ax.plot(xs, [r["model_f1"] for r in rows], "o-", color=COLORS[name], label=name)
    for x, r in zip(xs, first):
        if r["n_blocks"]:
            ax.axvspan(x - 0.4, x + 0.4, color="0.9", zorder=0)
    ax.set_xticks(xs, labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("validation wall-F1")
    ax.set_title("Robustness to the amount and type of damage (grey = unseen damage type)", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig_damage_sweep.png")


# ---- generation --------------------------------------------------------------------
def generation_grid(real, dae_samples, vae_samples, n=8):
    rows = [("real training plans", real), ("DAE: perturbed real codes", dae_samples),
            ("VAE: z ~ N(0, I) (new)", vae_samples)]
    fig, axes = plt.subplots(3, n, figsize=(1.25 * n, 4.0))
    for r, (label, batch) in enumerate(rows):
        for c in range(n):
            _img(axes[r, c], batch[c])
        axes[r, 0].set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center")
    fig.suptitle("Generation: only the VAE has a learned prior to sample from", fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig_generation.png")


def vae_beta_samples(samples_by_beta, n=8):
    """Prior samples z ~ N(0, I) from the VAE trained at each beta (rate-distortion trade-off)."""
    fig, axes = plt.subplots(len(samples_by_beta), n, figsize=(1.1 * n, 1.2 * len(samples_by_beta)))
    for r, (beta, batch) in enumerate(samples_by_beta.items()):
        for c in range(n):
            _img(axes[r, c], batch[c])
        axes[r, 0].set_ylabel(f"beta = {beta:g}", fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("VAE prior samples vs beta: larger beta = coherent plans, smaller beta = better repair",
                 fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig_vae_beta_samples.png")


# ---- hyper-parameter selection -------------------------------------------------------
def hparam_plot(vae_rows, gnn_rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
    ax = axes[0]
    betas = [r["beta"] for r in vae_rows]
    xs = np.arange(len(betas))
    ax.plot(xs, [r["val_wall_f1"] for r in vae_rows], "o-", color=COLORS["VAE"], label="val wall-F1 (repair)")
    ax.set_xticks(xs, [str(b) for b in betas])
    ax.set_xlabel("beta (KL weight)")
    ax.set_ylabel("validation wall-F1")
    ax2 = ax.twinx()
    ax2.plot(xs, [r["prior_wall_fraction"] for r in vae_rows], "s--", color="#495057",
             label="prior-sample wall fraction")
    ax2.axhline(vae_rows[0]["real_wall_fraction"], color="#495057", ls=":", lw=0.8)
    ax2.set_ylabel("wall fraction of samples")
    ax2.spines["right"].set_visible(True)
    for x, r in zip(xs, vae_rows):
        ax.annotate(f"{r['active_units']} AU", (x, r["val_wall_f1"]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_title("VAE: repair quality vs usable prior", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="center left")
    ax = axes[1]
    Ls = [r["n_layers"] for r in gnn_rows]
    ax.plot(Ls, [r["val_wall_f1"] for r in gnn_rows], "o-", color=COLORS["GNN"])
    for r in gnn_rows:
        ax.annotate(f"RF {r['receptive_px']} px", (r["n_layers"], r["val_wall_f1"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_xticks(Ls)
    ax.set_xlabel("GraphSAGE layers")
    ax.set_ylabel("validation wall-F1")
    ax.set_title("GNN: depth = how far messages travel", fontsize=9)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_hparam.png")
