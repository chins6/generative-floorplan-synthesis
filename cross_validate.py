"""k-fold cross-validation of the VAE beta and the GNN depth (the test set is never used).

    python cross_validate.py --labels walls            # 5 folds x 7 configs = 35 runs
    python cross_validate.py --labels walls --jobs 12  # parallel worker processes

The 4,200 TRAINING plans are split into k folds (fixed seed). For each configuration
and fold, the model is trained for 90 epochs on k-1 folds with the usual protocol
(fresh 80% dropout every epoch, best checkpoint by held-out loss) and scored on the
held-out fold: wall-F1 at the F1-optimal threshold, ROC-AUC, and for the VAE the
number of active latent units and the wall fraction of its prior samples.
The selection rules of select_models.py are then applied to the fold MEANS.

Writes results/<labels>/cv_folds.csv, cv_summary.csv and figures/<labels>/fig_cv.png.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONFIGS = ([("VAE", {"beta": b}) for b in (0.1, 0.5, 1.0, 4.0)]
           + [("GNN", {"n_layers": L}) for L in (2, 4, 6)])


def fold_indices(n, k, seed=0):
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def run_one(labels, name, kwargs, fold, k, epochs):
    """Train one configuration on k-1 folds and score it on the held-out fold."""
    from floorplan import config as C
    C.set_labels(labels)
    from floorplan.analysis import batched_predict, vae_active_units
    from floorplan.checkpoint import load_checkpoint
    from floorplan.data import corrupt, load_splits
    from floorplan.losses import best_threshold, pixel_auc
    from floorplan.models import build_model
    from floorplan.trainer import fit

    X_train, _, _ = load_splits(verbose=False)
    folds = fold_indices(len(X_train), k)
    hold = folds[fold]
    fit_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold])
    X_fit, X_hold = X_train[fit_idx], X_train[hold]
    hold_noisy = corrupt(X_hold, np.random.default_rng(C.VAL_NOISE_SEED))

    tag = "_".join(f"{a}{b}" for a, b in kwargs.items())
    ckpt = C.MODELS_DIR / "cv" / f"{name.lower()}_{tag}_fold{fold}.npz"
    t0 = time.time()
    hist = fit(build_model(name, **kwargs), X_fit, X_hold, hold_noisy, epochs=epochs,
               ckpt_path=ckpt, kl_warmup=C.VAE_KL_WARMUP if name == "VAE" else 0, log=None)
    model, _ = load_checkpoint(ckpt)
    prob = batched_predict(model, hold_noisy)
    thr, f1 = best_threshold(prob, X_hold)
    row = {"model": name, **kwargs, "fold": fold, "best_epoch": hist["best_epoch"],
           "hold_loss": hist["best_val_loss"], "threshold": thr, "wall_f1": f1,
           "roc_auc": pixel_auc(prob, X_hold), "minutes": (time.time() - t0) / 60}
    if name == "VAE":
        au, total = vae_active_units(model, X_hold)
        row.update(active_units=au, latent_units=total,
                   prior_wall_fraction=float((model.sample_prior(400, np.random.default_rng(0)) > thr).mean()),
                   real_wall_fraction=float((X_train > 0.5).mean()))
    return row


def summarise(df):
    """Mean +/- std over folds, then the selection rules of select_models.py."""
    out = []
    for (name, key), g in df.groupby(["model", "key"], sort=False):
        r = {"model": name, "config": key, "folds": len(g),
             "f1_mean": g.wall_f1.mean(), "f1_std": g.wall_f1.std(ddof=1),
             "auc_mean": g.roc_auc.mean(), "auc_std": g.roc_auc.std(ddof=1)}
        if name == "VAE":
            r.update(active_units_mean=g.active_units.mean(),
                     prior_wall_fraction_mean=g.prior_wall_fraction.mean(),
                     real_wall_fraction=g.real_wall_fraction.iloc[0])
            r["qualifies"] = bool(abs(r["prior_wall_fraction_mean"] / r["real_wall_fraction"] - 1) <= 0.5
                                  and r["active_units_mean"] >= 0.25 * g.latent_units.iloc[0])
        out.append(r)
    s = pd.DataFrame(out)
    s["selected"] = False
    for name in ("VAE", "GNN"):
        sub = s[s.model == name]
        if name == "VAE" and sub.qualifies.any():
            sub = sub[sub.qualifies]
        s.loc[sub.f1_mean.idxmax(), "selected"] = True
    return s


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from floorplan.plots import COLORS
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    for ax, name, label in ((axes[0], "VAE", "beta"), (axes[1], "GNN", "GraphSAGE layers")):
        s = summary[summary.model == name]
        x = np.arange(len(s))
        bars = ax.bar(x, s.f1_mean, yerr=s.f1_std, capsize=4, color=COLORS[name])
        for b, sel in zip(bars, s.selected):
            b.set_alpha(1.0 if sel else 0.45)     # the selected configuration is opaque
        for xi, (m, sd) in zip(x, zip(s.f1_mean, s.f1_std)):
            ax.text(xi, m + sd + 0.004, f"{m:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x, [c.split("=")[1] for c in s.config])
        ax.set_xlabel(label)
        lo = (s.f1_mean - s.f1_std).min()
        ax.set_ylim(lo - 0.03, (s.f1_mean + s.f1_std).max() + 0.02)
        ax.set_title(f"{name}: 5-fold CV wall-F1 (mean ± std)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("held-out fold wall-F1")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="walls", choices=["ink", "walls"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    from floorplan import config as C
    C.set_labels(args.labels)
    C.ensure_dirs()
    from floorplan.data import load_splits
    load_splits(verbose=False)          # build the cache once, before the workers start

    tasks = [(args.labels, n, kw, f, args.folds, args.epochs)
             for n, kw in CONFIGS for f in range(args.folds)]
    print(f"{len(tasks)} runs ({len(CONFIGS)} configs x {args.folds} folds) on {args.jobs} workers")
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, *t) for t in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            rows.append(r)
            cfg = json.dumps({k: v for k, v in r.items() if k in ("beta", "n_layers")})
            print(f"  [{len(rows):2d}/{len(tasks)}] {r['model']} {cfg} fold {r['fold']}: "
                  f"F1 {r['wall_f1']:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    df["key"] = [f"beta={b:g}" if m == "VAE" else f"L={int(L)}"
                 for m, b, L in zip(df.model, df.get("beta"), df.get("n_layers"))]
    order = {f"beta={b:g}": i for i, b in enumerate((0.1, 0.5, 1.0, 4.0))}
    order.update({f"L={L}": 10 + L for L in (2, 4, 6)})
    df = df.sort_values(["model", "key", "fold"], key=lambda c: c.map(order) if c.name == "key" else c)
    df.to_csv(C.RESULTS_DIR / "cv_folds.csv", index=False)
    summary = summarise(df)
    summary.to_csv(C.RESULTS_DIR / "cv_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    plot(summary, C.FIGURES_DIR / "fig_cv.png")
    print("saved cv_folds.csv, cv_summary.csv, fig_cv.png")


if __name__ == "__main__":
    main()
