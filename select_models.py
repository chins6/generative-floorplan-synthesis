"""Choose the VAE beta and the GNN depth on the VALIDATION set only.

Reads the sweep checkpoints written by
    python train.py --model vae --beta B --tag betaB      (B in 0.1 0.5 1.0 4.0)
    python train.py --model gnn --layers L --tag LL       (L in 2 4 6)
and copies the selected ones to models/best_vae.npz and models/best_gnn.npz.

Selection rules (fixed before looking at the results):
  GNN  highest validation wall-F1 (threshold tuned on validation).
  VAE  the point of the VAE is a USABLE prior, so a beta only qualifies if
       (a) its prior samples have a wall fraction within +/-50% of real plans and
       (b) at least 25% of its latent units are active (no posterior collapse).
       Among qualifying betas, take the highest validation wall-F1. If none
       qualifies, fall back to the highest validation wall-F1.
"""
import shutil

import numpy as np
import pandas as pd

import sys as _sys
from floorplan import config as _C
if "--labels" in _sys.argv:
    _C.set_labels(_sys.argv[_sys.argv.index("--labels") + 1])
from floorplan import config as C
from floorplan.analysis import batched_predict, vae_active_units
from floorplan.checkpoint import load_checkpoint
from floorplan.data import fixed_noisy_sets, load_splits
from floorplan.losses import best_threshold, pixel_auc

BETAS = ["0.1", "0.5", "1.0", "4.0"]
LAYERS = [2, 4, 6]


def val_scores(model, X_val, val_noisy):
    prob = batched_predict(model, val_noisy)
    thr, f1 = best_threshold(prob, X_val)
    return thr, f1, pixel_auc(prob, X_val)


def main():
    X_train, X_val, X_test = load_splits()
    val_noisy, _ = fixed_noisy_sets(X_val, X_test)
    real_wf = float((X_train > 0.5).mean())

    vae_rows, beta_samples = [], {}
    for b in BETAS:
        path = C.MODELS_DIR / f"best_vae_beta{b}.npz"
        if not path.exists():
            continue
        model, meta = load_checkpoint(path)
        thr, f1, auc = val_scores(model, X_val, val_noisy)
        au, total = vae_active_units(model, X_val)
        samples = model.sample_prior(400, np.random.default_rng(0)) > thr
        beta_samples[float(b)] = samples.astype(np.float32)
        vae_rows.append({"beta": float(b), "best_epoch": meta["epoch"], "val_loss": meta["val_loss"],
                         "threshold": thr, "val_wall_f1": f1, "val_auc": auc,
                         "active_units": au, "latent_units": total,
                         "prior_wall_fraction": float(samples.mean()),
                         "real_wall_fraction": real_wf, "path": path.name})
    gnn_rows = []
    for L in LAYERS:
        path = C.MODELS_DIR / f"best_gnn_L{L}.npz"
        if not path.exists():
            continue
        model, meta = load_checkpoint(path)
        thr, f1, auc = val_scores(model, X_val, val_noisy)
        gnn_rows.append({"n_layers": L, "best_epoch": meta["epoch"], "val_loss": meta["val_loss"],
                         "threshold": thr, "val_wall_f1": f1, "val_auc": auc,
                         "receptive_px": (2 * L + 1) * C.GNN_PATCH, "params": model.n_params(),
                         "path": path.name})

    # ---- apply the rules -----------------------------------------------------
    if vae_rows:
        for r in vae_rows:
            r["qualifies"] = (abs(r["prior_wall_fraction"] / real_wf - 1) <= 0.5
                              and r["active_units"] >= 0.25 * r["latent_units"])
        pool = [r for r in vae_rows if r["qualifies"]] or vae_rows
        chosen = max(pool, key=lambda r: r["val_wall_f1"])
        for r in vae_rows:
            r["selected"] = r is chosen
        shutil.copy(C.MODELS_DIR / chosen["path"], C.MODELS_DIR / "best_vae.npz")
        shutil.copy(C.RESULTS_DIR / f"history_vae_beta{chosen['beta']}.csv",
                    C.RESULTS_DIR / "history_vae.csv")
        pd.DataFrame(vae_rows).to_csv(C.RESULTS_DIR / "selection_vae_beta.csv", index=False)
        print(pd.DataFrame(vae_rows).drop(columns=["path"]).round(4).to_string(index=False))
        print(f"-> VAE beta = {chosen['beta']} copied to best_vae.npz\n")
    if gnn_rows:
        chosen = max(gnn_rows, key=lambda r: r["val_wall_f1"])
        for r in gnn_rows:
            r["selected"] = r is chosen
        shutil.copy(C.MODELS_DIR / chosen["path"], C.MODELS_DIR / "best_gnn.npz")
        shutil.copy(C.RESULTS_DIR / f"history_gnn_L{chosen['n_layers']}.csv",
                    C.RESULTS_DIR / "history_gnn.csv")
        pd.DataFrame(gnn_rows).to_csv(C.RESULTS_DIR / "selection_gnn_layers.csv", index=False)
        print(pd.DataFrame(gnn_rows).drop(columns=["path"]).round(4).to_string(index=False))
        print(f"-> GNN layers = {chosen['n_layers']} copied to best_gnn.npz")

    if vae_rows and gnn_rows:
        from floorplan.plots import hparam_plot, vae_beta_samples
        print("Saved", hparam_plot(vae_rows, gnn_rows))
        print("Saved", vae_beta_samples(beta_samples))


if __name__ == "__main__":
    main()
