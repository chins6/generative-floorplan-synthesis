"""Benchmark the BEST checkpoints of DAE, VAE and GNN on the held-out test set.

    python evaluate.py                      # uses models/best_{dae,vae,gnn}.npz
    python evaluate.py --models-dir models/demo   # benchmark the live-demo checkpoints

For each model: the decision threshold is tuned on validation, then frozen and
applied to the 400 unseen test plans. Writes results/*.csv and figures/*.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from floorplan import config as _C
if "--labels" in _sys.argv:
    _C.set_labels(_sys.argv[_sys.argv.index("--labels") + 1])
from floorplan import config as C
from floorplan import plots
from floorplan.analysis import (baseline_rows, damage_sweep, dae_mixture_samples,
                                error_breakdown, evaluate_model, generation_stats,
                                vae_active_units)
from floorplan.checkpoint import load_checkpoint
from floorplan.data import fixed_noisy_sets, load_splits

MODEL_NAMES = ["DAE", "VAE", "GNN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(C.MODELS_DIR))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--labels", default="ink", choices=["ink", "walls"], help="ink = dark pixels of the PNG (mid-review); walls = official SVG wall polygons")
    args = ap.parse_args()
    models_dir = Path(args.models_dir)
    demo = models_dir.resolve() != C.MODELS_DIR.resolve()
    res_dir = C.RESULTS_DIR / "demo" if demo else C.RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test = load_splits()
    val_noisy, test_noisy = fixed_noisy_sets(X_val, X_test)

    models, metas = {}, {}
    for name in MODEL_NAMES:
        path = models_dir / f"best_{name.lower()}.npz"
        if path.exists():
            models[name], metas[name] = load_checkpoint(path)
            print(f"Loaded {path.name}: {models[name].n_params():,} params, best epoch "
                  f"{metas[name]['epoch']} (val loss {metas[name]['val_loss']:.4f})")
        else:
            print(f"[skip] {path} not found")

    # ---- 1. headline comparison --------------------------------------------------
    rows, test_probs, val_probs, thresholds = [], {}, {}, {}
    base = baseline_rows(X_val, val_noisy, X_test, test_noisy, float(X_train.mean()))
    for name, res in base.items():
        for split, r in res.items():
            rows.append(dict(model=name, split=split, **r))
    for name, model in models.items():
        res, vp, tp, thr = evaluate_model(model, X_val, val_noisy, X_test, test_noisy)
        val_probs[name], test_probs[name], thresholds[name] = vp, tp, thr
        for split, r in res.items():
            rows.append(dict(model=name, split=split, params=model.n_params(),
                             best_epoch=metas[name]["epoch"], **r))
    table = pd.DataFrame(rows)
    table.to_csv(res_dir / "final_comparison.csv", index=False)
    cols = ["model", "threshold", "accuracy", "wall_precision", "wall_recall", "wall_f1",
            "wall_iou", "roc_auc", "loss"]
    print("\n=== HELD-OUT TEST SET (400 plans, best checkpoints, threshold from validation) ===")
    print(table[table.split == "test"][cols].round(4).to_string(index=False))

    # ---- 2. failure analysis -------------------------------------------------------
    breakdowns, per_plan = {}, {}
    for name in models:
        b, f1s, dens = error_breakdown(test_probs[name], X_test, test_noisy, thresholds[name])
        breakdowns[name], per_plan[name] = b, f1s
    b, _, _ = error_breakdown(test_noisy, X_test, test_noisy, 0.5)
    breakdowns_all = dict(breakdowns, copy_input=b)
    err = pd.DataFrame(breakdowns_all).T.rename_axis("model").reset_index()
    err.to_csv(res_dir / "error_analysis.csv", index=False)
    print("\n=== FAILURE ANALYSIS (test; errors per true wall pixel) ===")
    print(err.round(4).to_string(index=False))

    # ---- 3. robustness sweep -----------------------------------------------------------
    sweeps = {name: damage_sweep(m, X_val, thresholds[name]) for name, m in models.items()}
    sw = pd.concat([pd.DataFrame(r).assign(model=n) for n, r in sweeps.items()])
    sw.to_csv(res_dir / "damage_sweep_all.csv", index=False)
    print("\n=== DAMAGE SWEEP (validation wall-F1) ===")
    print(sw.pivot_table(index="damage", columns="model", values="model_f1", sort=False)
          .assign(copy=pd.DataFrame(next(iter(sweeps.values()))).set_index("damage")["copy_f1"])
          .round(4).to_string())

    # ---- 4. generation -------------------------------------------------------------------
    gen_rows, samples = [], {}
    real = (X_train[np.random.default_rng(1).permutation(len(X_train))[:400]] > 0.5).astype(np.float32)
    gen_rows.append(generation_stats("real training plans", real, X_train))
    # reference for the memorisation check: how far a genuinely NEW real plan is
    # from its nearest training plan
    gen_rows.append(generation_stats("real test plans (reference)",
                                     (X_test > 0.5).astype(np.float32), X_train))
    recon = {n: (test_probs[n] > thresholds[n]).astype(np.float32) for n in models}
    for n in models:
        gen_rows.append(generation_stats(f"{n} test reconstructions", recon[n], X_train))
    if "DAE" in models:
        samples["DAE"] = (dae_mixture_samples(models["DAE"], X_train, 400, np.random.default_rng(123))
                          > thresholds["DAE"]).astype(np.float32)
        gen_rows.append(generation_stats("DAE perturbed-code samples", samples["DAE"], X_train))
    if "VAE" in models:
        samples["VAE"] = (models["VAE"].sample_prior(400, np.random.default_rng(123))
                          > thresholds["VAE"]).astype(np.float32)
        gen_rows.append(generation_stats("VAE prior samples z~N(0,I)", samples["VAE"], X_train))
        au, total = vae_active_units(models["VAE"], X_val)
        print(f"\nVAE active latent units: {au}/{total}")
    gen = pd.DataFrame(gen_rows)
    gen.to_csv(res_dir / "generation.csv", index=False)
    print("\n=== GENERATION STATISTICS ===")
    print(gen.round(4).to_string(index=False))

    summary = {"thresholds": thresholds,
               "best_epochs": {n: metas[n]["epoch"] for n in models},
               "vae_active_units": vae_active_units(models["VAE"], X_val) if "VAE" in models else None}
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- 5. figures ---------------------------------------------------------------------
    if args.no_figures or demo:
        return
    hist = {}
    for n in models:
        f = C.RESULTS_DIR / f"history_{n.lower()}.csv"   # written by train.py / select_models.py
        if f.exists():
            hist[n] = pd.read_csv(f).to_dict("list")
    figs = [plots.architecture_diagram(metas.get("GNN", {}).get("n_layers", C.GNN_LAYERS))]
    if len(hist) == len(models):
        figs.append(plots.training_curves(hist))
    test_rows = {r["model"]: r for r in rows if r["split"] == "test"}
    figs.append(plots.roc_curves(test_probs, X_test, test_rows["copy_input"]))
    figs.append(plots.confusion_matrices({n: test_rows[n] for n in models}))
    idx = np.random.default_rng(0).choice(len(X_test), size=5, replace=False)
    figs.append(plots.qualitative(X_test, test_noisy, test_probs, thresholds, idx))
    figs.append(plots.error_breakdown_bars(breakdowns_all))
    figs.append(plots.damage_sweep_plot(sweeps))
    if "DAE" in samples and "VAE" in samples:
        figs.append(plots.generation_grid(real, samples["DAE"], samples["VAE"]))
    print("\nSaved figures:", ", ".join(p.name for p in figs))


if __name__ == "__main__":
    main()
