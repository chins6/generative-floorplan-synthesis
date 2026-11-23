import sys
import nbformat as nbf

out = sys.argv[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Final review — code demonstration
**Generative floorplan synthesis on CubiCasa5K: DAE vs VAE vs GNN**

This notebook is the demo the rubric asks for:

1. load the cached data and verify every hand-written backward pass,
2. **live-train** each model for a few epochs (written to `models/demo/`, so the submitted checkpoints are never overwritten),
3. load the **best checkpoints** (lowest validation loss) and **benchmark them on the unseen test set**,
4. show the loss curves, ROC curves, confusion matrices and failure analysis.

All models are NumPy code written from scratch, in the `floorplan/` package (see `README.md`)."""),
code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from floorplan import config as C
from floorplan.data import load_splits, fixed_noisy_sets
from floorplan.models import build_model
from floorplan.trainer import fit
from floorplan.checkpoint import load_checkpoint
from floorplan.analysis import evaluate_model, baseline_rows
from floorplan.gradcheck import gradient_check

X_train, X_val, X_test = load_splits()
val_noisy, test_noisy = fixed_noisy_sets(X_val, X_test)"""),
md("## 1. Are the hand-written gradients correct?"),
code("""for name in ("DAE", "VAE", "GNN"):
    gradient_check(name, seed=0)"""),
md("""## 2. Live training (a few epochs per model)
Change any argument here during the demo, e.g. `n_layers=2` for the GNN, `beta=4.0` for the VAE or `width=8` for the DAE."""),
code("""LIVE = {"DAE": dict(width=16), "VAE": dict(width=16, beta=1.0), "GNN": dict(hidden=64, n_layers=6)}
EPOCHS = {"DAE": 2, "VAE": 2, "GNN": 3}
live_hist = {}
for name, kw in LIVE.items():
    model = build_model(name, **kw)
    print(f"{name}: {model.n_params():,} parameters")
    live_hist[name] = fit(model, X_train, X_val, val_noisy, epochs=EPOCHS[name], print_every=1,
                          ckpt_path=C.MODELS_DIR / "demo" / f"best_{name.lower()}.npz",
                          kl_warmup=C.VAE_KL_WARMUP if name == "VAE" else 0)"""),
code("""fig, ax = plt.subplots(figsize=(5, 3))
for name, h in live_hist.items():
    ax.plot(h["epoch"], h["val_loss"], "o-", label=name)
ax.set_xlabel("epoch"); ax.set_ylabel("validation loss"); ax.set_title("Live run"); ax.legend(); plt.show()"""),
md("""## 3. Benchmark the BEST checkpoints on the unseen test set
The threshold for every model is tuned on validation and then frozen."""),
code("""rows = []
for name, res in baseline_rows(X_val, val_noisy, X_test, test_noisy, X_train.mean()).items():
    rows.append(dict(model=name, **res["test"]))
for name in ("DAE", "VAE", "GNN"):
    model, meta = load_checkpoint(C.MODELS_DIR / f"best_{name.lower()}.npz")
    res, _, _, thr = evaluate_model(model, X_val, val_noisy, X_test, test_noisy)
    rows.append(dict(model=name, best_epoch=meta["epoch"], params=model.n_params(), **res["test"]))
cols = ["model", "best_epoch", "params", "threshold", "wall_precision", "wall_recall", "wall_f1", "wall_iou", "roc_auc", "accuracy"]
pd.DataFrame(rows)[cols].round(4)"""),
md("## 4. Training curves of the submitted runs, ROC and confusion matrices"),
code("""for f in ["fig_training_curves.png", "fig_roc.png", "fig_confusion.png"]:
    display(Image(filename=str(C.FIGURES_DIR / f), width=820))"""),
md("## 5. Failure analysis and generation"),
code("""display(pd.read_csv(C.RESULTS_DIR / "error_analysis.csv").round(3))
for f in ["fig_error_breakdown.png", "fig_qualitative.png", "fig_damage_sweep.png", "fig_generation.png"]:
    display(Image(filename=str(C.FIGURES_DIR / f), width=760))"""),
code("""display(pd.read_csv(C.RESULTS_DIR / "generation.csv").round(4))
display(pd.read_csv(C.RESULTS_DIR / "selection_vae_beta.csv").drop(columns=["path"]).round(4))
display(pd.read_csv(C.RESULTS_DIR / "selection_gnn_layers.csv").drop(columns=["path"]).round(4))"""),
]
cells += [
md("""## 6. Second experiment: official wall labels
Same models and protocol, but the target is now only the annotated **wall polygons** of `model.svg`
(no text, furniture or dimension lines). Everything for this experiment lives in `models/walls/`,
`results/walls/` and `figures/walls/`."""),
code("""C.set_labels("walls")
Xw_train, Xw_val, Xw_test = load_splits()
wv_noisy, wt_noisy = fixed_noisy_sets(Xw_val, Xw_test)
rows = []
for name, res in baseline_rows(Xw_val, wv_noisy, Xw_test, wt_noisy, Xw_train.mean()).items():
    rows.append(dict(model=name, **res["test"]))
for name in ("DAE", "VAE", "GNN"):
    model, meta = load_checkpoint(C.MODELS_DIR / f"best_{name.lower()}.npz")
    res, _, _, thr = evaluate_model(model, Xw_val, wv_noisy, Xw_test, wt_noisy)
    rows.append(dict(model=name, best_epoch=meta["epoch"], params=model.n_params(), **res["test"]))
pd.DataFrame(rows)[cols].round(4)"""),
code("""for f in [C.PROJECT_DIR / "figures" / "fig_labels_comparison.png",
          C.FIGURES_DIR / "fig_qualitative.png", C.FIGURES_DIR / "fig_generation.png"]:
    display(Image(filename=str(f), width=760))"""),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, out)
print("wrote", out)
