# Generative Floorplan Synthesis on CubiCasa5K — DAE vs VAE vs GNN

Machine Learning course project, final review.
Team: Chinmayi S (2024UG000194), Devananda (2024UG000216).

**Task.** Each CubiCasa5K floorplan becomes a 32×32 binary wall map. 80% of its pixels are
erased, and a model has to **restore the plan**. Following the feedback from the mid-review
we compare three models on exactly the same data, damage, loss and metrics:

| Model | Idea | Parameters |
|---|---|---|
| **DAE** | Convolutional denoising autoencoder (the mid-review model, unchanged) | 21,025 |
| **VAE** | Same convolutional backbone. The encoder outputs μ and log σ², so the decoder can also be sampled from N(0, I) to **generate** new plans | 25,649 |
| **GNN** | GraphSAGE on a graph of 4×4-pixel patches (64 nodes, 8-neighbour edges). Each node predicts its own 16 pixels | 35,280 |

Two label sets are used, under the same protocol:
* **`ink`** (the mid-review representation): every dark pixel of the floorplan PNG. This includes text,
  furniture symbols and dimension lines.
* **`walls`** (final review): only the **official wall polygons** from each plan's `model.svg`
  annotation, drawn and reduced to 32×32. Add `--labels walls` to any command.

Every model is written **from scratch in NumPy**: forward pass, hand-derived backward pass and
the Adam optimiser. No deep-learning framework is used. `gradient_check.py` checks each
backward pass against finite differences.

---

## 1. Quick start

```bash
# Python 3.10+ (developed on 3.13)
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python gradient_check.py           # 1. verify all hand-written gradients      (~1 min)
python train.py --model gnn --epochs 3 --demo      # 2. LIVE training demo  (~10 s)
python evaluate.py                 # 3. benchmark the BEST checkpoints on test (~1 min)
```

During the demo, run `Final_Demo.ipynb` top to bottom. It does the same three steps
inside a notebook and also shows every plot.

### Live training during the demonstration
```bash
python train.py --model dae --epochs 2 --demo
python train.py --model vae --epochs 2 --demo --beta 1.0
python train.py --model gnn --epochs 3 --demo --layers 4 --hidden 64
python evaluate.py --models-dir models/demo     # benchmark the demo checkpoints

python train.py --labels walls --model gnn --epochs 3 --demo     # same, on the official wall labels
python evaluate.py --labels walls                               # benchmark the wall-label best checkpoints
```
`--demo` writes to `models/demo/` and `results/demo/`, so the submitted checkpoints are never
overwritten. You can change any hyper-parameter live (`--layers`, `--hidden`, `--width`,
`--beta`, `--lr`, `--batch-size`, `--subset N`).

### Reproduce every number in the report
```bash
python run_all.py                  # both label sets, ~30–45 min on a laptop CPU
```
For each label set, this runs the gradient check, then 90-epoch training of the DAE, the VAE β-sweep {0.1, 0.5, 1, 4}
and the GNN depth sweep {2, 4, 6}. Next it selects β and depth on the **validation** set
(`select_models.py`), and finally benchmarks the selected best checkpoints on the **test** set
(`evaluate.py`). `compare_labels.py` then puts the two label sets side by side. `python cross_validate.py --labels walls`
(35 runs, ~45 min on 14 CPU workers) re-checks β and depth with 5-fold cross-validation on the
training plans. The commands are deterministic, with seed 42 for training and fixed
corruption seeds for val/test.

---

## 2. Data

* **Dataset:** CubiCasa5K (Kalervo et al., 2019), 5,000 floorplans, official split
  4,200 / 400 / 400 (`train.txt`, `val.txt`, `test.txt`).
* **Pre-processed cache (shipped in this ZIP):** `cache/cache_{train,val,test}_32.npy` (ink) and
  `cache/cache_{train,val,test}_32_walls.npy` (official wall labels).
  All code runs from these files, so **the 5.5 GB raw dataset is not needed** to train or evaluate.
* **Raw dataset (only to rebuild the cache):** download CubiCasa5K from Kaggle and unzip it so
  that `archive/cubicasa5k/cubicasa5k/train.txt` exists. Delete `cache/` and the code
  rebuilds it (≈3 min).
* **Drive / GitHub links:** [FILL: Google Drive or GitHub link to the dataset, cache and checkpoints, shared with the course instructor]

Wall labels (`--labels walls`): parse every `class="Wall"` polygon of floor 1 in `model.svg`, crop to
the walls' bounding box, pad to a square (no stretching), fill at 512×512, and reduce to 32×32. A cell is
wall if ≥ 25% of its area is wall (5% gives 2-cell-thick walls; 35% breaks thin walls apart). 18.7% of
pixels are wall.

Ink labels (mid-review) pre-processing: grey level < 200 → wall (binarised at full resolution) → nearest-neighbour
resize to 32×32. Training corruption: each pixel is erased with probability 0.8, re-sampled
every epoch. Validation and test are damaged once, with fixed seeds (7 and 99), so all models
are scored on identical inputs.

---

## 3. Code layout

```
floorplan/                 the package (all models and logic)
  config.py                paths, seeds, shared hyper-parameters
  data.py                  split files, PNG -> 32x32 wall map, cache, corruption
  layers.py                im2col convolution, Linear, ReLU, Upsample, sigmoid, Adam
  losses.py                weighted BCE (+ its fused gradient), F1 / IoU / AUC, threshold search
  models.py                DAE, VAE (reparameterisation + KL), GNN (GraphSAGE, patch graph)
  trainer.py               mini-batch training loop, best-checkpoint saving (lowest val loss)
  checkpoint.py            save / load weights + metadata (.npz)
  gradcheck.py             finite-difference gradient check for any model
  analysis.py              evaluation, failure taxonomy, damage sweep, generation statistics
  plots.py                 every figure in the report and slides
train.py                   CLI: train one model, keep its best checkpoint
select_models.py           CLI: choose VAE beta / GNN depth on validation
evaluate.py                CLI: benchmark best checkpoints on test, write tables + figures
gradient_check.py          CLI: verify all backward passes
compare_labels.py          CLI: ink vs official wall labels, side by side
cross_validate.py          CLI: 5-fold cross-validation of VAE beta / GNN depth (training plans only)
run_all.py                 CLI: reproduce everything
Final_Demo.ipynb           demo notebook (gradient check, live training, test benchmark, plots)
Generative_Floorplan_Project.ipynb   stage-1 (mid-review) notebook: DAE exploration
models/best_{dae,vae,gnn}.npz        BEST checkpoints (ink labels) used for every reported test number
models/walls/                        the same, for the official wall labels
results/walls/, figures/walls/       tables and figures of the wall-label experiment
models/best_vae_beta*.npz, best_gnn_L*.npz   checkpoints of the hyper-parameter sweeps
results/*.csv              all tables (comparison, error analysis, sweeps, histories)
figures/*.png              all figures
```

### Checkpointing and evaluation protocol
1. After every epoch the model is scored on the damaged validation set. The weights with the
   **lowest validation loss** are saved to `models/best_<model>.npz`.
2. The decision threshold is tuned on **validation** (maximise wall-F1) and then frozen.
3. `evaluate.py` reloads the best checkpoint from disk and reports results on the **test**
   set, which training and model selection never touch.

---

## 4. Results (held-out test set, 400 plans)

### 4a. Official wall labels (final-review experiment)

Best checkpoints (`models/walls/`): DAE epoch 84, VAE (β = 0.1) epoch 82, GNN (6 layers) epoch 89.
β and depth were again selected on validation.

| Model | Precision | Recall | **Wall F1** | Wall IoU | ROC-AUC |
|---|---|---|---|---|---|
| Copy input (baseline) | 1.000 | 0.198 | 0.331 | 0.198 | 0.599 |
| DAE | 0.563 | 0.786 | 0.656 | 0.488 | 0.902 |
| VAE | 0.548 | 0.810 | 0.654 | 0.486 | 0.900 |
| **GNN** | **0.840** | 0.740 | **0.786** | **0.648** | **0.945** |

Compared with the ink labels below, every model gains 0.15–0.21 F1 while the copy baseline is unchanged.
The gain therefore comes from cleaner labels, not easier damage. The VAE's β is a trade-off on these labels:
β = 0.1 repairs best, but its prior samples are fragments; β ≈ 1 gives coherent plans with rooms, at a
cost of 0.012 validation F1 (`figures/walls/fig_vae_beta_samples.png`).

### 4b. All dark ink (mid-review representation)

Best checkpoints: DAE epoch 90, VAE (β = 0.1) epoch 81, GNN (6 layers) epoch 84. Thresholds
were tuned on validation.

| Model | Precision | Recall | **Wall F1** | Wall IoU | ROC-AUC | Accuracy |
|---|---|---|---|---|---|---|
| Copy input (baseline) | 1.000 | 0.200 | 0.333 | 0.200 | 0.600 | 0.893 |
| DAE | 0.430 | 0.600 | 0.501 | 0.334 | 0.831 | 0.840 |
| VAE | 0.422 | 0.608 | 0.498 | 0.331 | 0.828 | 0.835 |
| **GNN** | **0.631** | 0.534 | **0.578** | **0.407** | **0.855** | **0.895** |

| Generation | Wall fraction | Line continuity | Diversity |
|---|---|---|---|
| Real plans | 0.128 | 61 | 0.229 |
| DAE, perturbed real codes | 0.517 | 473 | 0.481 |
| VAE, z ~ N(0, I) | 0.094 | 68 | 0.174 |

**Main failure of each model** (`results/error_analysis.csv`):
* DAE: thick, blob-like strokes (0.69 thickening false positives per true wall pixel). It has no prior to sample from.
* VAE: the same blobs, since it shares the decoder. Prior samples are fragmentary. β = 4 collapses most latent units (78/256 active).
* GNN: under-completion. It leaves the most erased walls unrestored (0.47 per true wall pixel), but never removes a visible wall.

See `results/final_comparison.csv`, `results/error_analysis.csv`, `results/generation.csv` and
the report for the full analysis.

---

## 5. Dependencies

`numpy`, `pandas`, `matplotlib`, `scikit-learn` (for ROC-AUC only), `pillow` (only to rebuild the cache),
and `jupyter` (for the notebooks). See `requirements.txt`.
