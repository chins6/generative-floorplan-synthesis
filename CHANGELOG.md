# Final-review changelog (2026-11-23)

## Mid-review feedback → what was done
| Feedback | Response |
|---|---|
| (i) Experiment with VAE and GNN and compare the results | Added a convolutional β-VAE and a residual GraphSAGE GNN (patch graph). All three models share the data, damage, loss, optimiser, protocol and metrics. Comparison: Table 1 in the report, slide 4. |
| (ii) Identify each model's failures and the steps already taken | Added an error taxonomy (thickening / hallucination / unrestored erasure / evidence removed), posterior-collapse monitoring (active units), a patch-seam test and a damage sweep. Results: Table 2 in the report, slide 6. |

## Rubric items → where they are
| Rubric item | Where |
|---|---|
| Modular, documented code | `floorplan/` package with 10 modules and 5 CLI scripts |
| Loss/accuracy plots | `fig_training_curves.png` |
| Live training for a few epochs | `train.py --epochs N --demo`, `Final_Demo.ipynb` section 2 |
| Best checkpoint used for the test benchmark | `trainer.py` saves the lowest-val-loss weights; `evaluate.py` reloads them |
| Hyper-parameters justified by validation | `select_models.py` (β sweep, depth sweep), with rules fixed in advance |
| Report ≤ 5 pages, Abstract→Conclusion, formulas, Results & Discussion with error analysis, original diagrams | `report/ML_Project_Report.pdf`, 5 pages including references |
| ≤ 7 slides, model-limitations slide, original plots | `presentation/` (PPTX + PDF), slide 6 |
| README: how to run, dependencies, modules, links | `README.md` |

## Second experiment: official wall labels (added the evening of 2026-11-23)
* `data.load_svg_walls`: takes floor 1's `class="Wall"` polygons from `model.svg`, crops to the walls' bounding box, pads to a square, fills at 512×512 and reduces to 32×32 (a cell is wall if ≥ 25% of it is covered; chosen by visual check of 0.05 / 0.15 / 0.25 / 0.35).
* `--labels walls` flag on `train.py`, `select_models.py` and `evaluate.py`. Outputs go to `cache/*_walls.npy`, `models/walls/`, `results/walls/` and `figures/walls/`, so the ink results are untouched.
* The whole pipeline was re-run on wall labels: all 8 training runs, selection (again β = 0.1, L = 6) and the test benchmark.
* `compare_labels.py` puts both label sets side by side. `select_models.py` also draws the β prior-sample figure.
* Result: every model gains +0.15–0.21 F1 (GNN 0.578 → 0.786, IoU 0.648); copy is unchanged at 0.33.
* New finding: the VAE's β trades repair accuracy against coherent samples (prior holes at β = 0.1).
* Report: new Section 5.2, Table 3 and Figures 6–7; abstract, limitations and conclusion updated. Slides: slide 5 is now this experiment; slides 1, 6 and 7 updated.

## 5-fold cross-validation (added the evening of 2026-11-23)
* `cross_validate.py`: splits the 4,200 training plans into 5 folds (fixed seed) and runs 7 configurations (VAE β ∈ {0.1, 0.5, 1, 4}, GNN L ∈ {2, 4, 6}) × 5 folds = 35 runs of 90 epochs, with the same protocol and best-checkpoint rule. It applies the select_models rules to the fold means. The test set is untouched.
* Result (wall labels): CV picks **β = 0.1** (F1 0.650 ± 0.002) and **L = 6** (F1 0.779 ± 0.002), the same as the single validation split. Fold std ≤ 0.003 is smaller than the gaps between configurations.
* Outputs: `results/walls/cv_folds.csv`, `results/walls/cv_summary.csv`, `figures/walls/fig_cv.png`. Report: a "Cross-validation" paragraph in Section 5.2, and limitation (iv) updated. Slides: a CV bullet on slide 3.
* Feasibility note: CV was run for the wall labels only; the ink experiment keeps its single validation split.

## Code changes
* New package `floorplan/`: the mid-review DAE code was moved into reusable modules unchanged (it reproduces the mid-review numbers, test F1 0.501 vs 0.503). Added `VAE`, `GNN`, a generic gradient check, a trainer with best-checkpoint saving, checkpoint I/O, analysis and plots.
* Fix: the mid-review notebook saved the *last* epoch; checkpoints are now the *best* validation epoch.
* Fix: during the VAE's KL warm-up the objective changes, so warm-up epochs are not eligible as "best". Without this, β = 4 picked epoch 5. Runs shorter than the warm-up (live demo) still save.

## Feasibility notes
* Hyper-parameters were selected on the official validation split, not k-fold cross-validation (compute budget). This is stated in the report as future work.
* The 5.5 GB raw dataset is not in the ZIP; the 20 MB preprocessed cache is, so everything runs without it.

## Still to fill in
* ~~Group ID~~ done: Group 11 (report, slide 1, ZIP name)
* ~~Team-lead name~~ done: ZIP is `ML_Project_Group11_Chinmayi.zip`
* Drive/GitHub link: README section 2
* Speaker per slide: PPTX speaker notes
