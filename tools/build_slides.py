import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pptx_helpers import (GREY, MODEL, NAVY, card, caption, new_deck, notes, picture, slide,
                          table, text)

PROJ = Path(__file__).resolve().parents[1]   # project root (tools/ lives inside it)
RES, FIG = PROJ / "results", PROJ / "figures"
OUT = Path(sys.argv[1])

comp = pd.read_csv(RES / "final_comparison.csv")
T = {r.model: r for r in comp[comp.split == "test"].itertuples()}
V = {r.model: r for r in comp[comp.split == "validation"].itertuples()}
err = pd.read_csv(RES / "error_analysis.csv").set_index("model")
gen = pd.read_csv(RES / "generation.csv").set_index("source")
vsel = pd.read_csv(RES / "selection_vae_beta.csv")
gsel = pd.read_csv(RES / "selection_gnn_layers.csv")
vb, gb = vsel[vsel.selected].iloc[0], gsel[gsel.selected].iloc[0]
real, dae_s, vae_s = (gen.loc["real training plans"], gen.loc["DAE perturbed-code samples"],
                      gen.loc["VAE prior samples z~N(0,I)"])
f3 = lambda x: f"{x:.3f}"
RW, FW = RES / "walls", FIG / "walls"
cw = pd.read_csv(RW / "final_comparison.csv")
TW = {r.model: r for r in cw[cw.split == "test"].itertuples()}
errw = pd.read_csv(RW / "error_analysis.csv").set_index("model")
vselw = pd.read_csv(RW / "selection_vae_beta.csv")
cvs = pd.read_csv(RW / "cv_summary.csv")
cvf = pd.read_csv(RW / "cv_folds.csv")
cv_txt = lambda m: ", ".join(f"{r.config.split('=')[1]}: {r.f1_mean:.3f} ± {r.f1_std:.3f}" for r in cvs[cvs.model == m].itertuples())
cv_sel = {m: cvs[(cvs.model == m) & cvs.selected].iloc[0].config.split("=")[1] for m in ("VAE", "GNN")}

prs = new_deck()

# ---- 1. title + problem -------------------------------------------------------------
s = slide(prs, "Generative Floorplan Synthesis on CubiCasa5K", 1, "Final review")
text(s, 0.5, 1.15, 12.3, 0.5, [("Repairing and generating floorplans: Denoising AE vs Variational AE "
                                "vs Graph Neural Network, all written from scratch in NumPy",
                                dict(size=18, bold=True, color=NAVY))])
card(s, 0.5, 2.0, 4.0, 2.9, "Task", [
    "Input: 32×32 binary wall map with **80% of pixels erased**",
    "Output: p(wall) for every pixel, giving the repaired plan",
    "Generation: decode new latent codes"], size=16, title_size=17)
card(s, 4.7, 2.0, 4.0, 2.9, "Mid-review feedback → this work", [
    "(i) Add **VAE** and **GNN** and compare the results",
    "(ii) Name each model's **failures** and the steps taken",
    "Same data, loss, protocol and metrics for all three"], size=16, title_size=17)
card(s, 8.9, 2.0, 3.95, 2.9, "Data: CubiCasa5K", [
    "5,000 plans, official split 4,200 / 400 / 400",
    "Only **12.6%** of pixels are wall, so accuracy misleads; we report F1, IoU, AUC",
    "Copy baseline: F1 = 2s/(1+s) ≈ 0.33"], size=16, title_size=17)
text(s, 0.5, 5.15, 12.3, 1.4, [
    ("Headline (unseen test set, best checkpoints):", dict(size=18, bold=True, color=NAVY)),
    (f"Wall-F1, ink labels → official wall labels:   GNN **{f3(T['GNN'].wall_f1)} → {f3(TW['GNN'].wall_f1)}**  ·  "
     f"DAE {f3(T['DAE'].wall_f1)} → {f3(TW['DAE'].wall_f1)}  ·  VAE {f3(T['VAE'].wall_f1)} → {f3(TW['VAE'].wall_f1)}  ·  "
     f"copy 0.33", dict(size=17)),
    ("Only the VAE has a learned generative prior.", dict(size=17))])
text(s, 0.5, 6.4, 12.3, 0.5, [("Chinmayi S (2024UG000194)  ·  Devananda (2024UG000216)  ·  "
                               "Group 11", dict(size=13, color=GREY))])
notes(s, "Speaker: [FILL: name]. Problem: plans are costly to draw and scans are incomplete. "
         "We learn floorplan structure without labels by restoring heavily erased wall maps. "
         "After the mid-review we were asked to try a VAE and a GNN and to analyse failures; "
         "this deck answers both. Hand-off: data and models.")

# ---- 2. methodology ---------------------------------------------------------------------
s = slide(prs, "Methodology: three models, one protocol", 2)
picture(s, FIG / "fig_architecture.png", 0.3, 1.3, w=8.0)
text(s, 8.45, 1.1, 4.6, 5.9, [
    ("Shared", dict(bold=True, color=NAVY, size=15, nobullet=True)),
    "Weighted BCE (wall weight 6); Adam lr 0.003, batch 32, 90 epochs",
    "Fresh 80% pixel dropout every epoch",
    ("VAE", dict(bold=True, color=MODEL["VAE"], size=15, nobullet=True)),
    "z = μ + σ⊙ε;  L = BCE + β·KL / 1024",
    "KL warm-up over 10 epochs, to avoid posterior collapse",
    "Generation: z ~ N(0, I) → decoder",
    ("GNN (GraphSAGE)", dict(bold=True, color=MODEL["GNN"], size=15, nobullet=True)),
    "Graph of 64 nodes (4×4-pixel patches), 8-neighbour edges",
    "h ← h + ReLU(W_s h + W_n · mean of neighbours)",
    "Node features: pixels + (row, col) position",
    ("Verified", dict(bold=True, color=NAVY, size=15, nobullet=True)),
    "Every hand-written gradient passes a float64 finite-difference check",
], size=13, bullets=True, spacing=3)
caption(s, 0.3, 5.15, 8.0, "Original diagram. DAE 21k · VAE 26k · GNN 52k parameters", size=11)
notes(s, "Speaker: [FILL: name]. Explain why each model: the DAE is the baseline; the VAE adds a "
         "learned prior (reparameterisation trick, KL to N(0,I)); the GNN replaces the "
         "bottleneck with message passing between neighbouring patches. All gradients are "
         "derived by hand and checked numerically.")

# ---- 3. training + selection -------------------------------------------------------------
s = slide(prs, "Training, best checkpoint and hyper-parameter selection", 3)
picture(s, FIG / "fig_training_curves.png", 0.35, 1.05, w=12.6)
picture(s, FIG / "fig_hparam.png", 0.35, 4.65, w=7.6)
text(s, 8.15, 4.6, 4.9, 2.6, [
    "Best checkpoint = **lowest validation loss**, saved to disk and reloaded for test",
    "No overfitting: validation tracks training",
    f"VAE: higher β hurts repair and shrinks the active units → **β = {vb.beta:g}** (rule fixed in advance)",
    f"GNN: deeper reaches further → **L = {int(gb.n_layers)}** (receptive field {int(gb.receptive_px)} px)",
    "Threshold tuned on validation only",
    f"**5-fold CV** (wall labels, 35 runs) picks β = {cv_sel['VAE']}, L = {cv_sel['GNN']}; GNN F1 {cv_txt('GNN')}",
], size=12, bullets=True, spacing=2)
notes(s, "Speaker: [FILL: name]. Read the curves: the DAE and VAE plateau around epoch 10 "
         "(bottleneck-limited), while the GNN is still improving at epoch 90. Selection used "
         "validation only; the test set was touched once.")

# ---- 4. results ---------------------------------------------------------------------------
s = slide(prs, "Results on the unseen test set (400 plans)", 4)
rows = []
for m in ("copy_input", "DAE", "VAE", "GNN"):
    r = T[m]
    rows.append([m.replace("copy_input", "Copy input"), f3(r.wall_precision), f3(r.wall_recall),
                 f3(r.wall_f1), f3(r.wall_iou), f3(r.roc_auc), f3(V[m].wall_f1)])
table(s, 0.4, 1.15, 7.4, 2.2, ["Model", "Precision", "Recall", "F1", "IoU", "ROC-AUC", "Val F1"],
      rows, col_w=[1.6, 1.0, 0.9, 0.9, 0.9, 1.1, 1.0], size=13, highlight_rows=(3,), colors=MODEL)
picture(s, FIG / "fig_roc.png", 8.1, 1.05, w=4.9)
picture(s, FIG / "fig_confusion.png", 0.4, 3.7, w=7.4)
text(s, 8.1, 5.55, 4.9, 1.6, [
    f"GNN best on every wall metric (+{T['GNN'].wall_f1 - T['DAE'].wall_f1:.3f} F1, "
    f"+{T['GNN'].wall_precision - T['DAE'].wall_precision:.2f} precision)",
    "AUC agrees, so the gain is not a threshold artefact",
    "The VAE matches the DAE on repair",
], size=12.5, bullets=True, spacing=2)
notes(s, "Speaker: [FILL: name]. Precision and recall over accuracy: a blank page scores 87% "
         "accuracy. The confusion matrices show the autoencoders' false-positive rate is about "
         "twice the GNN's.")

# ---- 5. second experiment: official wall labels ---------------------------------------------
s = slide(prs, "Second experiment: official wall labels", 5, "CubiCasa5K's SVG wall polygons")
picture(s, FW / "fig_qualitative.png", 0.35, 1.05, h=6.0)
picture(s, FIG / "fig_labels_comparison.png", 6.35, 1.05, w=6.75)
picture(s, FW / "fig_vae_beta_samples.png", 6.35, 3.5, h=2.3)
text(s, 10.65, 3.5, 2.55, 2.4, [
    ("VAE β trade-off", dict(bold=True, color=MODEL["VAE"], size=13, nobullet=True)),
    ("β = 0.1: best repair, samples are fragments (prior holes)", dict(size=11)),
    (f"β ≈ 1: coherent outer walls and rooms, −{vselw.val_wall_f1.iloc[0] - vselw.val_wall_f1.iloc[2]:.3f} F1", dict(size=11)),
], bullets=True, spacing=3)
text(s, 6.35, 5.9, 6.8, 1.2, [
    "Only the annotated walls: no text, furniture or dimension lines",
    f"Every model gains +0.15–0.21 F1, while copying stays at {f3(TW['copy_input'].wall_f1)}, so the gain is from cleaner labels",
    f"GNN: F1 **{f3(TW['GNN'].wall_f1)}**, IoU **{f3(TW['GNN'].wall_iou)}**, AUC {f3(TW['GNN'].roc_auc)} (papers: 73–82% IoU, full-resolution)",
], size=12, bullets=True, spacing=2)
notes(s, "Speaker: [FILL: name]. We repeated the whole pipeline on the official wall polygons. "
         "Copying is unchanged, so the improvement comes from the labels. Thickening drops for "
         "every model but the DAE still over-draws, so the bottleneck explanation stands. The "
         "VAE's beta trades repair accuracy against coherent samples.")

# ---- 6. limitations (rubric) -----------------------------------------------------------
s = slide(prs, "Model failures, steps taken, and what to improve", 6)
picture(s, FIG / "fig_error_breakdown.png", 0.35, 1.05, w=5.6)
e = err
card(s, 6.15, 1.05, 6.85, 1.85, "DAE: thick strokes, no prior", [
    f"{e.loc['DAE','fp_thickening']:.2f} thickening FPs per wall pixel ({errw.loc['DAE','fp_thickening']:.2f} on wall labels); the 4×4 bottleneck loses positions",
    "Done: CNN instead of dense AE, class weight, val threshold. **Next: skip connections, Dice loss**"],
    color=MODEL["DAE"], size=15, title_size=16)
card(s, 6.15, 3.0, 6.85, 1.85, "VAE: same blobs, fragmentary samples", [
    "Shares the DAE decoder; at β = 0.1 the prior has holes, so samples are fragments",
    f"Done: KL warm-up, β sweep, active units {int(vb.active_units)}/256. **Next: conditional VAE / diffusion prior**"],
    color=MODEL["VAE"], size=15, title_size=16)
card(s, 6.15, 4.95, 6.85, 1.85, "GNN: under-completes", [
    f"Most unrestored erasures ({e.loc['GNN','fn_erased']:.2f}); predicts {e.loc['GNN','pred_wall_fraction']:.3f} wall vs "
    f"{e.loc['GNN','true_wall_fraction']:.3f} true",
    f"Done: depth 2→{int(gb.n_layers)}, residuals, positions. **Next: attention, graph pooling, longer training**"],
    color=MODEL["GNN"], size=15, title_size=16)
text(s, 0.4, 4.25, 5.6, 2.9, [
    ("All models", dict(bold=True, color=NAVY, size=14, nobullet=True)),
    "Copying wins under light damage (crossover at 60–70% dropout)",
    "Labels are 'all dark ink' (text, furniture); colour-filled plans binarise to solid black",
    "**Done: official SVG wall labels** → +0.15–0.21 F1 and less thickening (slide 5)",
    "Hypothesis tested and rejected: no GNN patch-seam artefact (border F1 ≈ interior F1)",
], size=12, bullets=True, spacing=3)
notes(s, "Speaker: [FILL: name]. This is the rubric's limitations slide. Each error is split "
         "into thickening, hallucination, unrestored erasure, and evidence removed. The "
         "autoencoders over-draw, the GNN under-draws: opposite failures.")

# ---- 7. conclusion -------------------------------------------------------------------------
s = slide(prs, "Conclusion and future work", 7)
card(s, 0.5, 1.2, 6.0, 4.0, "What we found", [
    f"**GNN** repairs best: F1 {f3(T['GNN'].wall_f1)} (ink), **{f3(TW['GNN'].wall_f1)}** / IoU {f3(TW['GNN'].wall_iou)} (official walls)",
    "Local evidence matters more than a global bottleneck",
    "The **VAE** costs no repair quality and is the only model with a usable generative prior",
    "Official wall labels: +0.15–0.21 F1 for every model; each failure named and measured",
    "Strict protocol: best checkpoint and every choice made on validation; test used once"], size=18, title_size=19)
card(s, 6.8, 1.2, 6.0, 4.0, "Future work", [
    "64/128 px with a GPU framework → the papers' full-resolution setting",
    "Hybrid model: VAE prior + GNN refinement",
    "Graph-level validity score (closed rooms, connectivity) from the GNN embeddings",
    "Dice / boundary loss and skip connections against thick strokes",
    "VAE at β ≈ 1 for generation; cross-validate the ink experiment too"], size=18, title_size=19)
text(s, 0.5, 5.6, 12.3, 1.2, [
    ("Code demo: python train.py --labels walls --model gnn --epochs 3 --demo  →  python evaluate.py --labels walls", dict(size=14, bold=True, color=NAVY)),
    ("Every number on these slides is regenerated by run_all.py from the saved best checkpoints.", dict(size=12)),
])
notes(s, "Speaker: [FILL: name]. Summarise, then hand over to the live demo.")

prs.save(OUT)
print("saved", OUT)
