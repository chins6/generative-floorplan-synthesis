"""Build the final report (DOCX) from the results CSVs and figures."""
import sys
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).parent))
from docx_helpers import (bullet, equation, equation_png, figure, new_document, para, table)

PROJ = Path(__file__).resolve().parents[1]   # project root (tools/ lives inside it)
RES, FIG = PROJ / "results", PROJ / "figures"
SCR = Path(__file__).parent / "eq"
SCR.mkdir(exist_ok=True)
OUT = Path(sys.argv[1])

comp = pd.read_csv(RES / "final_comparison.csv")
T = {r.model: r for r in comp[comp.split == "test"].itertuples()}
V = {r.model: r for r in comp[comp.split == "validation"].itertuples()}
err = pd.read_csv(RES / "error_analysis.csv").set_index("model")
gen = pd.read_csv(RES / "generation.csv").set_index("source")
sw = pd.read_csv(RES / "damage_sweep_all.csv")
vsel = pd.read_csv(RES / "selection_vae_beta.csv")
gsel = pd.read_csv(RES / "selection_gnn_layers.csv")
hist = {m: pd.read_csv(RES / f"history_{m.lower()}.csv") for m in ("DAE", "VAE", "GNN")}
f3 = lambda x: f"{x:.3f}"
vb = vsel[vsel.selected].iloc[0]
gb = gsel[gsel.selected].iloc[0]
sweep = lambda m, d: sw[(sw.model == m) & (sw.damage == d)].iloc[0]
real = gen.loc["real training plans"]
gtest = gen.loc["real test plans (reference)"]
dae_s = gen.loc["DAE perturbed-code samples"]
vae_s = gen.loc["VAE prior samples z~N(0,I)"]
RW, FW = RES / "walls", FIG / "walls"
compw = pd.read_csv(RW / "final_comparison.csv")
TW = {r.model: r for r in compw[compw.split == "test"].itertuples()}
errw = pd.read_csv(RW / "error_analysis.csv").set_index("model")
genw = pd.read_csv(RW / "generation.csv").set_index("source")
sww = pd.read_csv(RW / "damage_sweep_all.csv")
vselw = pd.read_csv(RW / "selection_vae_beta.csv")
realw, daew, vaew = (genw.loc["real training plans"], genw.loc["DAE perturbed-code samples"],
                     genw.loc["VAE prior samples z~N(0,I)"])
cvs = pd.read_csv(RES / "walls" / "cv_summary.csv")
cvf = pd.read_csv(RES / "walls" / "cv_folds.csv")
cv_txt = lambda m: ", ".join(f"{r.config.split('=')[1]}: {r.f1_mean:.3f} ± {r.f1_std:.3f}" for r in cvs[cvs.model == m].itertuples())
cv_sel = {m: cvs[(cvs.model == m) & cvs.selected].iloc[0].config.split("=")[1] for m in ("VAE", "GNN")}
sweepw = lambda m, d: sww[(sww.model == m) & (sww.damage == d)].iloc[0]


def side_by_side(doc, items, caption):
    t = doc.add_table(rows=1, cols=len(items))
    t.alignment = 1
    for cell, (png, w) in zip(t.rows[0].cells, items):
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.add_run().add_picture(str(png), width=Cm(w))
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for b in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{b}")
        e.set(qn("w:val"), "nil")
        borders.append(e)
    tblPr.append(borders)
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(8)
    c.paragraph_format.space_after = Pt(3)


def H(doc, text, lvl=1):
    return doc.add_heading(text, level=lvl)


doc = new_document()

# ---------------------------------------------------------------- title block
t = para(doc, "Repairing and Generating Floorplans on CubiCasa5K: a Denoising Autoencoder, "
         "a Variational Autoencoder and a Graph Neural Network Compared", align="center", size=14,
         after=2)
t.runs[0].bold = True
para(doc, "Chinmayi S (2024UG000194)  ·  Devananda (2024UG000216)  ·  Group 11  ·  "
     "Machine Learning course project, final review", align="center", size=9, after=6)

# ---------------------------------------------------------------- abstract
para(doc, f"We study how much floorplan structure small models can learn without supervision. "
     f"Each CubiCasa5K plan becomes a 32×32 binary wall map, 80% of its pixels are erased, and a "
     f"model must restore it. At the mid-review a convolutional denoising autoencoder (DAE) reached "
     f"a test wall-F1 of 0.503. Following the reviewers' feedback, we add a variational "
     f"autoencoder (VAE) and a GraphSAGE graph neural network (GNN), all implemented from scratch "
     f"in NumPy with numerically verified gradients and trained under one protocol: best "
     f"checkpoint and every hyper-parameter chosen on validation, a single benchmark on the 400 "
     f"unseen test plans. We run the full study twice: on the mid-review 'all dark ink' maps and "
     f"on CubiCasa5K's official wall polygons. The GNN repairs best on both: wall-F1 "
     f"**{f3(T['GNN'].wall_f1)}** (ink) and **{f3(TW['GNN'].wall_f1)}** (official walls; IoU "
     f"{f3(TW['GNN'].wall_iou)}, ROC-AUC {f3(TW['GNN'].roc_auc)}), against "
     f"{f3(T['DAE'].wall_f1)}/{f3(TW['DAE'].wall_f1)} for the DAE, "
     f"{f3(T['VAE'].wall_f1)}/{f3(TW['VAE'].wall_f1)} for the VAE and 0.33 for copying the input. "
     f"The VAE matches the DAE on repair and is the only model with a learned generative prior; "
     f"its β controls a clear trade-off between repair quality and coherent samples. For each "
     f"model we name its dominant failure, measure it, and report what was done about it.", bold_lead="Abstract.", align="justify", size=9)

# ---------------------------------------------------------------- 1 intro
H(doc, "1. Introduction")
para(doc, "Drafting a floorplan by hand is slow, and scanned or sketched plans are often "
     "incomplete. A model that has learned what a plausible layout looks like can **repair** a "
     "damaged plan and, if it has a usable latent distribution, **generate** new candidate plans. "
     "We frame both as unsupervised learning on CubiCasa5K. The input is a wall map with 80% of "
     "its pixels removed; the output is the per-pixel probability of a wall. The mid-review "
     "feedback asked us to (i) experiment with a VAE and a GNN and compare the results, and "
     "(ii) identify the failures of each model and the steps taken to handle them. This report "
     "answers both points. Our contributions are:", align="justify")
bullet(doc, "Three models on one task, data split, damage process, loss and metric set, so the "
       "comparison is controlled: DAE (21k parameters), β-VAE (26k) and residual GraphSAGE on a "
       "patch graph (52k).")
bullet(doc, "A strict protocol: best checkpoint by validation loss, β and depth chosen on "
       "validation, threshold chosen on validation, and a single benchmark on the test set.")
bullet(doc, "A failure taxonomy that splits every error into stroke thickening, hallucination, "
       "unrestored erasure or removal of surviving evidence. It turns 'the model is worse' into a "
       "named, measurable failure for each model.")
bullet(doc, "A second, label-clean experiment on the official SVG wall polygons, which separates "
       "failures caused by noisy labels from failures caused by the architectures.")

# ---------------------------------------------------------------- 2 related
H(doc, "2. Related Work")
para(doc, "Kalervo et al. [1] released CubiCasa5K with the official 4,200/400/400 split and a "
     "multi-task ResNet hourglass network (73.0% wall IoU). PCP-Net [2] recalibrates skip "
     "features so that thin walls survive down-sampling (81.9% wall IoU). Song & Yu [3] convert "
     "plans into region-adjacency graphs and classify elements with GraphSAGE-type GNNs (wall F1 "
     "0.947), which motivates our GNN. All three are supervised segmenters of the complete image. "
     "Our setting is unsupervised reconstruction of a heavily erased map, so the numbers are not "
     "directly comparable (Sections 5.2 and 6). The methods we build on are the denoising autoencoder [4], "
     "the VAE [5] with a β-weighted KL term [6] and KL annealing [7], and GraphSAGE [8].",
     align="justify")

# ---------------------------------------------------------------- 3 data
H(doc, "3. Dataset and Pre-processing")
para(doc, "CubiCasa5K contains 5,000 plans. We use the official split (4,200 train, 400 "
     "validation, 400 test, checked to be pairwise disjoint). Each full-resolution PNG is "
     "binarised (grey < 200 → wall) **before** nearest-neighbour resizing to 32×32, so 1-pixel "
     "lines are not averaged into grey. Only **12.6%** of pixels are walls, so a blank output "
     "already scores 86.6% accuracy. We therefore report wall precision, recall, F1 and IoU, "
     "plus the threshold-free pixel ROC-AUC. During training every pixel is erased with "
     "probability 0.8, and the corruption is re-sampled each epoch so that damage cannot be "
     "memorised. Validation and test are damaged once, with fixed seeds, so every model sees "
     "identical inputs. Damage only removes walls, so the copy-input baseline has precision 1 "
     "and its F1 depends only on the surviving fraction s. Sections 5–5.1 use these 'ink' maps; "
     "Section 5.2 repeats everything on the official wall polygons:", align="justify")
equation(doc, equation_png(r"F_{1}^{\mathrm{copy}} = \dfrac{2s}{1+s},\qquad s\approx 0.2 \;\Rightarrow\; F_{1}^{\mathrm{copy}}\approx 0.33", SCR / "copy.png"), 7.0, "1")

# ---------------------------------------------------------------- 4 method
H(doc, "4. Methodology")
para(doc, "All models output a logit per pixel and are trained with the same class-weighted "
     "binary cross-entropy (wall weight w = 6, normalised by the mean weight). With the sigmoid "
     "fused in, the gradient with respect to each logit is simply wᵢ(pᵢ − yᵢ)/W:", align="justify")
equation(doc, equation_png(r"\mathcal{L}_{\mathrm{rec}} = -\dfrac{1}{W}\,\Sigma_i\; w_i\,[\,y_i\log p_i + (1-y_i)\log(1-p_i)\,],\qquad w_i = 1 + (w-1)\,y_i,\quad W=\Sigma_i\, w_i", SCR / "bce.png"), 13.0, "2")
figure(doc, FIG / "fig_architecture.png", 16.5,
       "Figure 1. The three architectures (original diagram). Input, output, loss and training "
       "protocol are shared.")
para(doc, "Three stride-2 3×3 convolutions compress the 1×32×32 input to a 16×4×4 code "
     "(256 values, ReLU). Three convolution + 2× nearest-neighbour up-sampling stages and a 1×1 "
     "convolution decode it. This is the mid-review model, unchanged.",
     bold_lead="DAE.", align="justify")
para(doc, "Same backbone, but the last encoder stage is split into two heads that output μ and "
     "log σ² of q(z|x). We sample with the reparameterisation trick, and the KL term to the "
     "N(0, I) prior is divided by D = 1024 so that β = 1 is the usual ELBO:",
     bold_lead="VAE.", align="justify")
equation(doc, equation_png(r"z=\mu+\sigma\odot\varepsilon,\;\varepsilon\sim\mathcal{N}(0,I);\qquad \mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\frac{\beta}{D}\left(-\frac{1}{2}\sum_j(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)\right)", SCR / "vae.png"), 13.5, "3")
para(doc, "The hand-derived gradients are ∂L/∂μ = ∂L/∂z + cμ and ∂L/∂log σ² = ½ σ ε ∂L/∂z + "
     "½ c(σ² − 1), with c = β/(ND). β is ramped from 0 over the first 10 epochs, which guards "
     "against posterior collapse [7]. Repair uses z = μ; generation decodes z ~ N(0, I).",
     align="justify")
para(doc, "The image becomes a graph of 64 nodes, one per 4×4 patch, with 8-neighbour edges. "
     "Node features are the 16 damaged pixels plus the normalised (row, col) position: mean "
     "aggregation is permutation-invariant, so without the position the network could not "
     "locate a patch. A linear layer embeds each node into 64 dimensions, L residual GraphSAGE "
     "layers with a mean aggregator follow, and a linear read-out gives each node's 16 pixel "
     "logits:", bold_lead="GNN.", align="justify")
equation(doc, equation_png(r"h_u^{(l+1)} = h_u^{(l)} + \mathrm{ReLU}\left(W_s h_u^{(l)} + W_n \frac{1}{|\mathcal{N}(u)|}\sum_{v\in\mathcal{N}(u)} h_v^{(l)} + b\right)", SCR / "sage.png"), 9.8, "4")
para(doc, "Mini-batch Adam (lr 0.003, batch 32, 90 epochs, seed 42) is "
     "identical for all models. After every epoch the model is scored on the fixed damaged "
     "validation set, and the weights with the **lowest validation loss** are saved as the best "
     "checkpoint (for the VAE, only epochs after the warm-up). β ∈ {0.1, 0.5, 1, 4} and GNN depth "
     "L ∈ {2, 4, 6} were swept and chosen on validation, by rules fixed before the runs: "
     "the GNN takes the best validation F1; the VAE takes the best validation F1 among the β "
     "values whose prior samples are within ±50% of the real wall fraction and keep ≥ 25% of "
     "latent units active [9]. The decision threshold maximises validation F1. Test data is used "
     "only once, by reloading the saved best checkpoints. Every backward pass passes a "
     "float64 finite-difference check (|Δ| ≤ 10⁻⁵ + 10⁻³|g|).",
     bold_lead="Training and model selection.", align="justify")

# ---------------------------------------------------------------- 5 results
H(doc, "5. Results and Discussion")
rows = []
for m in ("copy_input", "DAE", "VAE", "GNN"):
    r, v = T[m], V[m]
    rows.append([m.replace("copy_input", "Copy input"), f"{int(r.params):,}" if m != "copy_input" else "–",
                 f"{r.threshold:.2f}", f3(v.wall_f1), f3(r.wall_precision), f3(r.wall_recall),
                 f3(r.wall_f1), f3(r.wall_iou), f3(r.roc_auc), f3(r.accuracy)])
table(doc, ["Model", "Params", "Thr.", "Val F1", "Test P", "Test R", "Test F1", "Test IoU",
            "Test AUC", "Test Acc."], rows,
      widths_cm=[2.2, 1.5, 1.1, 1.5, 1.5, 1.5, 1.5, 1.6, 1.6, 1.6], bold_rows=(3,),
      caption=f"Table 1. Held-out test results (400 plans) from the best checkpoints (epochs "
              f"{int(T['DAE'].best_epoch)}/{int(T['VAE'].best_epoch)}/{int(T['GNN'].best_epoch)}). "
              f"Selected configurations: VAE β = {vb.beta:g}, GNN L = {int(gb.n_layers)}. "
              f"Blank output: accuracy {f3(T['blank'].accuracy)}, F1 0.")
para(doc, f"The GNN is best on every wall metric: it improves the DAE's F1 by "
     f"{T['GNN'].wall_f1 - T['DAE'].wall_f1:+.3f} and its precision by "
     f"{T['GNN'].wall_precision - T['DAE'].wall_precision:+.3f}. AUC agrees "
     f"({f3(T['GNN'].roc_auc)} vs {f3(T['DAE'].roc_auc)}), so the gain is a better ranking of "
     f"wall pixels, not an artefact of the tuned threshold. The VAE is within 0.005 F1 of the "
     f"DAE. Validation and test scores differ by less than 0.02 for every model, so threshold "
     f"selection did not overfit validation. All three beat copying by ≥ 1.49×. The autoencoders' "
     f"accuracy ({f3(T['DAE'].accuracy)}) is below the blank baseline's ({f3(T['blank'].accuracy)}) "
     f"because restoring erased walls costs false positives on the 87% background. This is why "
     f"accuracy is not used to rank models.", align="justify")
figure(doc, FIG / "fig_training_curves.png", 16.5,
       "Figure 2. Training and validation loss; the dot marks the saved best checkpoint.")
para(doc, f"Validation loss tracks training loss for all models with no upward trend, so there is "
     f"no overfitting. Validation loss even sits slightly below training loss, because training "
     f"loss is averaged over an epoch of fresh corruptions while the weights are still moving, "
     f"and the VAE's training reconstruction also pays for sampling noise in z. The DAE and the "
     f"VAE plateau after about 10 epochs (DAE best {hist['DAE'].val_loss.min():.3f}); their 4×4 "
     f"bottleneck, not optimisation, limits them. The GNN is still improving at epoch 90 "
     f"(best epoch {int(T['GNN'].best_epoch)}), so it is under-trained if anything. In the β sweep, "
     f"validation F1 falls monotonically as β grows "
     f"({', '.join(f'{r.beta:g}: {r.val_wall_f1:.3f}' for r in vsel.itertuples())}), the expected "
     f"rate-distortion trade-off. Active latent units fall from "
     f"{int(vsel.active_units.iloc[0])} to {int(vsel.active_units.min())} of 256. In the depth sweep "
     f"the GNN's receptive field grows from {int(gsel.receptive_px.iloc[0])} to "
     f"{int(gsel.receptive_px.iloc[-1])} px and its F1 from {gsel.val_wall_f1.iloc[0]:.3f} to "
     f"{gsel.val_wall_f1.iloc[-1]:.3f}: completing a wall needs evidence from far along it.",
     bold_lead="Loss curves and model selection.", align="justify")
side_by_side(doc, [(FIG / "fig_confusion.png", 11.2), (FIG / "fig_roc.png", 5.4)],
             "Figure 3. Test confusion matrices at the validation-tuned thresholds (left) and "
             "pixel ROC curves (right).")

H(doc, "5.1 Failure analysis: what goes wrong in each model, and what we did", 2)
para(doc, f"The confusion matrices show the same total wall recall for the DAE and the VAE "
     f"(≈0.60), but their false-positive rates are far higher than the GNN's. To find out why, we "
     f"split each error by type (Fig. 4, per true wall pixel). A false positive within 1 px of a "
     f"true wall is **stroke thickening**; one further away is a **hallucination**. A missed wall "
     f"is either an **erased wall not restored** or a **surviving wall removed**, i.e. visible "
     f"evidence thrown away.", align="justify")
side_by_side(doc, [(FIG / "fig_qualitative.png", 7.0), (FIG / "fig_error_breakdown.png", 9.3)],
             "Figure 4. Random test plans (left) and the error taxonomy on the test set (right).")
e = err
frows = [
    ["DAE", f"Thick, blob-like strokes: {e.loc['DAE','fp_thickening']:.2f} thickening FPs per wall "
            f"pixel; predicts {e.loc['DAE','pred_wall_fraction']:.3f} wall vs {e.loc['DAE','true_wall_fraction']:.3f} true. "
            f"Removes {100 * (1 - e.loc['DAE','observed_walls_kept']):.1f}% of visible walls. No prior: perturbed-code samples "
            f"{dae_s.wall_fraction / real.wall_fraction:.1f}× too dense.",
     "Conv instead of dense AE (43× fewer params, no overfit); class weight + val-tuned threshold "
     "(0.65); prior problem handed to the VAE.",
     "Skip connections / U-Net, Dice or boundary loss."],
    ["VAE", f"Inherits the blobs ({e.loc['VAE','fp_thickening']:.2f} thickening). Prior samples are "
            f"realistic in density ({vae_s.wall_fraction:.3f}) but fragmentary and less diverse "
            f"({vae_s.diversity:.3f} vs {real.diversity:.3f}); walls do not close into rooms. "
            f"Posterior collapse at large β: β = 4 keeps only {int(vsel.active_units.iloc[-1])}/256 "
            f"units active and drops to val F1 {vsel.val_wall_f1.iloc[-1]:.3f}.",
     f"KL warm-up (10 ep.), best-checkpoint only after warm-up, β chosen on val ({vb.beta:g}), "
     f"active units monitored ({int(vb.active_units)}/256 active).",
     "Larger latent, hierarchical / conditional VAE, diffusion prior."],
    ["GNN", f"Under-completion: most unrestored erasures ({e.loc['GNN','fn_erased']:.2f}), predicts "
            f"only {e.loc['GNN','pred_wall_fraction']:.3f} wall; recall {T['GNN'].wall_recall:.2f}. "
            f"Keeps all visible walls (removes {e.loc['GNN','fn_ignored']:.3f}). Not generative.",
     f"Depth sweep (RF {int(gsel.receptive_px.iloc[0])}→{int(gb.receptive_px)} px, +"
     f"{gb.val_wall_f1 - gsel.val_wall_f1.iloc[0]:.3f} F1); residual layers to train 6 layers; "
     f"positional node features.",
     "Longer training, attention aggregator, graph pooling for global context."],
    ["All", f"Lose to copying under light damage; 'all dark ink' labels include text and "
            f"furniture, and colour-filled plans binarise to solid black (Fig. 5, top row).",
     "Damage sweep reports the crossover honestly; damage level justified by Eq. 1.",
     "Done: official wall labels (Section 5.2)."],
]
table(doc, ["Model", "Observed failure (test-set evidence)", "Steps already taken", "Next fix"],
      frows, widths_cm=[1.1, 7.6, 5.4, 3.3], font=7,
      caption="Table 2. Failure of each model, the evidence, and how it was handled.")
para(doc, f"Thickening accounts for {e.loc['DAE','fp_thickening'] / (e.loc['DAE','fp_thickening'] + e.loc['DAE','fp_hallucination']):.0%} "
     f"of the DAE's false positives. All information passes through a 4×4 grid without skip "
     f"connections and is expanded by nearest-neighbour up-sampling, so wall positions are "
     f"only known to about ±4 px. Under 80% erasure the cheapest way to cover a wall is a thick "
     f"stroke. The VAE shares this decoder and therefore the same failure. The GNN never routes "
     f"a node's own pixels through a bottleneck: each node sees its 16 inputs directly, which "
     f"acts like a skip connection. It is thin and precise, but it errs on the side of not "
     f"drawing: it is conservative where the CNNs are over-confident. We had expected the "
     f"isotropic mean aggregator to cause seams at patch borders. The data rejects this: F1 on "
     f"border pixels is {e.loc['GNN','f1_patch_border']:.3f} against {e.loc['GNN','f1_patch_interior']:.3f} "
     f"for interior pixels. Per-plan F1 correlates with wall density for every model (r = "
     f"{e.loc['DAE','plan_f1_vs_density_corr']:.2f} DAE, {e.loc['GNN','plan_f1_vs_density_corr']:.2f} GNN). "
     f"Sparse plans are hardest, because 20% of a sparse plan is too little evidence.",
     bold_lead="Why.", align="justify")

side_by_side(doc, [(FIG / "fig_generation.png", 10.4), (FIG / "fig_damage_sweep.png", 6.2)],
             "Figure 5. Generation from each model's latent space (left); wall-F1 against amount and "
             "type of damage, with grey marking damage types never seen in training (right).")
para(doc, f"Real plans have wall fraction {f3(real.wall_fraction)} and line continuity "
     f"{real.line_continuity:.0f}. The DAE, sampled by perturbing real codes, gives "
     f"{f3(dae_s.wall_fraction)} and {dae_s.line_continuity:.0f}: mostly solid ink, because its "
     f"ReLU code space has no density to sample from. The VAE's samples from N(0, I) give "
     f"{f3(vae_s.wall_fraction)} and {vae_s.line_continuity:.0f}, within 30% of real data. They "
     f"are not copies of training plans: their mean Hamming distance to the nearest training "
     f"plan is {f3(vae_s.nn_train_distance)}, slightly below that of a genuinely unseen test plan "
     f"({f3(gtest.nn_train_distance)}) but far from 0. So only the VAE gives a usable generative prior, as the "
     f"mid-review planned. Its samples are still fragments rather than closed rooms.",
     bold_lead="Generation.", align="justify")
d = lambda m, k: sweep(m, k).model_f1
para(doc, f"All models were trained at 80% dropout only. Copying wins at light damage, because "
     f"its precision is 1 (no damage: copy 1.000 vs GNN {d('GNN','none'):.3f}), so a learned "
     f"repair is useful only once the damage is heavy. The GNN overtakes copying from 60% "
     f"dropout ({d('GNN','drop 60%'):.3f} vs {sweep('GNN','drop 60%').copy_f1:.3f}); the "
     f"autoencoders only from 70%. On unseen block erasure combined with 50% dropout, all three "
     f"still beat copying (GNN {d('GNN','blocks+drop50 (unseen)'):.3f} vs "
     f"{sweep('GNN','blocks+drop50 (unseen)').copy_f1:.3f}). They learned to infer walls, not "
     f"one specific noise pattern.", bold_lead="Robustness.", align="justify")

H(doc, "5.2 Second experiment: official wall labels", 2)
para(doc, "The ink target mixes walls with text, furniture and dimension lines (Table 2, last row). "
     "We therefore repeated the whole pipeline (same models, damage, loss, sweeps, selection rules "
     "and test protocol) on CubiCasa5K's own annotation: every Wall polygon of floor 1 in model.svg, "
     "cropped to the walls' bounding box, padded to a square (no stretching), filled at 512×512 and "
     "reduced to 32×32. A cell is wall if ≥ 25% of its area is wall; 5% gave two-cell-thick walls "
     "and 35% broke thin walls apart. Walls cover 18.7% of pixels.", bold_lead="Labels.", align="justify")
rows3 = []
for m in ("copy_input", "DAE", "VAE", "GNN"):
    a, b = T[m], TW[m]
    th = (f"{err.loc[m,'fp_thickening']:.2f} → {errw.loc[m,'fp_thickening']:.2f}" if m != "copy_input" else "–")
    rows3.append([m.replace("copy_input", "Copy input"), f3(a.wall_f1), f3(a.wall_iou), f3(b.wall_precision),
                  f3(b.wall_recall), f3(b.wall_f1), f3(b.wall_iou), f3(b.roc_auc), th])
table(doc, ["Model", "Ink F1", "Ink IoU", "Walls P", "Walls R", "Walls F1", "Walls IoU", "Walls AUC",
            "Thickening ink → walls"], rows3, widths_cm=[2.0, 1.4, 1.4, 1.5, 1.5, 1.6, 1.7, 1.7, 3.0],
      bold_rows=(3,), caption="Table 3. Test results on both label sets (best checkpoints; β = 0.1 and "
      "L = 6 were again selected on validation). Thickening = false positives within 1 px of a wall, per true wall pixel.")
side_by_side(doc, [(FIG / "fig_labels_comparison.png", 9.6), (FW / "fig_qualitative.png", 6.6)],
             "Figure 6. Wall-F1 and IoU for both label sets (left); test plans repaired on official wall labels (right).")
para(doc, f"Every model improves markedly (F1 +{TW['DAE'].wall_f1 - T['DAE'].wall_f1:.2f} DAE, "
     f"+{TW['GNN'].wall_f1 - T['GNN'].wall_f1:.2f} GNN), while the copy baseline is unchanged "
     f"({f3(TW['copy_input'].wall_f1)}) because it depends only on the damage (Eq. 1). The gain "
     f"comes from cleaner targets, not easier inputs. The GNN reaches wall IoU "
     f"**{f3(TW['GNN'].wall_iou)}** and AUC {f3(TW['GNN'].roc_auc)}, and its lead over the "
     f"autoencoders widens ({f3(TW['GNN'].wall_f1)} vs {f3(TW['DAE'].wall_f1)}). Thickening falls for "
     f"every model, so part of the 'blob' failure was the network averaging over text and furniture. "
     f"But the DAE still thickens {errw.loc['DAE','fp_thickening'] / errw.loc['GNN','fp_thickening']:.1f}× "
     f"more than the GNN, so the bottleneck explanation of Section 5.1 stands. The GNN still "
     f"under-completes (predicts {errw.loc['GNN','pred_wall_fraction']:.3f} wall vs "
     f"{errw.loc['GNN','true_wall_fraction']:.3f} true) and now beats copying from 40% dropout "
     f"({sweepw('GNN','drop 40%').model_f1:.3f} vs {sweepw('GNN','drop 40%').copy_f1:.3f}). Its "
     f"{100 * TW['GNN'].wall_iou:.1f}% wall IoU approaches the 73–82% of supervised full-resolution "
     f"segmenters [1, 2]. The target class is now the same, but the task (repairing an 80%-erased "
     f"32×32 map) still differs.", bold_lead="Results.", align="justify")
_agree = cv_sel["VAE"] == f"{vselw[vselw.selected].iloc[0].beta:g}" and cv_sel["GNN"] == "6"
_gnn = cvs[cvs.model == "GNN"].sort_values("f1_mean")
_gap = _gnn.f1_mean.iloc[-1] - _gnn.f1_mean.iloc[-2]
para(doc, f"To check that β and L were not tuned to one lucky split, we ran 5-fold "
     f"cross-validation on the 4,200 training plans: 7 configurations × 5 folds = 35 runs of 90 "
     f"epochs under the same protocol, with the test set untouched. Held-out wall-F1 (mean ± std): "
     f"VAE β {cv_txt('VAE')}; GNN L {cv_txt('GNN')}. Applying the same selection rules to the fold "
     f"means picks β = {cv_sel['VAE']} and L = {cv_sel['GNN']}"
     + (", the same choices as the single validation split. " if _agree else
        ", which differs from the single validation split; the reported test numbers use the split-based choice. ")
     + f"Fold standard deviations are at most {cvs.f1_std.max():.3f}, "
     + ("smaller than the gap between the best and second-best GNN depth "
        f"({_gap:.3f}), so the ranking is not split noise."
        if _gap > _gnn.f1_std.max() else
        f"comparable to the gap between the best two GNN depths ({_gap:.3f}), so those two are statistically tied."),
     bold_lead="Cross-validation.", align="justify")
figure(doc, FW / "fig_vae_beta_samples.png", 11.5,
       "Figure 7. VAE samples z ~ N(0, I) on wall labels for each β (same random codes).")
para(doc, f"On wall labels the selected VAE (β = 0.1) produces samples with realistic statistics "
     f"(wall fraction {f3(vaew.wall_fraction)} vs {f3(realw.wall_fraction)}, continuity "
     f"{vaew.line_continuity:.0f} vs {realw.line_continuity:.0f}), but they are visibly fragments "
     f"(Fig. 7). This is the VAE's 'prior hole' failure: with a weak KL term the codes of real plans "
     f"do not fill N(0, I), so random codes land where the decoder was never trained. Larger β closes "
     f"the hole, and β = 1 gives coherent outer walls and rooms, at a cost of "
     f"{vselw.val_wall_f1.iloc[0] - vselw.val_wall_f1.iloc[2]:.3f} validation F1. Our selection rule, "
     f"fixed in advance, optimised repair; a generation-oriented system should use β ≈ 1. The DAE's "
     f"perturbed codes also look plan-like here, but only because each one is anchored on a real "
     f"training plan: that is resampling, not a learned prior.", bold_lead="Generation.", align="justify")

# ---------------------------------------------------------------- 6 limitations / future
H(doc, "6. Limitations and Future Work")
para(doc, "(i) Even on official wall labels, our task is repairing an 80%-erased 32×32 map, "
     "while [1, 2] segment full-resolution images; 64×64 and 128×128 inputs (needing a GPU "
     "framework) are the next step. (ii) The F1 gains depend on heavy damage, as the crossover "
     "shows. (iii) The repair-optimal VAE leaves prior holes, and even β = 1 samples lack "
     "guaranteed topological validity (closed rooms, doors). (iv) k-fold cross-validation was run "
     "for the wall labels only; the ink experiment relies on a single validation split. Next steps: "
     "a graph-level validity score from the GNN's node embeddings, a conditional VAE or diffusion "
     "prior, and skip connections or a Dice/boundary loss against thickening.", align="justify")

H(doc, "7. Conclusion")
para(doc, f"Under one controlled protocol and on two label sets, the GNN repairs damaged "
     f"floorplans best (test F1 {f3(T['GNN'].wall_f1)} on ink, {f3(TW['GNN'].wall_f1)} and IoU "
     f"{f3(TW['GNN'].wall_iou)} on official walls). Its patch nodes keep direct access to local "
     f"evidence, while the autoencoders lose wall positions in a 4×4 bottleneck and compensate with "
     f"thick strokes. Switching to official wall labels raised every model by 0.15–0.21 F1 and "
     f"reduced, but did not remove, the thickening failure. The VAE costs nothing in repair and "
     f"is the only model with a learned prior; β trades repair accuracy against coherent samples. "
     f"The three models fail in opposite ways (the autoencoders over-draw, the GNN under-draws), "
     f"which points to a hybrid of a VAE prior and GNN-style local refinement as the next model.", align="justify")

H(doc, "References")
refs = [
    "[1] A. Kalervo, J. Ylioinas, M. Häikiö, A. Karhu, J. Kannala. CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis. SCIA, 2019.",
    "[2] Zhou et al. PCP-Net: Biomimetic Cross-Scale Feature Recalibration with Axis-Decompositional Positional Embedding for Architectural Floor Plan Parsing. Biomimetics 11:581, 2026.",
    "[3] J. Song, K. Yu. Framework for Indoor Elements Classification via Inductive Learning on Floor Plan Graphs. ISPRS Int. J. Geo-Inf. 10:97, 2021.",
    "[4] P. Vincent, H. Larochelle, Y. Bengio, P.-A. Manzagol. Extracting and Composing Robust Features with Denoising Autoencoders. ICML, 2008.",
    "[5] D. P. Kingma, M. Welling. Auto-Encoding Variational Bayes. ICLR, 2014.",
    "[6] I. Higgins et al. β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR, 2017.",
    "[7] S. R. Bowman et al. Generating Sentences from a Continuous Space. CoNLL, 2016.",
    "[8] W. L. Hamilton, R. Ying, J. Leskovec. Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS, 2017.",
    "[9] Y. Burda, R. Grosse, R. Salakhutdinov. Importance Weighted Autoencoders. ICLR, 2016 (active-unit criterion).",
]
for r in refs:
    p = para(doc, r, size=7.5, after=0.5)

doc.save(OUT)
print("saved", OUT)
