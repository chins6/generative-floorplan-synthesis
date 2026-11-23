"""Evaluation, model-failure analysis and generation statistics.

Everything here works on a trained model and the fixed damaged val/test sets.
The decision threshold is always chosen on VALIDATION and then frozen for test.
"""
import numpy as np

from . import config as C
from .data import corrupt
from .losses import (bce_loss, best_threshold, classification_metrics, confusion,
                     pixel_auc)


def batched_predict(model, x, batch=400):
    return np.concatenate([model.predict(x[i:i + batch]) for i in range(0, len(x), batch)])


# ---- headline evaluation ----------------------------------------------------------
def evaluate_model(model, X_val, val_noisy, X_test, test_noisy):
    """Tune the threshold on validation, then score validation and test."""
    val_prob = batched_predict(model, val_noisy)
    test_prob = batched_predict(model, test_noisy)
    thr, _ = best_threshold(val_prob, X_val)
    rows = {}
    for split, prob, clean in (("validation", val_prob, X_val), ("test", test_prob, X_test)):
        row = {"loss": bce_loss(prob, clean, model.wall_weight), "threshold": thr}
        row.update(classification_metrics(prob, clean, thr))
        row["roc_auc"] = pixel_auc(prob, clean)
        tp, fp, fn, tn = confusion(prob, clean, thr)
        row.update(tp=tp, fp=fp, fn=fn, tn=tn)
        rows[split] = row
    return rows, val_prob, test_prob, thr


def baseline_rows(X_val, val_noisy, X_test, test_noisy, train_wall_fraction):
    """copy-input and blank baselines (both threshold-independent)."""
    out = {}
    for name, fn in (("copy_input", lambda noisy, clean: noisy),
                     ("blank", lambda noisy, clean: np.zeros_like(clean))):
        out[name] = {}
        for split, noisy, clean in (("validation", val_noisy, X_val), ("test", test_noisy, X_test)):
            pred = fn(noisy, clean)
            row = {"loss": bce_loss(pred, clean), "threshold": 0.5}
            row.update(classification_metrics(pred, clean, 0.5))
            row["roc_auc"] = pixel_auc(pred, clean) if pred.max() > 0 else 0.5
            tp, fp, fn_, tn = confusion(pred, clean, 0.5)
            row.update(tp=tp, fp=fp, fn=fn_, tn=tn)
            out[name][split] = row
    return out


# ---- failure analysis -------------------------------------------------------------
def _dilate(mask):
    """8-neighbour binary dilation of a (N, side, side) boolean array."""
    p = np.pad(mask, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(mask)
    s = mask.shape[1]
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out |= p[:, dr:dr + s, dc:dc + s]
    return out


def error_breakdown(prob, clean, noisy, threshold, patch=C.GNN_PATCH):
    """Split every error into a failure TYPE, so each model's failure can be named.

    false positives
        fp_thickening    predicted wall right next to a true wall (stroke too thick)
        fp_hallucination predicted wall more than 1 px away from any true wall
    false negatives
        fn_erased        a wall the damage erased and the model failed to restore
        fn_ignored       a wall that SURVIVED in the input but the model removed
    plus wall-F1 on pixels that lie on a 4x4 patch border vs inside a patch (the
    GNN's node boundaries), and per-plan F1 statistics.
    """
    s = C.SIDE
    pred = (prob > threshold).reshape(-1, s, s)
    truth = (clean > 0.5).reshape(-1, s, s)
    seen = (noisy > 0.5).reshape(-1, s, s)
    near = _dilate(truth)
    fp = pred & ~truth
    fn = ~pred & truth
    n_true = float(truth.sum())
    out = {
        "fp_thickening": float((fp & near).sum()) / n_true,
        "fp_hallucination": float((fp & ~near).sum()) / n_true,
        "fn_erased": float((fn & ~seen).sum()) / n_true,
        "fn_ignored": float((fn & seen).sum()) / n_true,
        "observed_walls_kept": float((pred & seen).sum()) / max(float(seen.sum()), 1.0),
        "pred_wall_fraction": float(pred.mean()),
        "true_wall_fraction": float(truth.mean()),
    }
    # patch-border vs patch-interior pixels
    idx = np.arange(s) % patch
    border_1d = (idx == 0) | (idx == patch - 1)
    border = border_1d[:, None] | border_1d[None, :]
    for tag, region in (("border", border), ("interior", ~border)):
        r = np.broadcast_to(region, pred.shape)
        tp = float((pred & truth & r).sum())
        fpr = float((pred & ~truth & r).sum())
        fnr = float((~pred & truth & r).sum())
        out[f"f1_patch_{tag}"] = 2 * tp / max(2 * tp + fpr + fnr, 1.0)
    # per-plan F1 and its relation to how dense the plan is
    tp = (pred & truth).sum((1, 2)).astype(float)
    fpp = (pred & ~truth).sum((1, 2)).astype(float)
    fnp = (~pred & truth).sum((1, 2)).astype(float)
    per_plan = 2 * tp / np.maximum(2 * tp + fpp + fnp, 1.0)
    density = truth.mean((1, 2))
    out["plan_f1_mean"] = float(per_plan.mean())
    out["plan_f1_p10"] = float(np.percentile(per_plan, 10))
    out["plan_f1_p90"] = float(np.percentile(per_plan, 90))
    out["plan_f1_vs_density_corr"] = float(np.corrcoef(per_plan, density)[0, 1])
    return out, per_plan, density


def damage_sweep(model, X_val, threshold, grid=None):
    """Wall-F1 of model and copy baseline at several damage levels and types."""
    grid = grid or [("none", 0, 0.0), ("drop 40%", 0, 0.4), ("drop 60%", 0, 0.6),
                    ("drop 70%", 0, 0.7), ("drop 80% (trained)", 0, 0.8), ("drop 90%", 0, 0.9),
                    ("blocks (unseen)", 10, 0.0), ("blocks+drop50 (unseen)", 10, 0.5)]
    rows = []
    for label, nb, dr in grid:
        noisy = corrupt(X_val, np.random.default_rng(C.VAL_NOISE_SEED), dropout=dr, n_blocks=nb)
        prob = batched_predict(model, noisy)
        rows.append({"damage": label, "n_blocks": nb, "dropout": dr,
                     "copy_f1": classification_metrics(noisy, X_val)["wall_f1"],
                     "model_f1": classification_metrics(prob, X_val, threshold)["wall_f1"]})
    return rows


# ---- generation -----------------------------------------------------------------
def line_continuity(binary):
    """Adjacent pixel pairs that are both wall (higher = longer connected strokes)."""
    img = binary.reshape(-1, C.SIDE, C.SIDE)
    return 0.5 * ((img[:, :, :-1] * img[:, :, 1:]).sum((1, 2))
                  + (img[:, :-1, :] * img[:, 1:, :]).sum((1, 2)))


def pairwise_diversity(batch, max_images=40, seed=0):
    pick = np.random.default_rng(seed).permutation(len(batch))[:max_images]
    b = np.asarray(batch, np.float32)[pick]
    d = [np.abs(b[i] - b[j]).mean() for i in range(len(b)) for j in range(i + 1, len(b))]
    return float(np.mean(d))


def nearest_train_distance(samples, X_train):
    """Mean Hamming distance (fraction of pixels) to the closest TRAINING plan.

    Near 0 would mean the generator is copying training plans (memorisation).
    """
    s = samples.astype(np.float32)
    t = (X_train > 0.5).astype(np.float32)
    # |a - b| for binary vectors = a.a + b.b - 2 a.b
    d = (s.sum(1)[:, None] + t.sum(1)[None, :] - 2 * s @ t.T) / s.shape[1]
    return float(d.min(axis=1).mean())


def generation_stats(name, binary, X_train):
    return {"source": name,
            "wall_fraction": float(binary.mean()),
            "line_continuity": float(line_continuity(binary).mean()),
            "diversity": pairwise_diversity(binary),
            "nn_train_distance": nearest_train_distance(binary, X_train)}


def dae_mixture_samples(dae, X_train, n, rng, sigma=0.5):
    """The DAE has no prior: perturb real training codes (mid-review method)."""
    codes = np.concatenate([dae.encode(X_train[i:i + 500]) for i in range(0, len(X_train), 500)])
    std = codes.std(axis=0, keepdims=True)
    pick = rng.integers(0, len(codes), n)
    z = codes[pick] + sigma * std * rng.standard_normal((n,) + codes.shape[1:])
    return dae.decode(z.astype(np.float32))


def vae_active_units(vae, x, tol=0.01):
    """Latent units whose posterior mean varies across inputs (Burda et al., 2016).

    A unit with Var_x(E[z|x]) < tol carries no information: that is posterior
    collapse, the classic VAE failure.
    """
    mu = np.concatenate([vae.encode(x[i:i + 400])[0] for i in range(0, len(x), 400)])
    var = mu.reshape(len(mu), -1).var(axis=0)
    return int((var > tol).sum()), int(var.size)
