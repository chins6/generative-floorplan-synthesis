"""Mini-batch training with Adam and best-checkpoint saving.

After every epoch the model is scored on the fixed damaged VALIDATION set. The
checkpoint with the lowest validation reconstruction loss (weighted BCE, the
quantity the models are trained to minimise, and independent of any decision
threshold) is written to disk. The test set is never touched here.
For the VAE, only epochs after the KL warm-up are eligible, because before that
the model is optimising a different (smaller-beta) objective.
"""
import time

import numpy as np

from . import config as C
from .checkpoint import save_checkpoint
from .data import corrupt
from .layers import Adam
from .losses import bce_loss, classification_metrics


def fit(model, X_train, X_val, val_noisy, epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
        lr=C.LEARNING_RATE, seed=C.TRAIN_SEED, ckpt_path=None, kl_warmup=0,
        print_every=5, log=print):
    """Train `model` in place. Returns the per-epoch history as a dict of lists."""
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    opt = Adam(model.parameters(), lr)
    history = {"epoch": [], "train_loss": [], "train_recon": [], "train_kl": [],
               "val_loss": [], "val_wall_f1": [], "is_best": [], "seconds": []}
    best_val, best_epoch = np.inf, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Re-corrupt the whole training set every epoch: the model never sees the
        # same damaged image twice, so it cannot memorise the damage.
        X_noisy = corrupt(X_train, rng)
        order = rng.permutation(n)
        # KL warm-up (VAE only): ramp beta linearly from 0 to its full value
        kl_scale = min(1.0, epoch / kl_warmup) if kl_warmup else 1.0
        sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            if len(idx) < 2:
                continue
            logs, grads = model.loss_and_grads(X_noisy[idx], X_train[idx], rng, kl_scale)
            opt.step(grads)
            for key in sums:
                sums[key] += logs.get(key, 0.0) * len(idx)

        val_prob = model.predict(val_noisy)
        val_loss = bce_loss(val_prob, X_val, model.wall_weight)
        val_f1 = classification_metrics(val_prob, X_val)["wall_f1"]
        # During the VAE's KL warm-up the objective is still changing (beta < full
        # value), so those epochs are not eligible as the "best" checkpoint.
        # (A run shorter than the warm-up, e.g. the live demo, still saves its last epochs.)
        is_best = val_loss < best_val and epoch >= min(kl_warmup, epochs)
        if is_best:
            best_val, best_epoch = val_loss, epoch
            if ckpt_path is not None:
                save_checkpoint(model, ckpt_path, epoch=epoch, val_loss=float(val_loss),
                                epochs_planned=epochs, lr=lr, batch_size=batch_size)

        for key, value in (("epoch", epoch), ("train_loss", sums["loss"] / n),
                           ("train_recon", sums["recon"] / n), ("train_kl", sums["kl"] / n),
                           ("val_loss", val_loss), ("val_wall_f1", val_f1),
                           ("is_best", is_best), ("seconds", time.time() - start)):
            history[key].append(value)

        if log and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            kl_txt = f"  kl={sums['kl'] / n:.2f}" if model.name == "VAE" else ""
            log(f"  [{model.name}] epoch {epoch:3d}/{epochs}  train={sums['loss'] / n:.4f}{kl_txt}  "
                f"val_loss={val_loss:.4f}  val_F1@0.5={val_f1:.4f}"
                f"{'  *best' if is_best else ''}  [{time.time() - start:.0f}s]")

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val)
    if log:
        log(f"  [{model.name}] best epoch {best_epoch} (val_loss {best_val:.4f})"
            + (f" -> saved {ckpt_path.name}" if ckpt_path is not None else ""))
    return history
