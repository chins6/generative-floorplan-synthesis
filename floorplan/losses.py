"""Weighted binary cross-entropy and the evaluation metrics shared by all models."""
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C

EPS = 1e-7


def bce_loss(prob, target, wall_weight=C.WALL_WEIGHT):
    """Class-weighted mean BCE, normalised by the mean weight in the batch.

        L = - sum_i w_i [y_i log p_i + (1 - y_i) log(1 - p_i)] / sum_i w_i
        w_i = wall_weight if y_i = 1 else 1
    """
    prob = np.clip(prob, EPS, 1.0 - EPS)
    weight = 1.0 + (wall_weight - 1.0) * target
    ce = target * np.log(prob) + (1 - target) * np.log(1 - prob)
    return float(-(weight * ce).mean() / weight.mean())


def bce_logit_grad(prob, target, wall_weight=C.WALL_WEIGHT):
    """dL/dz for the loss above, with the sigmoid fused in: w (p - y) / sum(w)."""
    weight = 1.0 + (wall_weight - 1.0) * target
    return (prob - target) * weight / (target.size * float(weight.mean()))


def confusion(prob, target, threshold=0.5):
    pred = prob > threshold
    truth = target > 0.5
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    tn = float((~pred & ~truth).sum())
    return tp, fp, fn, tn


def classification_metrics(prob, target, threshold=0.5):
    """Pixel accuracy and wall-class precision / recall / F1 / IoU."""
    tp, fp, fn, tn = confusion(prob, target, threshold)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1.0),
        "wall_precision": precision,
        "wall_recall": recall,
        "wall_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "wall_iou": tp / max(tp + fp + fn, 1.0),
    }


def pixel_auc(prob, target):
    return float(roc_auc_score(target.reshape(-1) > 0.5, prob.reshape(-1)))


def best_threshold(prob, target, grid=np.arange(0.05, 0.951, 0.05)):
    """Threshold maximising wall-F1. Call it on VALIDATION data only."""
    scores = [(float(t), classification_metrics(prob, target, t)["wall_f1"]) for t in grid]
    return max(scores, key=lambda s: s[1])
