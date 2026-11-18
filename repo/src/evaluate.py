"""Metrics, bootstrap confidence intervals and trivial baselines (NumPy / scikit-learn only)."""
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)

from .common import NUM_CLASSES


def compute_metrics(y_true, probs, with_auc=True):
    """Overall and per-class metrics from true labels and class probabilities."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    labels = list(range(NUM_CLASSES))
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    fp = cm.sum(0) - np.diag(cm)
    spec = np.divide(tn, tn + fp, out=np.zeros(len(labels)), where=(tn + fp) > 0)
    if not with_auc:
        auc_macro, auc_each = float("nan"), [float("nan")] * len(labels)
    else:
        try:
            auc_macro = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels))
            auc_each = [float(roc_auc_score((y_true == k).astype(int), probs[:, k])) for k in labels]
        except ValueError:
            auc_macro, auc_each = float("nan"), [float("nan")] * len(labels)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "auroc_macro": auc_macro,
        "per_class": {
            "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(),
            "support": s.astype(int).tolist(), "specificity": spec.tolist(), "auroc": auc_each,
        },
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true, probs, keys=("accuracy", "balanced_accuracy", "macro_f1", "mcc"),
                 n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap confidence intervals over test images."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n = len(y_true)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = compute_metrics(y_true[idx], probs[idx], with_auc=False)
        for k in keys:
            draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in draws.items()}


def majority_baseline(y_train, y_test):
    """Predict the most frequent training class for every test image."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    top = int(np.bincount(y_train, minlength=NUM_CLASSES).argmax())
    probs = np.zeros((len(y_test), NUM_CLASSES))
    probs[:, top] = 1.0
    return compute_metrics(y_test, probs)
