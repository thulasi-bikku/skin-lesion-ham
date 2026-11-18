"""CPU-only tests: metrics, bootstrap, splitting and report generation on synthetic data."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import CLASS_CODES, MODEL_ORDER, NUM_CLASSES
from src.data import make_splits
from src.evaluate import bootstrap_ci, compute_metrics, majority_baseline


def synthetic_meta(n_lesions=600, seed=0):
    rng = np.random.default_rng(seed)
    p = np.array([0.03, 0.05, 0.11, 0.01, 0.11, 0.67, 0.02])
    lesion_label = rng.choice(NUM_CLASSES, n_lesions, p=p)
    rows = []
    for i, lab in enumerate(lesion_label):
        for j in range(rng.integers(1, 3)):
            rows.append({"lesion_id": f"L{i}", "image_id": f"I{i}_{j}", "label": int(lab)})
    return pd.DataFrame(rows)


def test_split_is_lesion_grouped_and_stratified():
    sp = make_splits(synthetic_meta())
    groups = sp.groupby("lesion_id")["split"].nunique()
    assert (groups == 1).all()
    assert set(sp.split) == {"train", "val", "test"}
    assert 0.15 < (sp.split == "test").mean() < 0.25


def test_metrics_perfect_and_majority():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 5, 5, 5])
    probs = np.eye(NUM_CLASSES)[y]
    m = compute_metrics(y, probs)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    base = majority_baseline(np.array([5] * 8 + [0, 1]), y)
    assert abs(base["accuracy"] - 0.4) < 1e-9
    ci = bootstrap_ci(y, probs, n_boot=50)
    assert ci["accuracy"][0] <= 1.0


def test_report_generation(tmp_path):
    rng = np.random.default_rng(1)
    out = tmp_path / "outputs"
    frame = make_splits(synthetic_meta(1500))
    out.mkdir()
    frame.to_csv(out / "splits.csv", index=False)
    test = frame[frame.split == "test"]
    y = test["label"].values
    for m in MODEL_ORDER[:3] + ["inceptionv3"]:
        for seed in (0, 1):
            d = out / m / f"seed{seed}"
            d.mkdir(parents=True)
            logits = rng.normal(size=(len(y), NUM_CLASSES)) + 2.5 * np.eye(NUM_CLASSES)[y]
            probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
            met = compute_metrics(y, probs)
            met["ci95"] = bootstrap_ci(y, probs, n_boot=30)
            met["majority_baseline"] = majority_baseline(frame[frame.split == "train"]["label"].values, y)
            met.update({"parameters": 1_000_000, "train_time_s": 600.0, "epochs_run": 10,
                        "inference_ms_per_image": 5.0})
            (d / "metrics.json").write_text(json.dumps(met))
            hist = [{"epoch": e + 1, "train_loss": 1 / (e + 1), "train_acc": 0.5, "val_loss": 1.1 / (e + 1),
                     "val_acc": 0.6, "val_macro_f1": 0.5, "val_balanced_acc": 0.55} for e in range(10)]
            (d / "history.json").write_text(json.dumps(hist))
            np.savez(d / "test_predictions.npz", y_true=y, probs=probs)
    paper = tmp_path / "paper"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "src.report", "--out-dir", str(out), "--paper-dir", str(paper)],
                   check=True, cwd=root)
    for f in ("table_overall.tex", "table_perclass.tex", "table_dataset.tex", "table_ci.tex", "table_cost.tex"):
        assert (paper / "generated" / f).stat().st_size > 50
    for f in ("fig_confusion", "fig_curves", "fig_roc", "fig_comparison", "fig_class_distribution"):
        for ext in ("eps", "pdf", "png"):
            assert (paper / "figures" / f"{f}.{ext}").exists()
