"""Build all manuscript figures (EPS/PDF/PNG, 600 dpi) and LaTeX tables from saved results.

Example:
    python -m src.report --out-dir outputs --paper-dir paper --data-dir data/HAM10000
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .common import CLASS_CODES, CLASS_NAMES, MODEL_LABELS, MODEL_ORDER, NUM_CLASSES

TEXT_WIDTH_IN = 12.2 / 2.54          # JAMI typing-area width
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})
PALETTE = ["#1b4f72", "#b9770e", "#1e8449", "#922b21", "#6c3483", "#117a8b", "#5d6d7e"]


def save_fig(fig, base):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("eps", "pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def load_runs(out_dir):
    runs = {}
    for m in MODEL_ORDER:
        seeds = sorted((Path(out_dir) / m).glob("seed*"))
        res = []
        for s in seeds:
            if (s / "metrics.json").exists():
                res.append({"metrics": json.loads((s / "metrics.json").read_text()),
                            "history": json.loads((s / "history.json").read_text()),
                            "preds": np.load(s / "test_predictions.npz")})
        if res:
            runs[m] = res
    return runs


def ms(values, digits=3):
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return f"${v[0]:.{digits}f}$"
    return f"${v.mean():.{digits}f} \\pm {v.std(ddof=1) if len(v) > 1 else 0.0:.{digits}f}$"


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


# --------------------------------------------------------------------------- tables
def table_dataset(frame, gen):
    rows = []
    for k, code in enumerate(CLASS_CODES):
        sub = frame[frame.label == k]
        rows.append(f"{code} & {CLASS_NAMES[code]} & {len(sub)} & "
                    f"{(sub.split == 'train').sum()} & {(sub.split == 'val').sum()} & {(sub.split == 'test').sum()} \\\\")
    rows.append("\\hline")
    rows.append(f"Total & & {len(frame)} & {(frame.split == 'train').sum()} & "
                f"{(frame.split == 'val').sum()} & {(frame.split == 'test').sum()} \\\\")
    write(gen / "table_dataset.tex",
          "\\begin{tabular}{llrrrr}\n\\hline\nCode & Diagnostic category & All & Train & Val. & Test \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def table_overall(runs, gen):
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Bal. acc."), ("macro_f1", "Macro-F1"),
            ("mcc", "MCC"), ("auroc_macro", "AUROC")]
    rows = []
    for m, res in runs.items():
        cells = [ms([r["metrics"][k] for r in res]) for k, _ in keys]
        rows.append(f"{MODEL_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    base = next(iter(runs.values()))[0]["metrics"]["majority_baseline"]
    rows.append("\\hline")
    rows.append("Majority-class baseline & " + " & ".join(
        f"${base[k]:.3f}$" if k != "auroc_macro" else "$0.500$" for k, _ in keys) + " \\\\")
    write(gen / "table_overall.tex",
          "\\begin{tabular}{lccccc}\n\\hline\nModel & " + " & ".join(n for _, n in keys) + " \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def table_perclass(runs, primary, gen):
    res = runs[primary]
    pc = {k: np.mean([r["metrics"]["per_class"][k] for r in res], axis=0)
          for k in ("precision", "recall", "specificity", "f1", "auroc")}
    support = res[0]["metrics"]["per_class"]["support"]
    rows = [f"{CLASS_CODES[i]} & {pc['precision'][i]:.3f} & {pc['recall'][i]:.3f} & {pc['specificity'][i]:.3f} & "
            f"{pc['f1'][i]:.3f} & {pc['auroc'][i]:.3f} & {support[i]} \\\\" for i in range(NUM_CLASSES)]
    write(gen / "table_perclass.tex",
          "\\begin{tabular}{lcccccr}\n\\hline\nClass & Precision & Sensitivity & Specificity & F1 & AUROC & Support \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def table_ci(runs, gen):
    rows = []
    for m, res in runs.items():
        ci = res[0]["metrics"]["ci95"]
        mt = res[0]["metrics"]
        cells = [f"${mt[k]:.3f}$ [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")]
        rows.append(f"{MODEL_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    write(gen / "table_ci.tex",
          "\\begin{tabular}{lcccc}\n\\hline\nModel & Accuracy & Bal. acc. & Macro-F1 & MCC \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def table_cost(runs, gen):
    rows = []
    for m, res in runs.items():
        mt = [r["metrics"] for r in res]
        per_epoch = np.mean([x["train_time_s"] / x["epochs_run"] for x in mt]) / 60
        rows.append(f"{MODEL_LABELS[m]} & {mt[0]['parameters'] / 1e6:.2f} & {per_epoch:.2f} & "
                    f"{np.mean([x['inference_ms_per_image'] for x in mt]):.2f} \\\\")
    write(gen / "table_cost.tex",
          "\\begin{tabular}{lccc}\n\\hline\nModel & Parameters (M) & Training (min/epoch) & Inference (ms/image) \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


# --------------------------------------------------------------------------- figures
def fig_class_distribution(frame, figdir):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.6))
    counts = frame.groupby(["label", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    bottom = np.zeros(NUM_CLASSES)
    for s, c in zip(["train", "val", "test"], PALETTE):
        v = counts[s].reindex(range(NUM_CLASSES)).values
        ax.bar(CLASS_CODES, v, bottom=bottom, label=s, color=c)
        bottom += v
    ax.set_ylabel("Number of images")
    ax.set_yscale("log")
    ax.legend()
    save_fig(fig, figdir / "fig_class_distribution")


def fig_curves(runs, primary, figdir):
    h = runs[primary][0]["history"]
    ep = [x["epoch"] for x in h]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.5))
    axes[0].plot(ep, [x["train_loss"] for x in h], color=PALETTE[0], label="Training")
    axes[0].plot(ep, [x["val_loss"] for x in h], color=PALETTE[1], label="Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(ep, [x["val_acc"] for x in h], color=PALETTE[0], label="Accuracy")
    axes[1].plot(ep, [x["val_balanced_acc"] for x in h], color=PALETTE[2], label="Balanced accuracy")
    axes[1].plot(ep, [x["val_macro_f1"] for x in h], color=PALETTE[1], label="Macro-F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation score"); axes[1].legend()
    fig.tight_layout()
    save_fig(fig, figdir / "fig_curves")


def fig_confusion(runs, figdir, primary="inceptionv3", max_models=6):
    ranked = sorted(runs, key=lambda m: -np.mean([r["metrics"]["macro_f1"] for r in runs[m]]))
    names = [primary] + [m for m in ranked if m != primary][:max_models - 1]
    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(TEXT_WIDTH_IN, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, names):
        cm = np.array(runs[m][0]["metrics"]["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.grid(False)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=5,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_CODES, rotation=45, fontsize=5)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_CODES, fontsize=5)
        ax.set_title(MODEL_LABELS[m], fontsize=7)
        ax.set_xlabel("Predicted", fontsize=6); ax.set_ylabel("True", fontsize=6)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, figdir / "fig_confusion")


def fig_roc(runs, primary, figdir):
    d = runs[primary][0]["preds"]
    y, p = d["y_true"], d["probs"]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.7, 3.2))
    for k, c in enumerate(PALETTE):
        if (y == k).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve((y == k).astype(int), p[:, k])
        auc = runs[primary][0]["metrics"]["per_class"]["auroc"][k]
        ax.plot(fpr, tpr, color=c, label=f"{CLASS_CODES[k]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
    save_fig(fig, figdir / "fig_roc")


def fig_comparison(runs, figdir):
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1")]
    names = list(runs)
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.8))
    for i, (k, lab) in enumerate(keys):
        vals = [np.mean([r["metrics"][k] for r in runs[m]]) for m in names]
        err = [np.std([r["metrics"][k] for r in runs[m]], ddof=1) if len(runs[m]) > 1 else 0 for m in names]
        ax.bar(x + (i - 1) * w, vals, w, yerr=err if any(err) else None, capsize=2, label=lab, color=PALETTE[i])
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in names], rotation=20)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    save_fig(fig, figdir / "fig_comparison")


def fig_samples(frame, figdir, n=5):
    from PIL import Image
    fig, axes = plt.subplots(NUM_CLASSES, n, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * NUM_CLASSES / n * 0.9))
    for k, code in enumerate(CLASS_CODES):
        sub = frame[frame.label == k].sample(n, random_state=1)
        for j, path in enumerate(sub["path"]):
            axes[k, j].imshow(Image.open(path).convert("RGB"))
            axes[k, j].axis("off")
        axes[k, 0].set_title(code, fontsize=8, loc="left")
    fig.tight_layout()
    save_fig(fig, figdir / "fig_samples")


# --------------------------------------------------------------------------- numbers and text
def choose_primary(runs):
    """Model with the highest mean (over seeds) of the best validation macro-F1; no test data involved."""
    def val_f1(m):
        return float(np.mean([max(h["val_macro_f1"] for h in r["history"]) for r in runs[m]]))
    return max(runs, key=val_f1)


PREFIX = {"simple_cnn": "Cnn", "cnn_lstm": "Lstm", "mobilenetv2": "Mob", "inceptionv3": "Inc",
          "efficientnet_b0": "Eff", "efficientnet_v2_s": "Effv", "convnext_tiny": "Cnx", "swin_t": "Swin",
          "vit_small": "Vit", "dinov2_small": "Dino"}


def write_numbers_and_text(runs, primary, frame, gen):
    """Write LaTeX macros and short results paragraphs that are computed directly from the saved runs."""
    def mean(m, k):
        return float(np.mean([r["metrics"][k] for r in runs[m]]))

    pm = runs[primary][0]["metrics"]
    n_seeds = len(runs[primary])
    base = pm["majority_baseline"]
    macros = {
        "PrimaryName": MODEL_LABELS[primary], "PrimaryAcc": f"{mean(primary, 'accuracy'):.3f}",
        "PrimaryBalAcc": f"{mean(primary, 'balanced_accuracy'):.3f}",
        "PrimaryMacroF": f"{mean(primary, 'macro_f1'):.3f}", "PrimaryMCC": f"{mean(primary, 'mcc'):.3f}",
        "PrimaryAUC": f"{mean(primary, 'auroc_macro'):.3f}", "BaselineAcc": f"{base['accuracy']:.3f}",
        "NTest": str(int((frame.split == 'test').sum())), "NTrain": str(int((frame.split == 'train').sum())),
        "NVal": str(int((frame.split == 'val').sum())), "NSeeds": str(n_seeds),
    }
    mel_idx = CLASS_CODES.index("mel")
    for m in runs:
        pf = PREFIX[m]
        macros[pf + "Acc"] = f"{mean(m, 'accuracy'):.3f}"
        macros[pf + "BalAcc"] = f"{mean(m, 'balanced_accuracy'):.3f}"
        macros[pf + "MacroF"] = f"{mean(m, 'macro_f1'):.3f}"
        macros[pf + "MCC"] = f"{mean(m, 'mcc'):.3f}"
        macros[pf + "AUC"] = f"{mean(m, 'auroc_macro'):.3f}"
        macros[pf + "MelSens"] = f"{np.mean([r['metrics']['per_class']['recall'][mel_idx] for r in runs[m]]):.3f}"
        macros[pf + "MelSpec"] = f"{np.mean([r['metrics']['per_class']['specificity'][mel_idx] for r in runs[m]]):.3f}"
    macros["OverallNote"] = ("single training run per model" if n_seeds == 1
                             else "mean $\\pm$ standard deviation over seeds")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    write(gen / "numbers.tex", "\n".join(lines) + "\n")

    order = sorted(runs, key=lambda m: -mean(m, "macro_f1"))
    rank = order.index(primary) + 1
    rank_txt = ", ".join(f"{MODEL_LABELS[m]} ({mean(m, 'macro_f1'):.3f})" for m in order)
    verdict = ("exceeds" if mean(primary, "accuracy") > base["accuracy"] else "does not exceed")
    overall = (
        f"On the held-out test partition of {macros['NTest']} images, {MODEL_LABELS[primary]} reached an accuracy of "
        f"{macros['PrimaryAcc']}, a balanced accuracy of {macros['PrimaryBalAcc']}, a macro-averaged F1-score of "
        f"{macros['PrimaryMacroF']} and a Matthews correlation coefficient of {macros['PrimaryMCC']} "
        f"({'single training run' if n_seeds == 1 else 'mean over ' + str(n_seeds) + ' training seeds'}). A classifier that always predicts the most frequent training class "
        f"reaches an accuracy of {macros['BaselineAcc']} on the same images, so the accuracy of {MODEL_LABELS[primary]} "
        f"{verdict} this reference value. Ranked by macro-F1, the models order as follows: {rank_txt}. "
        f"{MODEL_LABELS[primary]} ranks {rank} of {len(order)}.")
    write(gen / "results_overall.tex", overall + "\n")

    res = runs[primary]
    rec = np.mean([r["metrics"]["per_class"]["recall"] for r in res], axis=0)
    spec = np.mean([r["metrics"]["per_class"]["specificity"] for r in res], axis=0)
    hi, lo = int(rec.argmax()), int(rec.argmin())
    mel = CLASS_CODES.index("mel")
    perclass = (
        f"For {MODEL_LABELS[primary]}, sensitivity was highest for {CLASS_CODES[hi]} ({rec[hi]:.3f}) and lowest for "
        f"{CLASS_CODES[lo]} ({rec[lo]:.3f}). For melanoma, the sensitivity was {rec[mel]:.3f} and the "
        f"specificity was {spec[mel]:.3f}.")
    write(gen / "results_perclass.tex", perclass + "\n")

    ci = pm["ci95"]
    ci_txt = (
        f"For {MODEL_LABELS[primary]}, the 95\\% bootstrap interval was "
        f"[{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}] for accuracy, "
        f"[{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}] for balanced accuracy and "
        f"[{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}] for macro-F1.")
    write(gen / "results_ci.tex", ci_txt + "\n")

    fastest = min(runs, key=lambda m: np.mean([r["metrics"]["inference_ms_per_image"] for r in runs[m]]))
    cost = (
        f"{MODEL_LABELS[primary]} has {pm['parameters'] / 1e6:.2f} million parameters and needed "
        f"{np.mean([r['metrics']['inference_ms_per_image'] for r in res]):.2f} ms per image at inference on the "
        f"evaluation hardware ({pm.get('device', 'unspecified device')}). The fastest model was {MODEL_LABELS[fastest]} with "
        f"{np.mean([r['metrics']['inference_ms_per_image'] for r in runs[fastest]]):.2f} ms per image.")
    write(gen / "results_cost.tex", cost + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--paper-dir", default="paper")
    ap.add_argument("--primary", default="auto",
                    help="Model for the detailed analysis; auto = best mean validation macro-F1")
    ap.add_argument("--data-dir", default=None, help="Optional: HAM10000 folder, used for the sample-image figure")
    args = ap.parse_args()

    gen, figdir = Path(args.paper_dir) / "generated", Path(args.paper_dir) / "figures"
    frame = pd.read_csv(Path(args.out_dir) / "splits.csv")
    runs = load_runs(args.out_dir)
    if args.primary == "auto":
        args.primary = choose_primary(runs)
        print("Primary model (best validation macro-F1):", args.primary)
    if args.primary not in runs:
        raise SystemExit(f"No results found for primary model '{args.primary}' in {args.out_dir}")

    table_dataset(frame, gen); table_overall(runs, gen); table_perclass(runs, args.primary, gen)
    table_ci(runs, gen); table_cost(runs, gen)
    fig_class_distribution(frame, figdir); fig_curves(runs, args.primary, figdir)
    fig_confusion(runs, figdir, args.primary); fig_roc(runs, args.primary, figdir); fig_comparison(runs, figdir)
    if args.data_dir and "path" in frame:
        fig_samples(frame, figdir)
    write_numbers_and_text(runs, args.primary, frame, gen)
    write(gen / "primary.txt", args.primary + "\n")
    summary = {m: {k: float(np.mean([r["metrics"][k] for r in res]))
                   for k in ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "auroc_macro")}
               for m, res in runs.items()}
    write(gen / "summary.json", json.dumps(summary, indent=2))
    print("Report written to", args.paper_dir)


if __name__ == "__main__":
    main()
