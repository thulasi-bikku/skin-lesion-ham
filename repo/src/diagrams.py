"""Schematic figures (study workflow and InceptionV3 layout); no experimental data involved."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .report import TEXT_WIDTH_IN, save_fig


def _box(ax, x, y, w, h, text, fc="#eaf2f8", ec="#1b4f72", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9))


def workflow(figdir):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.8); ax.axis("off"); ax.grid(False)
    top = [("HAM10000\n10,015 images,\n7 classes", 0.1),
           ("Lesion-level\nstratified split\n(train / val / test)", 3.1),
           ("Resizing,\nnormalisation,\naugmentation", 6.1),
           ("Backbone\n(transfer\nlearning)", 9.1)]
    for t, x in top:
        _box(ax, x, 3.6, 2.4, 1.7, t, fs=6)
    for i in range(3):
        _arrow(ax, top[i][1] + 2.4, 4.45, top[i + 1][1], 4.45)
    bottom = [("Class-weighted\nloss, AdamW,\nearly stopping", 9.1),
              ("Model selection:\nvalidation\nmacro-F1", 6.1),
              ("Test evaluation:\nmetrics, CI,\nROC curves", 3.1),
              ("Tables and\nfigures", 0.1)]
    for t, x in bottom:
        _box(ax, x, 0.5, 2.4, 1.7, t, fc="#fef5e7", ec="#b9770e", fs=6)
    _arrow(ax, 10.3, 3.6, 10.3, 2.2)
    for i in range(3):
        _arrow(ax, bottom[i][1], 1.35, bottom[i + 1][1] + 2.4, 1.35)
    save_fig(fig, Path(figdir) / "fig_workflow")


def inception(figdir):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off"); ax.grid(False)
    blocks = [("Input\n299 x 299 x 3", "#f2f3f4"), ("Stem\n(conv + pool)", "#eaf2f8"),
              ("3 x\nInception-A", "#d6eaf8"), ("Reduction\nblock A", "#fdebd0"),
              ("4 x\nInception-B", "#d6eaf8"), ("Reduction\nblock B", "#fdebd0"),
              ("2 x\nInception-C", "#d6eaf8"), ("Global\naverage pool", "#e8f6f3"),
              ("Dropout", "#fadbd8"), ("Linear\n(7 classes)", "#fadbd8")]
    w, h, step = 1.9, 1.3, 2.45
    pos = []
    for i, (t, c) in enumerate(blocks):
        row, col = divmod(i, 5)
        x, y = 0.1 + col * step, 3.0 - row * 2.3
        pos.append((x, y))
        _box(ax, x, y, w, h, t, fc=c, fs=6.5)
        if col:
            _arrow(ax, x - (step - w), y + h / 2, x, y + h / 2)
    _arrow(ax, pos[4][0] + w / 2, pos[4][1], pos[5][0] + w / 2, pos[5][1] + h)
    save_fig(fig, Path(figdir) / "fig_inception")


if __name__ == "__main__":
    out = Path("paper/figures")
    workflow(out)
    inception(out)
    print("Diagrams written to", out)
