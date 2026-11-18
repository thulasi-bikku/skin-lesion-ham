"""Grad-CAM saliency maps for the convolutional models.

Example:
    python -m src.gradcam --data-dir data/HAM10000 --model inceptionv3 \
        --checkpoint outputs/inceptionv3/seed0/best.pt --out paper/figures/fig_gradcam
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .common import CLASS_CODES
from .data import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from .models import INPUT_SIZE, build_model

TARGET_LAYER = {
    "inceptionv3": lambda m: m.Mixed_7c,
    "mobilenetv2": lambda m: m.features[-1],
    "efficientnet_b0": lambda m: m.features[-1],
    "efficientnet_v2_s": lambda m: m.features[-1],
    "convnext_tiny": lambda m: m.features[-1],
    "simple_cnn": lambda m: m.features[-1],
}


def grad_cam(model, layer, x, class_idx=None):
    acts, grads = [], []
    h1 = layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    model.eval()
    logits = model(x)
    cls = int(logits.argmax(1)) if class_idx is None else class_idx
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove()
    h2.remove()
    weights = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", default="outputs/splits.csv")
    ap.add_argument("--model", required=True, choices=list(TARGET_LAYER))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per-class", type=int, default=1, help="test images shown per class")
    ap.add_argument("--out", default="paper/figures/fig_gradcam")
    args = ap.parse_args()

    frame = pd.read_csv(args.splits)
    frame = frame[frame.split == "test"]
    model, _ = build_model(args.model, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    layer = TARGET_LAYER[args.model](model)
    size = INPUT_SIZE[args.model]
    tf = build_transforms(size, train=False)
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]

    rows = []
    for k, code in enumerate(CLASS_CODES):
        rows += list(frame[frame.label == k].sample(args.per_class, random_state=0).itertuples())
    fig, axes = plt.subplots(2, len(rows), figsize=(1.7 * len(rows), 3.6))
    for j, r in enumerate(rows):
        x = tf(Image.open(r.path).convert("RGB"))[None].requires_grad_(True)
        cam, pred = grad_cam(model, layer, x)
        img = np.clip((x[0].detach().numpy() * std + mean).transpose(1, 2, 0), 0, 1)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"true: {CLASS_CODES[r.label]}\npred: {CLASS_CODES[pred]}", fontsize=7)
        axes[1, j].imshow(img)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        for a in axes[:, j]:
            a.axis("off")
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
