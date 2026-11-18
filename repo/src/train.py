"""Train and evaluate one model on HAM10000 with a lesion-level split.

Example:
    python -m src.train --data-dir data/HAM10000 --model inceptionv3 --seeds 0 1 2
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import MODEL_ORDER, NUM_CLASSES
from .data import LesionDataset, build_transforms, load_metadata, make_splits
from .evaluate import bootstrap_ci, compute_metrics, majority_baseline
from .models import INPUT_SIZE, PRETRAINED, build_model, count_parameters


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight, self.gamma = weight, gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(labels, power=0.5):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    w = (counts.sum() / (NUM_CLASSES * counts)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def make_optimizer(model, lr, wd):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def run_one(args, seed, frame, out_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    size = INPUT_SIZE[args.model]
    pretrained = args.model in PRETRAINED
    tr, va, te = (frame[frame.split == s] for s in ("train", "val", "test"))
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(LesionDataset(tr, build_transforms(size, True)), shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(LesionDataset(va, build_transforms(size, False)), **kw)
    test_loader = DataLoader(LesionDataset(te, build_transforms(size, False)), **kw)

    model, head = build_model(args.model, pretrained=pretrained)
    model.to(device)
    w = class_weights(tr["label"].values).to(device) if args.loss != "ce" else None
    criterion = FocalLoss(w) if args.loss == "focal" else nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(out_root) / args.model / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_f1, best_state, bad = [], -1.0, None, 0
    warm = args.warmup_epochs if pretrained else 0
    opt = sched = None
    t0 = time.time()

    for epoch in range(args.epochs):
        if epoch == 0 and warm > 0:                      # stage 1: classifier head only
            for p in model.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = True
            opt, sched = make_optimizer(model, args.lr_head, args.weight_decay), None
        if epoch == warm:                                # stage 2: full network
            for p in model.parameters():
                p.requires_grad = True
            lr = args.lr_finetune if pretrained else args.lr_head
            opt = make_optimizer(model, lr, args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warm))

        model.train()
        run_loss, seen, correct = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"{args.model} s{seed} e{epoch + 1}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        if sched is not None:
            sched.step()

        yv, pv = predict(model, val_loader, device)
        vm = compute_metrics(yv, pv)
        val_loss = float(F.nll_loss(torch.log(torch.tensor(pv).clamp_min(1e-8)), torch.tensor(yv)))
        history.append({"epoch": epoch + 1, "train_loss": run_loss / seen, "train_acc": correct / seen,
                        "val_loss": val_loss, "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
                        "val_balanced_acc": vm["balanced_accuracy"]})
        print(json.dumps(history[-1]))
        if vm["macro_f1"] > best_f1:                     # model selection on validation macro-F1
            best_f1, bad = vm["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if epoch >= warm and bad >= args.patience:
                print("Early stopping.")
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    t1 = time.time()
    yt, pt = predict(model, test_loader, device)
    infer_ms = 1000 * (time.time() - t1) / len(yt)

    metrics = compute_metrics(yt, pt)
    metrics["ci95"] = bootstrap_ci(yt, pt)
    metrics["majority_baseline"] = majority_baseline(tr["label"].values, yt)
    metrics.update({"model": args.model, "seed": seed, "parameters": count_parameters(model),
                    "train_time_s": train_time, "epochs_run": len(history),
                    "inference_ms_per_image": infer_ms, "loss": args.loss,
                    "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    "n_train": len(tr), "n_val": len(va), "n_test": len(te)})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(out_dir / "test_predictions.npz", y_true=yt, probs=pt)
    print(f"[{args.model} seed {seed}] acc={metrics['accuracy']:.4f} "
          f"bal_acc={metrics['balanced_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Folder with HAM10000_metadata.csv and the .jpg images")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", required=True,
                    choices=MODEL_ORDER)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    split_file = out_root / "splits.csv"
    if split_file.exists():
        import pandas as pd
        frame = pd.read_csv(split_file)
    else:
        frame = make_splits(load_metadata(args.data_dir), seed=args.split_seed)
        frame.to_csv(split_file, index=False)
    for seed in args.seeds:
        run_one(args, seed, frame, out_root)


if __name__ == "__main__":
    main()
