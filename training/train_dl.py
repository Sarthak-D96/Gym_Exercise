"""Phase 4, rounds 3–4 — LSTM / GRU / 1D-CNN (TCN) on the raw (16, 42) sequences.

Adam 1e-3, batch 64, class-weighted loss, early stopping on validation macro-F1.
Augmentation (training only) is applied to the raw keypoints, then features are
recomputed (common.augment): left/right mirror, rotation, keypoint noise, time stretch,
and with --augment strong also camera yaw/scale, corrupted keypoints and dropped frames.

Exports a TorchScript file that takes (batch, T, 42) and returns logits, which
is what ml_models/README.md expects (T may be shorter than 16 in the website).

Example:
    python training/train_dl.py --data /data/gym/prepared --model lstm
    python training/train_dl.py --data /data/gym/prepared --model cnn
    python training/train_dl.py --data /data/gym/prepared --model lstm --eval-test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import FEATURE_SIZE, augment, extend_features, features_np, load_data, load_stage, report, save_report


# ---------------------------------------------------------------- models
class RNNClassifier(nn.Module):
    def __init__(self, n_classes, cell="lstm", hidden=128, layers=2, dropout=0.3, in_dim=FEATURE_SIZE):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):  # x: (B, T, 42)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class TCNClassifier(nn.Module):
    """1D-CNN with dilated residual blocks (a small TCN) and global average pooling."""

    def __init__(self, n_classes, channels=128, dropout=0.3, in_dim=FEATURE_SIZE):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, channels, 1)
        self.blocks = nn.ModuleList()
        for d in (1, 2, 4):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=d, dilation=d), nn.BatchNorm1d(channels), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 3, padding=d, dilation=d), nn.BatchNorm1d(channels), nn.ReLU(),
            ))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, n_classes))

    def forward(self, x):  # x: (B, T, 42)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return self.head(h.mean(-1))


def build(name, n_classes, hidden, in_dim=FEATURE_SIZE):
    if name in ("lstm", "gru"):
        return RNNClassifier(n_classes, cell=name, hidden=hidden, in_dim=in_dim)
    return TCNClassifier(n_classes, channels=hidden, in_dim=in_dim)


# ---------------------------------------------------------------- training
@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[s:s + bs]).to(device)).argmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="output folder of build_dataset.py")
    ap.add_argument("--model", choices=["lstm", "gru", "cnn"], required=True)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--hidden", type=int, default=128, help="units (RNN) or channels (CNN)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--augment", choices=["basic", "strong"], default="basic",
                    help="strong: also camera yaw/scale, bigger rotation, corrupted keypoints, dropped frames")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--workers-seed", type=int, default=42, dest="seed")
    ap.add_argument("--eval-test", action="store_true", help="also score the test set (do this once, at the end)")
    ap.add_argument("--stage", type=Path, help="two-stage: JSON stage file written by run_all.py")
    ap.add_argument("--extra-features", action="store_true", help="append the 13 extra features (42 -> 55)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    stage, classes = load_stage(args.stage, args.data)
    Xtr, ytr, KPtr, _ = load_data(args.data, "train", stage, args.extra_features)
    Xva, yva, _, _ = load_data(args.data, "val", stage, args.extra_features)
    in_dim = Xtr.shape[-1]

    def featurize(kp):
        f = features_np(kp)
        return extend_features(f) if args.extra_features else f
    print(f"train {Xtr.shape}, val {Xva.shape}, {len(classes)} classes, device {args.device}")

    counts = np.bincount(ytr, minlength=len(classes)).astype(np.float32)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), device=args.device)
    model = build(args.model, len(classes), args.hidden, in_dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    args.out.mkdir(parents=True, exist_ok=True)
    best_f1, best_state, bad, history = -1.0, None, 0, []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(ytr))
        total, t0 = 0.0, time.time()
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            xb = Xtr[b] if args.no_augment else featurize(augment(KPtr[b], rng, args.augment))
            xb = torch.from_numpy(xb).to(args.device)
            yb = torch.from_numpy(ytr[b]).to(args.device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        rep = report(yva, predict(model, Xva, args.device), classes)
        history.append({"epoch": epoch, "loss": total / len(ytr), "val_acc": rep["accuracy"],
                        "val_f1": rep["macro_f1"]})
        print(f"epoch {epoch:3d}  loss {total / len(ytr):.4f}  val acc {rep['accuracy']:.4f}  "
              f"val macro-F1 {rep['macro_f1']:.4f}  ({time.time() - t0:.0f}s)")
        if rep["macro_f1"] > best_f1:
            best_f1, bad = rep["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop: no val improvement for {args.patience} epochs")
                break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    model.eval().cpu()
    ts_file = args.out / f"{args.model}.pt"
    scripted = torch.jit.trace(model, torch.zeros(1, 16, in_dim))
    scripted.save(str(ts_file))
    torch.save({"state_dict": best_state, "classes": classes, "args": vars(args) | {"data": str(args.data),
                "out": str(args.out)}}, args.out / f"{args.model}_checkpoint.pth")
    (args.out / "classes.json").write_text(json.dumps(classes, indent=2))
    (args.out / f"{args.model}_history.json").write_text(json.dumps(history, indent=2))

    # speed on CPU, one window at a time (as in the website)
    x1 = torch.zeros(1, 16, in_dim)
    with torch.no_grad():
        for _ in range(5):
            scripted(x1)
        t0 = time.time()
        for _ in range(100):
            scripted(x1)
    ms = (time.time() - t0) / 100 * 1000
    extra = {"model": args.model, "train_seconds": train_time, "ms_per_window_cpu": ms,
             "size_mb": ts_file.stat().st_size / 1e6, "best_val_macro_f1": best_f1,
             "features": "extended" if args.extra_features else "base"}

    rep = report(yva, predict(model, Xva, "cpu"), classes)
    save_report(args.out, f"{args.model}_val", rep, classes, extra)
    print(f"[{args.model}] best val accuracy {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}  "
          f"{ms:.2f} ms/window (CPU)  {extra['size_mb']:.1f} MB")

    if args.eval_test:
        Xte, yte, _, _ = load_data(args.data, "test", stage, args.extra_features)
        rep = report(yte, predict(model, Xte, "cpu"), classes)
        save_report(args.out, f"{args.model}_test", rep, classes, extra)
        print(f"[{args.model}] TEST accuracy {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}")
    print(f"Saved {ts_file} (TorchScript, input (B, T, {in_dim}), output logits)")


if __name__ == "__main__":
    main()
