"""Print the Phase 4 results table (for JOURNAL.md) from the files in runs/.

    python training/summarize.py --runs runs          # validation scores
    python training/summarize.py --runs runs --test   # test scores (after --eval-test)
"""

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM", "gru": "GRU", "cnn": "1D-CNN / TCN"}

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, default=Path("runs"))
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
split = "test" if args.test else "val"

print(f"Scores on the {split} set\n")
print("| Model | Accuracy | Macro F1 | Time per window | Size |")
print("|---|---|---|---|---|")
for key, name in NAMES.items():
    f = args.runs / f"{key}_{split}.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    ms = r.get("ms_per_window", r.get("ms_per_window_cpu"))
    print(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {ms:.2f} ms | {r['size_mb']:.1f} MB |")

    cm, classes = np.array(r["confusion_matrix"]), r["classes"]
    np.fill_diagonal(cm, 0)
    top = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:3], cm.shape))[0]
    pairs = [f"{classes[i]}→{classes[j]} ({cm[i, j]})" for i, j in top if cm[i, j] > 0]
    if pairs:
        print(f"|   most confused: {', '.join(pairs)} | | | | |")
