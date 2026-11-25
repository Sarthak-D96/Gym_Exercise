"""Phase 4, rounds 1–2 — Random Forest and XGBoost on window statistics.

Each 16-frame window becomes 210 numbers (mean, std, min, max, range of the
42 features). Tuning uses the validation set only; add --eval-test once, at
the end, to score the test set.

Example:
    python training/train_ml.py --data /data/gym/prepared --model rf
    python training/train_ml.py --data /data/gym/prepared --model xgb --device cuda
    python training/train_ml.py --data /data/gym/prepared --model rf --eval-test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from common import load_data, load_stage, report, save_report, window_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="output folder of build_dataset.py")
    ap.add_argument("--model", choices=["rf", "xgb"], required=True)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="xgb only: 'cuda' to train on GPU")
    ap.add_argument("--eval-test", action="store_true", help="also score the test set (do this once, at the end)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", type=Path, help="two-stage: JSON stage file written by run_all.py")
    ap.add_argument("--extra-features", action="store_true", help="append the 13 extra features (42 -> 55)")
    args = ap.parse_args()

    stage, classes = load_stage(args.stage, args.data)
    Xtr, ytr, _, _ = load_data(args.data, "train", stage, args.extra_features)
    Xva, yva, _, _ = load_data(args.data, "val", stage, args.extra_features)
    Ftr, Fva = window_stats(Xtr), window_stats(Xva)
    weights = compute_sample_weight("balanced", ytr)
    print(f"train {Ftr.shape}, val {Fva.shape}, {len(classes)} classes")

    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                     class_weight="balanced", n_jobs=-1, random_state=args.seed)
        t0 = time.time()
        clf.fit(Ftr, ytr)
    else:
        from xgboost import XGBClassifier

        clf = XGBClassifier(n_estimators=args.n_estimators * 4, max_depth=args.max_depth or 6,
                            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", device=args.device, eval_metric="mlogloss",
                            early_stopping_rounds=50, random_state=args.seed, n_jobs=-1)
        t0 = time.time()
        clf.fit(Ftr, ytr, sample_weight=weights, eval_set=[(Fva, yva)], verbose=100)
    train_time = time.time() - t0

    name = args.model
    args.out.mkdir(parents=True, exist_ok=True)
    model_file = args.out / f"{name}.joblib"
    n_feat = Xtr.shape[-1]
    joblib.dump({"model": clf, "classes": classes, "input": f"window_stats(16x{n_feat})->{n_feat * 5}",
                 "features": "extended" if args.extra_features else "base"}, model_file)

    # speed: one window at a time, like the website
    one = Fva[:1] if len(Fva) else Ftr[:1]
    clf.predict_proba(one)
    t0 = time.time()
    for _ in range(50):
        clf.predict_proba(one)
    ms = (time.time() - t0) / 50 * 1000
    extra = {"model": name, "train_seconds": train_time, "ms_per_window": ms,
             "size_mb": model_file.stat().st_size / 1e6,
             "features": "extended" if args.extra_features else "base"}

    rep = report(yva, clf.predict(Fva), classes)
    save_report(args.out, f"{name}_val", rep, classes, extra)
    print(f"[{name}] val  accuracy {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}  "
          f"{ms:.2f} ms/window  {extra['size_mb']:.1f} MB")

    if args.eval_test:
        Xte, yte, _, _ = load_data(args.data, "test", stage, args.extra_features)
        rep = report(yte, clf.predict(window_stats(Xte)), classes)
        save_report(args.out, f"{name}_test", rep, classes, extra)
        print(f"[{name}] TEST accuracy {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}")
    print(f"Saved {model_file}")


if __name__ == "__main__":
    main()
