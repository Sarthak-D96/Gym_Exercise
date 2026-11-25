"""Phase 4, final step — pick the best model, score the test set once, export it for the website.

1. Ranks every trained model by its VALIDATION score (--select-by, then accuracy,
   then speed). The test set plays no part in the choice.
2. Scores every model on the test set once, for the report (runs/<model>_test.json).
3. Writes <out>/gymrec_model_<id>/ and <out>/gymrec_model_<id>.zip containing:
       classifier.pt | classifier.joblib   the winning model
       classes.json                        label order
       yolo11n-pose.pt                     pose weights used for training (if found)
       metrics_test.json                   scores, stored in the website's MLModel row
       model_card.json                     everything the website needs to install it
       results.md                          comparison table for JOURNAL.md

Normally run by run_all.py. Stand-alone:
    python training/select_best.py --data gym/prepared --runs gym/runs --out gym --run-config gym/run_config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from common import load_classes, load_split, report, save_report, window_stats

NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM", "gru": "GRU", "cnn": "1D-CNN / TCN"}
SUFFIX = {"rf": ".joblib", "xgb": ".joblib", "lstm": ".pt", "gru": ".pt", "cnn": ".pt"}


def predict(model_file: Path, X: np.ndarray) -> np.ndarray:
    if model_file.suffix == ".pt":
        import torch

        net = torch.jit.load(str(model_file), map_location="cpu").eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 512):
                out.append(net(torch.from_numpy(X[s:s + 512])).argmax(-1).numpy())
        return np.concatenate(out) if out else np.zeros(0, int)
    import joblib

    bundle = joblib.load(model_file)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    return model.predict(window_stats(X)).astype(int) if len(X) else np.zeros(0, int)


def ms(r: dict) -> float:
    return r.get("ms_per_window", r.get("ms_per_window_cpu", 0.0))


def versions() -> dict:
    out = {}
    for pkg in ("torch", "scikit-learn", "xgboost", "ultralytics", "numpy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def table(rows: list[dict], best: str, split: str) -> list[str]:
    lines = [f"Scores on the {split} set ({'chosen on validation' if split == 'test' else 'used to choose'})", "",
             "| Model | Accuracy | Macro F1 | Time per window | Size |", "|---|---|---|---|---|"]
    for r in rows:
        s = r.get(split)
        if not s:
            continue
        star = " **(chosen)**" if r["model"] == best else ""
        lines.append(f"| {NAMES[r['model']]}{star} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | "
                     f"{r['ms']:.2f} ms | {r['size_mb']:.1f} MB |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="prepared/ folder from build_dataset.py")
    ap.add_argument("--runs", required=True, type=Path, help="runs/ folder from train_ml.py / train_dl.py")
    ap.add_argument("--out", required=True, type=Path, help="where to write the model folder and zip")
    ap.add_argument("--run-config", type=Path, help="run_config.json written by run_all.py")
    ap.add_argument("--models", nargs="*", default=list(NAMES))
    ap.add_argument("--select-by", choices=["macro_f1", "accuracy"], default="macro_f1")
    ap.add_argument("--force", choices=list(NAMES), help="ship this model regardless of scores")
    ap.add_argument("--pose-weights", type=Path, help="YOLO-pose .pt used for extraction, copied into the zip")
    args = ap.parse_args()

    classes = load_classes(args.data)
    run_cfg = json.loads(args.run_config.read_text()) if args.run_config else {}

    # ---- 1. collect validation results
    rows = []
    for m in args.models:
        f, model_file = args.runs / f"{m}_val.json", args.runs / f"{m}{SUFFIX[m]}"
        if not (f.exists() and model_file.exists()):
            print(f"  - {m}: not trained, skipped")
            continue
        r = json.loads(f.read_text())
        if r.get("classes") != classes:
            print(f"  ! {m}: trained on different classes {r.get('classes')}, skipped (retrain it)")
            continue
        rows.append({"model": m, "file": model_file, "ms": ms(r), "size_mb": r.get("size_mb", 0.0),
                     "val": {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}, "raw": r})
    if not rows:
        raise SystemExit(f"No trained models found in {args.runs}.")

    if args.force:
        best = next((r for r in rows if r["model"] == args.force), None)
        if best is None:
            raise SystemExit(f"--force {args.force}: that model hasn't been trained.")
    else:
        best = max(rows, key=lambda r: (r["val"][args.select_by], r["val"]["accuracy"], -r["ms"]))
    print(f"Chosen on validation {args.select_by}: {NAMES[best['model']]} "
          f"({best['val'][args.select_by]:.4f})" + (" [forced]" if args.force else ""))

    # ---- 2. test every model once
    Xte, yte, _, _ = load_split(args.data, "test")
    if len(yte):
        for r in rows:
            rep = report(yte, predict(r["file"], Xte), classes)
            extra = {k: r["raw"][k] for k in ("model", "ms_per_window", "ms_per_window_cpu", "size_mb")
                     if k in r["raw"]}
            save_report(args.runs, f"{r['model']}_test", rep, classes, extra)
            r["test"] = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]}
            r["confusion"] = rep["confusion_matrix"]
            print(f"  {NAMES[r['model']]:14s} test accuracy {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}")
    else:
        print("  ! test split is empty: no test scores (validation scores are reported instead)")

    # ---- 3. export the winner
    model_id = f"{best['model']}-{datetime.now():%Y%m%d-%H%M}"
    bundle_dir = args.out / f"gymrec_model_{model_id}"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    clf_name = "classifier" + best["file"].suffix
    shutil.copy2(best["file"], bundle_dir / clf_name)
    (bundle_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    pose_file = None
    pose_name = (run_cfg.get("preprocessing") or {}).get("pose_model", "yolo11n-pose.pt")
    if args.pose_weights and args.pose_weights.exists():
        pose_file = Path(pose_name).name
        shutil.copy2(args.pose_weights, bundle_dir / pose_file)
    else:
        print(f"  ! {pose_name} not found locally; the website will download it on install")

    scores = best.get("test") or best["val"]
    metrics = {**scores, "split": "test" if best.get("test") else "val",
               "val_accuracy": best["val"]["accuracy"], "val_macro_f1": best["val"]["macro_f1"],
               "ms_per_window": best["ms"], "size_mb": best["size_mb"], "model": best["model"]}
    (bundle_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))

    results = table(rows, best["model"], "val") + table(rows, best["model"], "test")
    if best.get("confusion"):
        cm = np.array(best["confusion"])
        np.fill_diagonal(cm, 0)
        top = [(i, j) for i, j in zip(*np.unravel_index(np.argsort(-cm.ravel())[:3], cm.shape)) if cm[i, j]]
        if top:
            results.append("Most confused (true → predicted): " +
                           ", ".join(f"{classes[i]} → {classes[j]} ({cm[i, j]})" for i, j in top))
    (bundle_dir / "results.md").write_text("\n".join(results) + "\n")

    card = {
        "id": model_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "classifier": {"model": best["model"], "name": NAMES[best["model"]], "file": clf_name,
                       "backend": "torchscript" if clf_name.endswith(".pt") else "sklearn"},
        "pose": {"file": pose_file, "name": pose_name},
        "classes": classes,
        "exercises": run_cfg.get("exercises", {}),
        "preprocessing": run_cfg.get("preprocessing", {}),
        "selected_by": "forced" if args.force else f"validation {args.select_by}",
        "metrics": metrics,
        "candidates": [{"model": r["model"], "val": r["val"], "test": r.get("test"), "ms_per_window": r["ms"],
                        "size_mb": r["size_mb"]} for r in rows],
        "library_versions": versions(),
        "data": {k: run_cfg[k] for k in ("data_dirs", "test_dirs") if k in run_cfg},
    }
    (bundle_dir / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))

    zip_path = args.out / f"gymrec_model_{model_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle_dir.iterdir()):
            z.write(f, f"{bundle_dir.name}/{f.name}")

    print("\n" + "\n".join(results))
    print(f"Exported {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"On the laptop:  python manage.py install_model {zip_path.name}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"({time.time() - t0:.0f}s)")
