"""Phase 4, two-stage version — pick the best model for every stage, test the whole chain, export it.

    stage 1  router       window -> exercise group (legs, press, pull, arms, core, ...)
    stage 2  specialists  one per group with 2+ exercises: window -> exercise inside that group

Every stage is chosen on its own VALIDATION score from the models in runs/<stage>/
(so the router may be an XGBoost and the legs specialist a CNN). The final
probability of an exercise is P(group) x P(exercise | group). The whole chain is
then scored on the validation and test sets, next to the best single-stage model
if one was trained on the same classes (WORK_DIR/runs/), for the report.

Output, like select_best.py:  <out>/gymrec_model_two_stage-<date>/ and .zip with
    two_stage.json        which file is which stage (read by the website)
    router.*, <group>.*   the chosen model of each stage (.pt or .joblib)
    classes.json, pose weights, metrics_test.json, model_card.json, results.md

Normally run by run_all.py when gym_config.TWO_STAGE = True.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (append_history, history_table, load_classes, load_data, report, save_report,
                    smoothed_accuracy, video_accuracy)
from select_best import NAMES, SUFFIX, ms, predict, predict_proba, versions


def name_of(c: dict) -> str:
    return " + ".join(NAMES[m["model"]] for m in c["members"])


def stage_proba(c: dict, X: np.ndarray, n: int) -> np.ndarray:
    """Average of the stage's members' probabilities (one member = a single model)."""
    return np.mean([predict_proba(m["file"], X, n) for m in c["members"]], 0)


def choose_stage(rows, spec, data, select_by, force, max_k):
    """
    Best single model on validation, then the average of the top 2..max_k if that scores
    higher on validation (by at least 0.002, so a tie keeps the faster single model).
    """
    if force:
        hit = next((r for r in rows if r["model"] == force), None)
        if hit:
            return {"members": [hit], "val": hit["val"]}
        print(f"  ! FORCE_MODEL {force} not trained for this stage; choosing by score")
    ranked = sorted(rows, key=lambda r: (r["val"][select_by], r["val"]["accuracy"], -r["ms"]), reverse=True)
    best = {"members": ranked[:1], "val": ranked[0]["val"]}
    if max_k > 1 and len(ranked) > 1:
        Xv, yv, _, _ = load_data(data, "val", spec, extra=True)
        if len(yv):
            n = len(spec["classes"])
            probs = [predict_proba(r["file"], Xv, n) for r in ranked[:max_k]]
            for k in range(2, min(max_k, len(ranked)) + 1):
                rep = report(yv, np.mean(probs[:k], 0).argmax(-1), spec["classes"])
                if rep[select_by] > best["val"][select_by] + 0.002:
                    best = {"members": ranked[:k], "val": {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]}}
    return best


def candidates(runs: Path, models: list[str], classes: list[str]) -> list[dict]:
    rows = []
    for m in models:
        f, model_file = runs / f"{m}_val.json", runs / f"{m}{SUFFIX[m]}"
        if not (f.exists() and model_file.exists()):
            continue
        r = json.loads(f.read_text())
        if r.get("classes") != classes:
            print(f"  ! {runs.name}/{m}: trained on different classes, skipped (retrain it)")
            continue
        rows.append({"model": m, "file": model_file, "ms": ms(r), "size_mb": r.get("size_mb", 0.0),
                     "features": r.get("features", "base"),
                     "val": {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}})
    return rows


def choose(rows: list[dict], select_by: str, force: str | None) -> dict:
    if force:
        hit = next((r for r in rows if r["model"] == force), None)
        if hit:
            return hit
        print(f"  ! FORCE_MODEL {force} not trained for this stage; choosing by score")
    return max(rows, key=lambda r: (r["val"][select_by], r["val"]["accuracy"], -r["ms"]))


def chain_proba(plan: dict, chosen: dict, X: np.ndarray, classes: list[str]) -> np.ndarray:
    """P(exercise) = P(group) * P(exercise | group), for (N, T, 55) windows."""
    P = np.zeros((len(X), len(classes)), np.float32)
    groups = list(plan["groups"])
    if chosen.get("router"):
        Pg = stage_proba(chosen["router"], X, len(groups))
    else:
        Pg = np.ones((len(X), 1), np.float32)
    for gi, g in enumerate(groups):
        members = plan["groups"][g]
        idx = [classes.index(c) for c in members]
        if len(members) == 1:
            P[:, idx[0]] = Pg[:, gi]
        else:
            P[:, idx] = Pg[:, gi:gi + 1] * stage_proba(chosen[g], X, len(members))
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="prepared/ folder from build_dataset.py")
    ap.add_argument("--plan", required=True, type=Path, help="two_stage/plan.json written by run_all.py")
    ap.add_argument("--runs", required=True, type=Path, help="two_stage/runs/ (one sub-folder per stage)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-config", type=Path)
    ap.add_argument("--models", nargs="*", default=list(NAMES))
    ap.add_argument("--select-by", choices=["macro_f1", "accuracy"], default="macro_f1")
    ap.add_argument("--force", choices=list(NAMES))
    ap.add_argument("--form-profiles", type=Path, help="form_profiles.json from build_form_profiles.py, shipped in the zip")
    ap.add_argument("--pose-weights", type=Path)
    ap.add_argument("--single-runs", type=Path, help="runs/ of a single-stage run on the same classes, to compare")
    ap.add_argument("--ensemble-max", type=int, default=1,
                    help="per stage, try averaging the top 2..N model types (kept only if validation improves)")
    ap.add_argument("--settings", default="", help="short description of this run for score_history.json")
    args = ap.parse_args()

    classes = load_classes(args.data)
    plan = json.loads(args.plan.read_text())
    run_cfg = json.loads(args.run_config.read_text()) if args.run_config else {}
    if sorted(c for cs in plan["groups"].values() for c in cs) != sorted(classes):
        raise SystemExit("plan.json groups don't match prepared/classes.json: run the build step again.")

    # ---- 1. best model per stage, on validation
    chosen, per_stage = {}, {}
    for name in plan["stages"]:
        spec = json.loads((args.plan.parent / "stages" / f"{name}.json").read_text())
        rows = candidates(args.runs / name, args.models, spec["classes"])
        if not rows:
            raise SystemExit(f"No trained models for stage '{name}' in {args.runs / name}.")
        best = choose_stage(rows, spec, args.data, args.select_by, args.force, args.ensemble_max)
        c = {**best, "classes": spec["classes"], "model": "+".join(m["model"] for m in best["members"]),
             "ms": sum(m["ms"] for m in best["members"]), "size_mb": sum(m["size_mb"] for m in best["members"])}
        chosen[name] = c
        per_stage[name] = rows
        print(f"{name:12s} -> {name_of(c):30s} val accuracy {c['val']['accuracy']:.3f}  "
              f"macro-F1 {c['val']['macro_f1']:.3f}   ({len(spec['classes'])} classes)")

    # ---- 2. score the whole chain
    scores, confusion = {}, None
    for split in ("val", "test"):
        X, y, _, vid = load_data(args.data, split, extra=True)
        if not len(y):
            continue
        pred = chain_proba(plan, chosen, X, classes).argmax(-1)
        rep = report(y, pred, classes)
        save_report(args.runs, f"two_stage_{split}", rep, classes)
        scores[split] = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"],
                         "smoothed_accuracy": smoothed_accuracy(pred, y, vid),
                         "video_accuracy": video_accuracy(pred, y, vid)}
        if split == "test":
            confusion = rep["confusion_matrix"]
        # each stage on its own (router: right group? specialist: right exercise, given the right group?)
        for name, c in chosen.items():
            spec = json.loads((args.plan.parent / "stages" / f"{name}.json").read_text())
            Xs, ys, _, _ = load_data(args.data, split, spec, extra=True)
            if len(ys):
                r = report(ys, stage_proba(c, Xs, len(spec["classes"])).argmax(-1), spec["classes"])
                c.setdefault("scores", {})[split] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
        print(f"two-stage {split:4s} accuracy {rep['accuracy']:.4f}  macro-F1 {rep['macro_f1']:.4f}  "
              f"(website-smoothed {scores[split]['smoothed_accuracy']:.4f}, per video "
              f"{scores[split]['video_accuracy']:.4f})")
    if not scores:
        raise SystemExit("val and test splits are both empty.")

    # ---- 3. single-stage comparison (if one was trained on the same classes)
    single = None
    if args.single_runs and args.single_runs.exists():
        rows = candidates(args.single_runs, args.models, classes)
        if rows:
            best = choose(rows, args.select_by, None)
            single = {"model": best["model"], "val": best["val"]}
            X, y, _, vid = load_data(args.data, "test")
            if len(y):
                pred = predict(best["file"], X, len(classes))
                r = report(y, pred, classes)
                single["test"] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                                  "smoothed_accuracy": smoothed_accuracy(pred, y, vid)}

    # ---- 4. export
    model_id = f"two_stage-{datetime.now():%Y%m%d-%H%M}"
    bundle = args.out / f"gymrec_model_{model_id}"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    manifest = {"features": "extended", "groups": {}, "router": None}
    for name, c in chosen.items():
        files = []
        for m in c["members"]:
            fname = f"{name}{m['file'].suffix}" if len(c["members"]) == 1 else f"{name}_{m['model']}{m['file'].suffix}"
            shutil.copy2(m["file"], bundle / fname)
            files.append({"file": fname, "model": m["model"],
                          "backend": "torchscript" if fname.endswith(".pt") else "sklearn"})
        entry = {**files[0], "model": c["model"], "classes": c["classes"]}
        if len(files) > 1:
            entry["ensemble"] = files  # the website averages these models' probabilities
        if name == "router":
            manifest["router"] = entry
        else:
            manifest["groups"][name] = entry
    for g, members in plan["groups"].items():
        manifest["groups"].setdefault(g, {"file": None, "model": None, "backend": None, "classes": members})
    manifest["groups"] = {g: manifest["groups"][g] for g in plan["groups"]}  # router output order
    (bundle / "two_stage.json").write_text(json.dumps(manifest, indent=2))
    (bundle / "classes.json").write_text(json.dumps(classes, indent=2))

    pose_file = None
    pose_name = (run_cfg.get("preprocessing") or {}).get("pose_model", "yolo11n-pose.pt")
    if args.pose_weights and args.pose_weights.exists():
        pose_file = Path(pose_name).name
        shutil.copy2(args.pose_weights, bundle / pose_file)
    else:
        print(f"  ! {pose_name} not found locally; the website will download it on install")

    main_scores = scores.get("test") or scores["val"]
    total_ms = sum(c["ms"] for c in chosen.values())
    metrics = {**main_scores, "split": "test" if "test" in scores else "val",
               "val_accuracy": scores.get("val", {}).get("accuracy"),
               "val_macro_f1": scores.get("val", {}).get("macro_f1"),
               "ms_per_window": total_ms, "size_mb": sum(c["size_mb"] for c in chosen.values()),
               "model": "two_stage"}
    (bundle / "metrics_test.json").write_text(json.dumps(metrics, indent=2))

    # ---- results.md
    def fmt(s):
        return f"{s['accuracy']:.3f} | {s['macro_f1']:.3f}" if s else "– | –"

    lines = ["## Two-stage model", "", "Each stage chosen on its validation score.", "",
             "| Stage | Exercises | Chosen model | Val acc | Val F1 | Test acc | Test F1 |",
             "|---|---|---|---|---|---|---|"]
    for name, c in chosen.items():
        what = "groups: " + ", ".join(c["classes"]) if name == "router" else ", ".join(c["classes"])
        sc = c.get("scores", {})
        lines.append(f"| {name} | {what} | {name_of(c)} | {fmt(sc.get('val'))} | {fmt(sc.get('test'))} |")
    singles = [g for g, m in plan["groups"].items() if len(m) == 1]
    if singles:
        lines.append(f"\nOne-exercise groups (no specialist needed): {', '.join(singles)}")
    lines += ["", "Specialist scores assume the router picked the right group.", "",
              "## Whole chain vs single model", "",
              "| Approach | Val acc | Val F1 | Test acc | Test F1 |", "|---|---|---|---|---|",
              f"| **Two-stage** | {fmt(scores.get('val'))} | {fmt(scores.get('test'))} |"]
    if single:
        lines.append(f"| Single {NAMES[single['model']]} | {fmt(single['val'])} | {fmt(single.get('test'))} |")
    else:
        lines.append("| Single model | (not trained on these classes) | | | |")
    lines += ["", "All stages, every candidate (validation):", ""]
    for name, rows in per_stage.items():
        lines.append(f"- {name}: " + ", ".join(f"{NAMES[r['model']]} {r['val']['macro_f1']:.3f}" for r in rows))
    t = scores.get("test")
    if t:
        lines += ["", "## What the website shows", "",
                  "The website labels each frame with the majority of its last 8 predictions (refine.py). "
                  "The same smoothing on the test videos:", "",
                  f"* per window (the score above): **{t['accuracy']:.3f}**",
                  f"* per window, after the website's 8-window majority vote: **{t['smoothed_accuracy']:.3f}**",
                  f"* per video (majority over the whole video): **{t['video_accuracy']:.3f}**", "",
                  "Report the per-window score as the model's accuracy; the other two describe the app."]
    history = append_history(args.out, {
        "id": model_id, "approach": "two-stage", "settings": args.settings,
        "test_accuracy": t["accuracy"] if t else None, "test_macro_f1": t["macro_f1"] if t else None,
        "test_smoothed_accuracy": t["smoothed_accuracy"] if t else None,
        "test_video_accuracy": t["video_accuracy"] if t else None,
        "val_accuracy": scores.get("val", {}).get("accuracy"),
        "stages": {n: c["model"] for n, c in chosen.items()}})
    lines += ["", "## Score history (WORK_DIR/score_history.json)", "", *history_table(history)]
    if confusion:
        cm = np.array(confusion)
        np.fill_diagonal(cm, 0)
        top = [(i, j) for i, j in zip(*np.unravel_index(np.argsort(-cm.ravel())[:5], cm.shape)) if cm[i, j]]
        if top:
            lines += ["", "Most confused on test (true → predicted): " +
                      ", ".join(f"{classes[i]} → {classes[j]} ({cm[i, j]})" for i, j in top)]
    (bundle / "results.md").write_text("\n".join(lines) + "\n")

    card = {
        "id": model_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "classifier": {"model": "two_stage", "name": "Two-stage (router + specialists)", "file": "two_stage.json",
                       "backend": "two_stage",
                       "backends": sorted({m["file"].suffix == ".pt" and "torchscript" or "sklearn"
                                           for c in chosen.values() for m in c["members"]}),
                       "models": sorted({m["model"] for c in chosen.values() for m in c["members"]})},
        "stages": {name: {"model": c["model"], "classes": c["classes"], "val": c["val"],
                          "scores": c.get("scores", {})} for name, c in chosen.items()},
        "groups": plan["groups"],
        "pose": {"file": pose_file, "name": pose_name},
        "classes": classes,
        "exercises": run_cfg.get("exercises", {}),
        "preprocessing": run_cfg.get("preprocessing", {}),
        "selected_by": f"validation {args.select_by} per stage" + (f" (forced {args.force})" if args.force else ""),
        "metrics": metrics,
        "single_stage_comparison": single,
        "library_versions": versions(),
        "data": {k: run_cfg[k] for k in ("data_dirs", "test_dirs") if k in run_cfg},
    }
    if args.form_profiles and args.form_profiles.exists():
        shutil.copy2(args.form_profiles, bundle / "form_profiles.json")
        card["form_profiles"] = "form_profiles.json"
    (bundle / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))

    zip_path = args.out / f"{bundle.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.iterdir()):
            z.write(f, f"{bundle.name}/{f.name}")
    print("\n" + "\n".join(lines))
    print(f"Exported {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"On the laptop:  python manage.py install_model {zip_path.name}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"({time.time() - t0:.0f}s)")
