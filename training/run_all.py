"""
Run the whole training pipeline from gym_config.py and produce a model zip for the website.

    extract  YOLO-pose keypoints for the selected classes   (slow; resumable)
    build    clean, split by video, cut windows             -> WORK_DIR/prepared/
    train    every model in config.MODELS                    -> WORK_DIR/runs/
    select   best model on the validation set, test every
             model once, export the winner                   -> WORK_DIR/gymrec_model_<id>.zip

With config.TWO_STAGE = True, `train` and `select` instead build the two-stage
model (router -> exercise group, one specialist per group; every stage gets
its best model type) in WORK_DIR/two_stage/ and export
WORK_DIR/gymrec_model_two_stage-<date>.zip. Same install command.

Command line:
    python training/run_all.py                    # all steps
    python training/run_all.py --preview          # only show which folders map to which classes
    python training/run_all.py --steps train select

Notebook (values can be overridden first):
    import sys; sys.path.insert(0, "/path/to/Gym_Exercise/training")
    import run_all
    config = run_all.config          # the loaded training/gym_config.py
    config.CLASSES = ["squat", "push_up"]
    run_all.preview()
    run_all.main()

Then on the laptop:  python manage.py install_model gymrec_model_<id>.zip
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from common import slugify  # noqa: E402


def _load_config():
    """Load training/gym_config.py by path, so another module called 'config' can't shadow it."""
    import importlib.util

    path = HERE / "gym_config.py"
    mod = sys.modules.get("gym_config")
    if mod is not None and Path(getattr(mod, "__file__", "")).resolve() == path.resolve():
        return mod
    spec = importlib.util.spec_from_file_location("gym_config", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gym_config"] = mod
    spec.loader.exec_module(mod)
    return mod


config = _load_config()  # notebooks: `config = run_all.config`, then override values

STEPS = ["extract", "build", "train", "select"]
ML_MODELS, DL_MODELS = {"rf", "xgb"}, {"lstm", "gru", "cnn"}


# ----------------------------------------------------------------- config helpers
def resolve_classes() -> list[str]:
    """
    Turn config.CLASSES into slugs. Entries may be slugs ('push_up') or dataset
    folder names ('push-up', 'barbell biceps curl', 'pull Up'). The list is
    rewritten in place so every later step sees slugs.
    """
    by_folder = {slugify(f): slug for slug, ex in config.EXERCISES.items() for f in [slug, *ex["folders"]]}
    resolved, unknown = [], []
    for entry in config.CLASSES:
        slug = by_folder.get(slugify(entry))
        if slug is None:
            unknown.append(entry)
        elif slug not in resolved:
            resolved.append(slug)
    if unknown:
        raise SystemExit(
            f"Unknown classes in config.CLASSES: {unknown}.\n"
            f"Use one of these slugs (or one of their dataset folder names in gym_config.EXERCISES):\n  "
            + ", ".join(config.EXERCISES))
    config.CLASSES[:] = resolved
    return resolved


def folder_map() -> dict[str, str]:
    """Dataset folder slug -> class slug, for the selected classes only."""
    resolve_classes()
    mapping: dict[str, str] = {}
    for cls in config.CLASSES:
        for folder in [cls, *config.EXERCISES[cls]["folders"]]:
            key = slugify(folder)
            if mapping.get(key, cls) != cls:
                raise SystemExit(f"Folder '{folder}' is listed under both '{mapping[key]}' and '{cls}'.")
            mapping[key] = cls
    return mapping


def work() -> Path:
    return Path(config.WORK_DIR)


def two_stage() -> bool:
    return bool(getattr(config, "TWO_STAGE", False))


def group_plan() -> dict[str, list[str]]:
    """config.GROUPS restricted to config.CLASSES (sorted, as the trainers' label order)."""
    resolve_classes()
    groups, seen = {}, {}
    for g, members in getattr(config, "GROUPS", {}).items():
        g = slugify(g)
        picked = sorted(c for c in config.CLASSES if c in members)
        for c in picked:
            if c in seen:
                raise SystemExit(f"'{c}' is in two groups: '{seen[c]}' and '{g}' (config.GROUPS).")
            seen[c] = g
        if picked:
            groups[g] = picked
    for c in config.CLASSES:
        if c not in seen:
            groups[c] = [c]  # not in any group: its own group
    return dict(sorted(groups.items()))


def write_stages() -> Path:
    """WORK_DIR/two_stage/plan.json + stages/<name>.json, read by the trainers and select_two_stage.py."""
    groups = group_plan()
    root = work() / "two_stage"
    (root / "stages").mkdir(parents=True, exist_ok=True)
    stages = []
    if len(groups) > 1:
        router = {"name": "router", "classes": list(groups), "map": {c: g for g, cs in groups.items() for c in cs}}
        _write_if_changed(root / "stages" / "router.json", json.dumps(router, indent=2))
        stages.append("router")
    for g, cs in groups.items():
        if len(cs) > 1:
            _write_if_changed(root / "stages" / f"{g}.json",
                              json.dumps({"name": g, "classes": cs, "map": {c: c for c in cs}}, indent=2))
            stages.append(g)
    plan = {"groups": groups, "stages": stages}
    (root / "plan.json").write_text(json.dumps(plan, indent=2))
    return root / "plan.json"


def show_groups() -> None:
    groups = group_plan()
    print("Two-stage plan:" + ("" if len(groups) > 1 else " (only one group: no router needed)"))
    for g, cs in groups.items():
        print(f"  {g:10s} {'specialist' if len(cs) > 1 else 'no model  '}  {', '.join(cs)}")


def _device():
    try:
        import torch

        if torch.cuda.is_available():
            return "0", "cuda"
        if torch.backends.mps.is_available():
            return "mps", "cpu"
    except ImportError:
        pass
    return "cpu", "cpu"


# ----------------------------------------------------------------- preview
def scan() -> dict[str, dict[str, Counter]]:
    """{'train': {folder_name: Counter(dir)}, 'test': {...}} for every class folder found."""
    from extract_keypoints import find_items, label_of

    found = {"train": {}, "test": {}}
    for part, dirs in (("train", config.DATA_DIRS), ("test", config.TEST_DIRS)):
        for d in map(Path, dirs):
            if not d.exists():
                print(f"  ! {part} folder not found: {d}")
                continue
            for _, p in find_items(d):
                found[part].setdefault(label_of(p, d), Counter())[str(d)] += 1
    return found


def preview() -> dict[str, int]:
    """Print which dataset folders feed which class. Returns videos per selected class."""
    mapping = folder_map()
    found = scan()
    per_class = Counter()
    print(f"{'dataset folder':32s} {'videos':>7s} {'test':>6s}   -> class")
    for name in sorted(set(found["train"]) | set(found["test"]), key=str.lower):
        n_tr = sum(found["train"].get(name, Counter()).values())
        n_te = sum(found["test"].get(name, Counter()).values())
        cls = mapping.get(slugify(name))
        per_class[cls] += n_tr if cls else 0
        print(f"{name:32s} {n_tr:7d} {n_te:6d}   -> {cls or '(not used)'}")
    missing = [c for c in config.CLASSES if not per_class[c]]
    print(f"\nSelected classes: {', '.join(config.CLASSES)}")
    if missing:
        print(f"  ! No videos found for: {', '.join(missing)}. Add their folder names to "
              "config.EXERCISES[...]['folders'] or remove them from config.CLASSES.")
    if two_stage():
        print()
        show_groups()
    return {c: per_class[c] for c in config.CLASSES}


# ----------------------------------------------------------------- steps
def run(cmd: list[str]) -> None:
    """Run a training script, streaming its output (also visible inside notebooks)."""
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # child output is a pipe: without this, prints arrive in chunks
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"Step failed ({proc.returncode}): {' '.join(cmd)}")


def step_extract():
    pose_device, _ = _device()
    folders = sorted(folder_map())
    for part, dirs in (("keypoints", config.DATA_DIRS), ("keypoints_test", config.TEST_DIRS)):
        for i, d in enumerate(dirs):
            run([sys.executable, HERE / "extract_keypoints.py", "--data", d, "--out", work() / part / f"src{i}",
                 "--pose-model", config.POSE_MODEL, "--fps", config.FPS, "--frames-fps", config.FRAMES_FPS,
                 "--imgsz", config.IMGSZ, "--device", pose_device, "--labels", *folders])


def step_build():
    aliases = [f"{folder}={cls}" for folder, cls in folder_map().items() if folder != cls]
    cmd = [sys.executable, HERE / "build_dataset.py", "--keypoints", work() / "keypoints",
           "--out", work() / "prepared", "--classes", *config.CLASSES,
           "--window", config.WINDOW, "--stride", config.STRIDE, "--min-conf", config.MIN_CONF,
           "--val", config.VAL_FRACTION, "--test", config.TEST_FRACTION, "--seed", config.SEED]
    if aliases:
        cmd += ["--alias", *aliases]
    if config.TEST_DIRS and (work() / "keypoints_test").exists():
        cmd += ["--test-keypoints", work() / "keypoints_test"]
    if config.GROUP_REGEX:
        cmd += ["--group-regex", config.GROUP_REGEX]
    # skip if nothing changed since the last build (same settings, no newer keypoints): keeps trained models valid
    stamp, out = work() / "prepared" / "build_command.txt", work() / "prepared" / "train.npz"
    key = " ".join(map(str, cmd[1:]))
    if not getattr(config, "RETRAIN", False) and stamp.exists() and stamp.read_text() == key and out.exists():
        newest = max((f.stat().st_mtime for d in ("keypoints", "keypoints_test")
                      for f in (work() / d).rglob("*.npz")), default=0)
        if newest < out.stat().st_mtime:
            print("Dataset already built with these settings, skipped.")
            return
    run(cmd)
    stamp.write_text(key)


def _up_to_date(runs: Path, m: str, inputs: list[Path]) -> bool:
    """True if model m was already trained after its inputs last changed (resume after a disconnect)."""
    if getattr(config, "RETRAIN", False):
        return False
    done = runs / f"{m}_val.json"
    model = runs / f"{m}{'.joblib' if m in ML_MODELS else '.pt'}"
    if not (done.exists() and model.exists()):
        return False
    return all(done.stat().st_mtime > p.stat().st_mtime for p in inputs if p.exists())


def _write_if_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text() != text:  # keep the mtime, so finished stages are skipped
        path.write_text(text)


def _train_models(data: Path, runs: Path, extra: list) -> None:
    _, xgb_device = _device()
    inputs = [data / "train.npz"] + [Path(e) for e in extra if str(e).endswith(".json")]
    for m in config.MODELS:
        if _up_to_date(runs, m, inputs):
            print(f"  {m}: already trained for this data, skipped (config.RETRAIN = True to redo)", flush=True)
            continue
        if m in ML_MODELS:
            cmd = [sys.executable, HERE / "train_ml.py", "--data", data, "--model", m, "--out", runs,
                   "--seed", config.SEED, "--n-estimators", config.ML_PARAMS.get("n_estimators", 500)]
            if m == "xgb":
                cmd += ["--device", xgb_device]
        elif m in DL_MODELS:
            p = config.DL_PARAMS
            cmd = [sys.executable, HERE / "train_dl.py", "--data", data, "--model", m, "--out", runs,
                   "--hidden", p["hidden"], "--epochs", p["epochs"], "--patience", p["patience"],
                   "--batch", p["batch"], "--lr", p["lr"], "--workers-seed", config.SEED]
        else:
            raise SystemExit(f"Unknown model '{m}' in config.MODELS (use rf, xgb, lstm, gru, cnn).")
        run(cmd + extra)


def step_train():
    data = work() / "prepared"
    if not two_stage():
        _train_models(data, work() / "runs", [])
        return
    plan = json.loads(write_stages().read_text())
    show_groups()
    for name in plan["stages"]:
        print(f"\n----- stage: {name} -----", flush=True)
        stage_file = work() / "two_stage" / "stages" / f"{name}.json"
        _train_models(data, work() / "two_stage" / "runs" / name, ["--stage", stage_file, "--extra-features"])


def step_select():
    snapshot = work() / "run_config.json"
    snapshot.write_text(json.dumps({
        "classes": config.CLASSES,
        "exercises": {c: config.EXERCISES[c]["website"] for c in config.CLASSES},
        "preprocessing": {"pose_model": config.POSE_MODEL, "fps": config.FPS, "imgsz": config.IMGSZ,
                          "window": config.WINDOW, "stride": config.STRIDE, "min_conf": config.MIN_CONF},
        "data_dirs": config.DATA_DIRS, "test_dirs": config.TEST_DIRS,
    }, indent=2, ensure_ascii=False))
    if two_stage():
        cmd = [sys.executable, HERE / "select_two_stage.py", "--data", work() / "prepared",
               "--plan", write_stages(), "--runs", work() / "two_stage" / "runs", "--single-runs", work() / "runs",
               "--out", work(), "--run-config", snapshot, "--models", *config.MODELS,
               "--select-by", config.SELECT_BY]
    else:
        cmd = [sys.executable, HERE / "select_best.py", "--data", work() / "prepared", "--runs", work() / "runs",
               "--out", work(), "--run-config", snapshot, "--models", *config.MODELS,
               "--select-by", config.SELECT_BY]
    if config.FORCE_MODEL:
        cmd += ["--force", config.FORCE_MODEL]
    for candidate in (REPO / config.POSE_MODEL, Path.cwd() / config.POSE_MODEL, Path(config.POSE_MODEL)):
        if candidate.exists():
            cmd += ["--pose-weights", candidate]
            break
    run(cmd)


def main(steps: list[str] | None = None) -> None:
    steps = steps or STEPS
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"Unknown steps {bad}; choose from {STEPS}")
    resolve_classes()
    work().mkdir(parents=True, exist_ok=True)
    print(f"Classes: {config.CLASSES}\nModels:  {config.MODELS}\nWork dir: {work()}\n"
          f"Two-stage: {two_stage()}")
    for s in steps:
        print(f"\n{'=' * 20} {s} {'=' * 20}", flush=True)
        {"extract": step_extract, "build": step_build, "train": step_train, "select": step_select}[s]()
    if "select" in steps:
        print(f"\nDone. Download the gymrec_model_*.zip from {work()} and run on the laptop:\n"
              "    python manage.py install_model <path to the zip>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS)
    ap.add_argument("--preview", action="store_true", help="only show the folder -> class mapping")
    a = ap.parse_args()
    preview() if a.preview else main(a.steps)
