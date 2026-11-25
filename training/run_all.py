"""
Run the whole training pipeline from gym_config.py and produce a model zip for the website.

    extract  YOLO-pose keypoints for the selected classes   (slow; resumable)
    build    clean, split by video, cut windows             -> WORK_DIR/prepared/
    train    every model in config.MODELS                    -> WORK_DIR/runs/
    select   best model on the validation set, test every
             model once, export the winner                   -> WORK_DIR/gymrec_model_<id>.zip

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
def folder_map() -> dict[str, str]:
    """Dataset folder slug -> class slug, for the selected classes only."""
    unknown = [c for c in config.CLASSES if c not in config.EXERCISES]
    if unknown:
        raise SystemExit(f"config.CLASSES has classes without an entry in config.EXERCISES: {unknown}")
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
    return {c: per_class[c] for c in config.CLASSES}


# ----------------------------------------------------------------- steps
def run(cmd: list[str]) -> None:
    """Run a training script, streaming its output (also visible inside notebooks)."""
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
    run(cmd)


def step_train():
    _, xgb_device = _device()
    data, runs = work() / "prepared", work() / "runs"
    for m in config.MODELS:
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
        run(cmd)


def step_select():
    snapshot = work() / "run_config.json"
    snapshot.write_text(json.dumps({
        "classes": config.CLASSES,
        "exercises": {c: config.EXERCISES[c]["website"] for c in config.CLASSES},
        "preprocessing": {"pose_model": config.POSE_MODEL, "fps": config.FPS, "imgsz": config.IMGSZ,
                          "window": config.WINDOW, "stride": config.STRIDE, "min_conf": config.MIN_CONF},
        "data_dirs": config.DATA_DIRS, "test_dirs": config.TEST_DIRS,
    }, indent=2, ensure_ascii=False))
    cmd = [sys.executable, HERE / "select_best.py", "--data", work() / "prepared", "--runs", work() / "runs",
           "--out", work(), "--run-config", snapshot, "--models", *config.MODELS, "--select-by", config.SELECT_BY]
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
    work().mkdir(parents=True, exist_ok=True)
    print(f"Classes: {config.CLASSES}\nModels:  {config.MODELS}\nWork dir: {work()}")
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
