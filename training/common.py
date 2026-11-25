"""Shared helpers for the training scripts (Phase 3 + 4 in JOURNAL.md).

The feature code here is a vectorised NumPy copy of
recognition/pipeline/features.py, so training and the website compute the
same 42 numbers per frame. build_dataset.py checks that both give identical
results before it writes anything.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# COCO-17 order, same as recognition/pipeline/types.py
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
IDX = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
ANGLE_TRIPLETS = [
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_shoulder", "left_elbow"),
    ("right_hip", "right_shoulder", "right_elbow"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
]
FEATURE_SIZE = 17 * 2 + len(ANGLE_TRIPLETS)  # 42

# index permutation that swaps left_* and right_* keypoints (for mirroring)
FLIP_PERM = np.array([
    IDX[n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")]
    for n in KEYPOINT_NAMES
])


def features_np(kp: np.ndarray) -> np.ndarray:
    """kp: (..., 17, 2+) normalised x,y -> (..., 42) features, same as features.extract_features."""
    xy = kp[..., :2].astype(np.float64)
    hip = (xy[..., IDX["left_hip"], :] + xy[..., IDX["right_hip"], :]) / 2
    sh = (xy[..., IDX["left_shoulder"], :] + xy[..., IDX["right_shoulder"], :]) / 2
    torso = np.maximum(np.linalg.norm(sh - hip, axis=-1), 1e-6)
    coords = (xy - hip[..., None, :]) / torso[..., None, None]
    coords = coords.reshape(*coords.shape[:-2], 34)
    angles = []
    for a, b, c in ANGLE_TRIPLETS:
        v1 = xy[..., IDX[a], :] - xy[..., IDX[b], :]
        v2 = xy[..., IDX[c], :] - xy[..., IDX[b], :]
        n1 = np.linalg.norm(v1, axis=-1)
        n2 = np.linalg.norm(v2, axis=-1)
        ok = (n1 > 0) & (n2 > 0)
        cos = np.where(ok, (v1 * v2).sum(-1) / np.where(ok, n1 * n2, 1), 1.0)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        angles.append(np.where(ok, ang, 0.0) / 180.0)
    out = np.concatenate([coords, np.stack(angles, -1)], -1)
    return np.round(out, 4).astype(np.float32)


def check_against_website(kp_frames: np.ndarray, n: int = 50) -> None:
    """Assert features_np == recognition.pipeline.features.extract_features on some frames."""
    from recognition.pipeline.features import extract_features
    from recognition.pipeline.types import Keypoint

    rng = np.random.default_rng(0)
    pick = kp_frames[rng.choice(len(kp_frames), size=min(n, len(kp_frames)), replace=False)]
    mine = features_np(pick)
    for frame, f in zip(pick, mine):
        ref = extract_features({nm: Keypoint(float(frame[i, 0]), float(frame[i, 1]), float(frame[i, 2]))
                                for i, nm in enumerate(KEYPOINT_NAMES)})
        if np.abs(np.array(ref) - f).max() > 2e-4:
            raise AssertionError("NumPy features differ from the website's features.py")


def slugify(name: str) -> str:
    """'barbell biceps curl' -> 'barbell_biceps_curl', 'push-up' -> 'push_up'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def window_stats(X: np.ndarray) -> np.ndarray:
    """(N, T, 42) -> (N, 210): mean, std, min, max, range over time (for RF / XGBoost)."""
    mn, mx = X.min(1), X.max(1)
    return np.concatenate([X.mean(1), X.std(1), mn, mx, mx - mn], 1)


def load_split(data_dir: Path, name: str):
    d = np.load(data_dir / f"{name}.npz", allow_pickle=False)
    return d["X"], d["y"], d["kp"], d["video"]


def load_classes(data_dir: Path) -> list[str]:
    return json.loads((data_dir / "classes.json").read_text())


# ---------------------------------------------------------------- two-stage
def extend_features(X: np.ndarray) -> np.ndarray:
    """(..., 42) -> (..., 55): the website's recognition/pipeline/extra_features.py, shared code."""
    from recognition.pipeline.extra_features import extend

    return extend(X)


def load_stage(stage_file: Path | None, data_dir: Path):
    """
    A stage of the two-stage model, written by run_all.py as JSON:
        {"name": "legs", "classes": [...output labels...], "map": {dataset class: output label}}
    The router maps every class to its group; a specialist maps only its group's classes.
    Without a stage file this is the plain single-stage setup.
    """
    if stage_file is None:
        return None, load_classes(data_dir)
    stage = json.loads(Path(stage_file).read_text())
    return stage, stage["classes"]


def load_data(data_dir: Path, split: str, stage: dict | None = None, extra: bool = False):
    """load_split, restricted/relabelled for a stage, optionally with the extra features."""
    X, y, kp, video = load_split(data_dir, split)
    if stage is not None:
        base = load_classes(data_dir)
        out = stage["classes"]
        new = np.array([out.index(stage["map"][c]) if c in stage["map"] else -1 for c in base], dtype=np.int64)
        y = new[y] if len(y) else y
        keep = y >= 0
        X, y, kp, video = X[keep], y[keep], kp[keep], video[keep]
    if extra:
        X = extend_features(X)
    return X, y, kp, video


def report(y_true, y_pred, classes) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_report(out_dir: Path, name: str, rep: dict, classes: list[str], extra: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({**rep, **(extra or {}), "classes": classes}, indent=2))
    cm = np.array(rep["confusion_matrix"])
    lines = ["true\\pred," + ",".join(classes)]
    lines += [classes[i] + "," + ",".join(map(str, row)) for i, row in enumerate(cm)]
    (out_dir / f"{name}_confusion.csv").write_text("\n".join(lines) + "\n")
