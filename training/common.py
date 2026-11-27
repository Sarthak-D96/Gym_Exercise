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


def augment(kp: np.ndarray, rng: np.random.Generator, strength: str = "basic") -> np.ndarray:
    """
    kp: (B, T, 17, 3) -> augmented copy. Applied to raw keypoints, features are recomputed after.

    basic   left/right mirror, rotation ±10°, keypoint jitter, time stretch ±20 %
    strong  + camera yaw (horizontal squash 0.65..1) and pitch (vertical 0.85..1.1), rotation ±15°,
              more jitter, 1-3 keypoints per window replaced by wrong positions in some frames
              (YOLO-pose mistakes on blurry / occluded frames), dropped-and-repeated frames
    The dataset's test videos are noisier and filmed differently from its training videos;
    "strong" trains for that.
    """
    kp = kp.copy()
    B, T = kp.shape[:2]
    strong = strength == "strong"
    flip = rng.random(B) < 0.5
    kp[flip] = kp[flip][:, :, FLIP_PERM]
    kp[flip, :, :, 0] = 1.0 - kp[flip, :, :, 0]
    c = kp[:, :, 11:13, :2].mean(2, keepdims=True)  # hip centre per frame
    x, y = kp[..., 0] - c[..., 0], kp[..., 1] - c[..., 1]
    if strong:  # camera yaw / pitch: the body looks narrower / shorter
        x = x * rng.uniform(0.65, 1.0, B)[:, None, None]
        y = y * rng.uniform(0.85, 1.1, B)[:, None, None]
    lim = 15 if strong else 10
    th = np.radians(rng.uniform(-lim, lim, B))[:, None, None]
    kp[..., 0] = c[..., 0] + x * np.cos(th) - y * np.sin(th)
    kp[..., 1] = c[..., 1] + x * np.sin(th) + y * np.cos(th)
    kp[..., :2] += rng.normal(0, 0.01 if strong else 0.005, kp[..., :2].shape)
    if strong:
        for i in np.flatnonzero(rng.random(B) < 0.3):  # corrupted keypoints
            joints = rng.choice(17, size=rng.integers(1, 4), replace=False)
            frames = rng.random(T) < rng.uniform(0.2, 0.6)
            for j in joints:
                kp[i, frames, j, :2] = kp[i, frames, j, :2] + rng.normal(0, 0.08, (frames.sum(), 2))
        for i in np.flatnonzero(rng.random(B) < 0.3):  # dropped frames (repeat the previous one)
            for t in np.flatnonzero(rng.random(T - 1) < 0.15) + 1:
                kp[i, t] = kp[i, t - 1]
    for i in np.flatnonzero(rng.random(B) < 0.5):  # time stretch ±20 %
        span = T * rng.uniform(0.8, 1.2)
        start = rng.uniform(0, max(T - span, 0)) if span < T else 0.0
        src = np.clip(np.round(start + np.linspace(0, span - 1, T)).astype(int), 0, T - 1)
        kp[i] = kp[i, src]
    return kp


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


def smoothed_accuracy(pred: np.ndarray, y: np.ndarray, video: np.ndarray, k: int = 8) -> float:
    """
    Accuracy after the website's smoothing (recognition/pipeline/refine.py): each window's
    label is the majority of the last k predictions of the same video. Windows of a video are
    stored in time order by build_dataset.py. (The website's 0.6 confidence threshold is ignored.)
    """
    if not len(y):
        return float("nan")
    out = np.empty_like(pred)
    start = 0
    for i in range(len(pred)):
        if i and video[i] != video[i - 1]:
            start = i
        out[i] = np.bincount(pred[max(start, i - k + 1):i + 1]).argmax()
    return float((out == y).mean())


def video_accuracy(pred: np.ndarray, y: np.ndarray, video: np.ndarray) -> float:
    """Share of videos whose majority prediction is right (one vote per video)."""
    if not len(y):
        return float("nan")
    right = []
    for v in np.unique(video):
        m = video == v
        right.append(np.bincount(pred[m]).argmax() == y[m][0])
    return float(np.mean(right))


def append_history(out_dir: Path, entry: dict) -> list[dict]:
    """WORK_DIR/score_history.json: one line per exported model, to compare runs in the journal."""
    f = out_dir / "score_history.json"
    hist = json.loads(f.read_text()) if f.exists() else []
    hist.append(entry)
    f.write_text(json.dumps(hist, indent=2))
    return hist


def history_table(hist: list[dict], last: int = 8) -> list[str]:
    lines = ["| Run | Approach | Settings | Test acc | Test F1 | Website-smoothed acc | Video acc |",
             "|---|---|---|---|---|---|---|"]
    for h in hist[-last:]:
        def f(k):
            v = h.get(k)
            return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "–"
        lines.append(f"| {h.get('id', '')} | {h.get('approach', '')} | {h.get('settings', '')} | {f('test_accuracy')} | "
                     f"{f('test_macro_f1')} | {f('test_smoothed_accuracy')} | {f('test_video_accuracy')} |")
    return lines


def save_report(out_dir: Path, name: str, rep: dict, classes: list[str], extra: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({**rep, **(extra or {}), "classes": classes}, indent=2))
    cm = np.array(rep["confusion_matrix"])
    lines = ["true\\pred," + ",".join(classes)]
    lines += [classes[i] + "," + ",".join(map(str, row)) for i, row in enumerate(cm)]
    (out_dir / f"{name}_confusion.csv").write_text("\n".join(lines) + "\n")
