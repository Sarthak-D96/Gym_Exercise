"""
Extra per-frame features for the two-stage classifier.

They are computed from the 42 base features (features.py), so the website and
the training scripts (training/common.py imports this file) produce exactly the
same numbers. They spell out body geometry that separates look-alike exercises:

    torso tilt            upright (0) .. lying flat (0.5): flat vs incline bench, push-up vs plank
    shoulder / hip width  how square to the camera the body is
    wrist / elbow height  relative to the shoulders: presses vs curls vs pulls
    wrist reach           horizontal wrist-to-shoulder distance
    shin angle            knee travelling forward: squat vs deadlift vs RDL
    body aspect           standing (tall) vs lying (wide)
    grip width            distance between the wrists
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_SIZE
from .types import KEYPOINT_NAMES

_I = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
EXTRA_NAMES = [
    "torso_tilt", "shoulder_width", "hip_width",
    "l_wrist_dy", "r_wrist_dy", "l_wrist_dx", "r_wrist_dx",
    "l_elbow_dy", "r_elbow_dy", "l_shin_angle", "r_shin_angle",
    "body_aspect", "grip_width",
]
EXTRA_SIZE = len(EXTRA_NAMES)
EXTENDED_SIZE = FEATURE_SIZE + EXTRA_SIZE  # 55


def _xy(f: np.ndarray, name: str) -> np.ndarray:
    i = _I[name]
    return f[..., 2 * i:2 * i + 2]


def extra(f: np.ndarray) -> np.ndarray:
    """(..., 42) base features -> (..., 13) extra features."""
    f = np.asarray(f, dtype=np.float32)
    ls, rs = _xy(f, "left_shoulder"), _xy(f, "right_shoulder")
    lh, rh = _xy(f, "left_hip"), _xy(f, "right_hip")
    le, re = _xy(f, "left_elbow"), _xy(f, "right_elbow")
    lw, rw = _xy(f, "left_wrist"), _xy(f, "right_wrist")
    lk, rk = _xy(f, "left_knee"), _xy(f, "right_knee")
    la, ra = _xy(f, "left_ankle"), _xy(f, "right_ankle")
    sh = (ls + rs) / 2  # coordinates are already relative to the hip midpoint

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    def from_vertical(top, bottom):
        d = top - bottom
        return np.arctan2(np.abs(d[..., 0]), np.abs(d[..., 1]) + 1e-6) / np.pi

    pts = f[..., :FEATURE_SIZE - 8].reshape(*f.shape[:-1], len(KEYPOINT_NAMES), 2)
    span = pts.max(-2) - pts.min(-2)
    cols = [
        np.arctan2(np.abs(sh[..., 0]), -sh[..., 1]) / np.pi,
        dist(ls, rs), dist(lh, rh),
        lw[..., 1] - ls[..., 1], rw[..., 1] - rs[..., 1],
        np.abs(lw[..., 0] - ls[..., 0]), np.abs(rw[..., 0] - rs[..., 0]),
        le[..., 1] - ls[..., 1], re[..., 1] - rs[..., 1],
        from_vertical(lk, la), from_vertical(rk, ra),
        span[..., 0] / np.maximum(span[..., 1], 1e-3),
        dist(lw, rw),
    ]
    out = np.stack(cols, -1)
    return np.round(np.clip(np.nan_to_num(out), -10, 10), 4).astype(np.float32)


def extend(f: np.ndarray) -> np.ndarray:
    """(..., 42) -> (..., 55): base features followed by the extra ones."""
    f = np.asarray(f, dtype=np.float32)
    return np.concatenate([f, extra(f)], -1)
