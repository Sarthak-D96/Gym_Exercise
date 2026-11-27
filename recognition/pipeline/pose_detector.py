"""
4.2 Feature Extraction — body pose / keypoints.

`PoseDetector` is the interface the rest of the system uses (class diagram:
PoseDetector.detectPose() / getLandmarks()). `DummyPoseDetector` is a
placeholder that does NOT look at the image: it synthesises a plausible,
animated skeleton so the rest of the pipeline (rep counting, posture,
UI overlay, history) can be developed and demoed end-to-end.

TODO(model): implement a real detector, e.g. MediaPipe Pose / MoveNet /
YOLO-pose, as a subclass of `PoseDetector`, and register it in
`model_store.POSE_BACKENDS`.
"""

from __future__ import annotations

import hashlib
import math
import random

from PIL import Image

from .types import KEYPOINT_NAMES, Keypoint, PoseResult


class PoseDetector:
    """Base interface for pose detectors."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def detect_pose(self, frame: Image.Image | None, *, timestamp_ms: float, context: dict) -> PoseResult:
        raise NotImplementedError

    def get_landmarks(self, result: PoseResult) -> dict[str, Keypoint]:
        return result.keypoints


# ---------------------------------------------------------------------------
# DUMMY implementation
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


def _rotate(v: tuple[float, float], deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return (v[0] * math.cos(r) - v[1] * math.sin(r), v[0] * math.sin(r) + v[1] * math.cos(r))


# Joint-angle trajectories per exercise. `p` goes 0 -> 1 -> 0 over a rep
# (0 = start position). `fault` lets the dummy produce bad-form reps so the
# posture module has something to react to.
def _params(slug: str, p: float, fault: str | None) -> dict:
    base = {"knee": 174.0, "shoulder": 10.0, "elbow": 168.0, "lean": 3.0, "hip_shift": 0.0}
    if slug == "squat":
        depth = 112 if fault == "shallow" else 82
        base.update(knee=_lerp(174, depth, p), shoulder=_lerp(10, 80, p),
                    lean=_lerp(3, 58 if fault == "lean" else 12, p))
    elif slug == "lunge":
        base.update(knee=_lerp(174, 88, p), lean=_lerp(3, 34 if fault == "lean" else 8, p))
    elif slug == "push_up":
        base.update(shoulder=75, elbow=_lerp(165, 78, p), hip_shift=0.09 * p if fault == "sag" else 0.0)
    elif slug == "bicep_curl":
        base.update(elbow=_lerp(165, 38, p), shoulder=_lerp(8, 42 if fault == "swing" else 12, p))
    elif slug == "shoulder_press":
        top = 140 if fault == "lockout" else 172
        base.update(elbow=_lerp(82, top, p), shoulder=_lerp(95, 168, p),
                    lean=_lerp(3, 22 if fault == "lean" else 5, p))
    elif slug == "jumping_jack":
        base.update(shoulder=_lerp(15, 120 if fault == "partial" else 168, p))
    return base


FAULTS = {
    "squat": ["shallow", "lean"],
    "lunge": ["lean"],
    "push_up": ["sag"],
    "bicep_curl": ["swing"],
    "shoulder_press": ["lockout", "lean"],
    "jumping_jack": ["partial"],
}

AUTO_CYCLE = ["squat", "bicep_curl", "jumping_jack", "push_up", "shoulder_press", "lunge"]


def _build_skeleton(prm: dict, jitter: random.Random) -> dict[str, Keypoint]:
    kp: dict[str, tuple[float, float]] = {}
    shin = thigh = 0.18
    a = (180 - prm["knee"]) / 2
    for side, ax in (("left", 0.56), ("right", 0.44)):
        s = 1 if side == "left" else -1
        ankle = (ax, 0.93)
        knee = (ankle[0] + s * shin * math.sin(math.radians(a)), ankle[1] - shin * math.cos(math.radians(a)))
        hip = (knee[0] - s * thigh * math.sin(math.radians(a)), knee[1] - thigh * math.cos(math.radians(a)))
        kp[f"{side}_ankle"], kp[f"{side}_knee"], kp[f"{side}_hip"] = ankle, knee, hip

    hip_c = ((kp["left_hip"][0] + kp["right_hip"][0]) / 2 + prm["hip_shift"], kp["left_hip"][1])
    kp["left_hip"] = (kp["left_hip"][0] + prm["hip_shift"], kp["left_hip"][1])
    kp["right_hip"] = (kp["right_hip"][0] + prm["hip_shift"], kp["right_hip"][1])
    torso = 0.28
    lean = math.radians(prm["lean"])
    sh_c = (hip_c[0] + torso * math.sin(lean), hip_c[1] - torso * math.cos(lean))

    upper, fore = 0.14, 0.13
    for side in ("left", "right"):
        s = 1 if side == "left" else -1
        sh = (sh_c[0] + s * 0.085, sh_c[1])
        phi = math.radians(prm["shoulder"])
        d1 = (s * math.sin(phi), math.cos(phi))
        elbow = (sh[0] + upper * d1[0], sh[1] + upper * d1[1])
        d2 = _rotate(d1, -s * (180 - prm["elbow"]))
        wrist = (elbow[0] + fore * d2[0], elbow[1] + fore * d2[1])
        kp[f"{side}_shoulder"], kp[f"{side}_elbow"], kp[f"{side}_wrist"] = sh, elbow, wrist

    nose = (sh_c[0] + 0.08 * math.sin(lean), sh_c[1] - 0.09)
    kp["nose"] = nose
    kp["left_eye"], kp["right_eye"] = (nose[0] + 0.018, nose[1] - 0.015), (nose[0] - 0.018, nose[1] - 0.015)
    kp["left_ear"], kp["right_ear"] = (nose[0] + 0.04, nose[1] - 0.005), (nose[0] - 0.04, nose[1] - 0.005)

    return {
        name: Keypoint(
            x=kp[name][0] + jitter.gauss(0, 0.003),
            y=kp[name][1] + jitter.gauss(0, 0.003),
            score=round(jitter.uniform(0.82, 0.99), 3),
        )
        for name in KEYPOINT_NAMES
    }


class DummyPoseDetector(PoseDetector):
    """
    Placeholder pose detector — ignores pixels and returns synthetic keypoints.

    `context` may contain:
      * "simulate_exercise": slug the user selected (the dummy animates it);
        when absent, the dummy cycles through AUTO_CYCLE every 30 s to
        exercise the auto-detect path.
      * "seed": per-session seed so faults are reproducible.
    """

    REP_PERIOD_MS = 2600
    AUTO_SWITCH_MS = 30000

    def detect_pose(self, frame, *, timestamp_ms, context):
        slug = context.get("simulate_exercise")
        if not slug:
            slug = AUTO_CYCLE[int(timestamp_ms // self.AUTO_SWITCH_MS) % len(AUTO_CYCLE)]
        rep_index = int(timestamp_ms // self.REP_PERIOD_MS)
        phase = (timestamp_ms % self.REP_PERIOD_MS) / self.REP_PERIOD_MS
        p = (1 - math.cos(2 * math.pi * phase)) / 2  # 0 -> 1 -> 0

        # Deterministically pick a fault for ~30 % of reps (skip the first two).
        seed = f"{context.get('seed', 0)}:{rep_index}"
        h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        fault = None
        if rep_index >= 2 and slug in FAULTS and h % 10 < 3:
            fault = FAULTS[slug][h // 10 % len(FAULTS[slug])]

        keypoints = _build_skeleton(_params(slug, p, fault), random.Random(timestamp_ms))
        return PoseResult(
            keypoints=keypoints,
            score=0.93,
            meta={"simulated_exercise": slug, "simulated_fault": fault, "dummy": True},
        )
