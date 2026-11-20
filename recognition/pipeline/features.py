"""Keypoints -> fixed-length feature vector consumed by the exercise classifier."""

from .geometry import angle, midpoint
from .types import KEYPOINT_NAMES, Keypoint

# Joint angles that are informative for distinguishing gym exercises.
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

FEATURE_SIZE = len(KEYPOINT_NAMES) * 2 + len(ANGLE_TRIPLETS)


def extract_features(kp: dict[str, Keypoint]) -> list[float]:
    """
    Translation/scale-normalised keypoints (centred on the hip midpoint, scaled
    by torso length) followed by joint angles scaled to 0..1.
    Length: FEATURE_SIZE (= 42).
    """
    hip = midpoint(kp["left_hip"], kp["right_hip"])
    sh = midpoint(kp["left_shoulder"], kp["right_shoulder"])
    torso = max(((sh.x - hip.x) ** 2 + (sh.y - hip.y) ** 2) ** 0.5, 1e-6)
    coords: list[float] = []
    for name in KEYPOINT_NAMES:
        p = kp[name]
        coords += [(p.x - hip.x) / torso, (p.y - hip.y) / torso]
    angles = [angle(kp[a], kp[b], kp[c]) / 180.0 for a, b, c in ANGLE_TRIPLETS]
    return [round(v, 4) for v in coords + angles]
