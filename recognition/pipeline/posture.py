"""
Processes 6/7 — posture analysis and feedback.

Rule-based checks on joint geometry. Each rule returns a Feedback when the
pose violates it. Thresholds are deliberately simple and easy to tune; a
learned posture model can replace `analyze()` later (MLModel kind "posture").
"""

from .geometry import angle, midpoint, tilt_from_vertical
from .types import Feedback, Keypoint


def _torso_lean(kp):
    return tilt_from_vertical(midpoint(kp["left_hip"], kp["right_hip"]),
                              midpoint(kp["left_shoulder"], kp["right_shoulder"]))


def _upper_arm_abduction(kp):
    return max(angle(kp["left_hip"], kp["left_shoulder"], kp["left_elbow"]),
               angle(kp["right_hip"], kp["right_shoulder"], kp["right_elbow"]))


def _body_line(kp):
    return min(angle(kp["left_shoulder"], kp["left_hip"], kp["left_knee"]),
               angle(kp["right_shoulder"], kp["right_hip"], kp["right_knee"]))


RULES = {
    "squat": [
        (lambda kp: _torso_lean(kp) > 28, "chest_up", "Keep your chest up — you're leaning too far forward."),
    ],
    "lunge": [
        (lambda kp: _torso_lean(kp) > 22, "torso_upright", "Keep your torso upright."),
    ],
    "push_up": [
        (lambda kp: _body_line(kp) < 160, "hips_line", "Keep your hips in line with your shoulders."),
    ],
    "bicep_curl": [
        (lambda kp: _upper_arm_abduction(kp) > 30, "elbows_pinned", "Keep your elbows pinned to your sides."),
    ],
    "shoulder_press": [
        (lambda kp: _torso_lean(kp) > 15, "no_lean_back", "Avoid arching / leaning back while pressing."),
    ],
    "jumping_jack": [],
}

PARTIAL_MESSAGES = {
    "squat": "Partial rep — squat deeper (thighs parallel to the floor).",
    "lunge": "Partial rep — lower your back knee further.",
    "push_up": "Partial rep — lower your chest closer to the floor.",
    "bicep_curl": "Partial rep — curl all the way up.",
    "shoulder_press": "Partial rep — fully lock out your arms overhead.",
    "jumping_jack": "Partial rep — bring your arms all the way overhead.",
    "bench_press": "Partial rep — lower the bar to your chest and press to full lockout.",
    "pull_up": "Partial rep — pull until your chin clears the bar, then lower to a full hang.",
}


def analyze(slug: str, kp: dict[str, Keypoint]) -> list[Feedback]:
    issues = []
    for check, code, message in RULES.get(slug, []):
        if check(kp):
            issues.append(Feedback("warning", code, message))
    return issues


def partial_rep_feedback(slug: str) -> Feedback:
    return Feedback("warning", "partial_rep", PARTIAL_MESSAGES.get(slug, "Partial rep — use full range of motion."))
