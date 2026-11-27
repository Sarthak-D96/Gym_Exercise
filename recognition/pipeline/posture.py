"""
Processes 6/7 — posture (form) analysis and graded feedback.

Every rep is measured (depth, lockout, tempo, torso lean, hip line, elbow drift,
left/right symmetry) and each measurement is compared with what correct reps
look like:

  * a form profile learned from the training dataset
    (training/build_form_profiles.py -> Exercise.form_profile), when installed:
    the normal range ends at Tukey's fence of the dataset's reps
    (75th percentile + 1.5 x IQR, capped at the 95th; mirrored for "too low"),
    so a minority of sloppy reps in the dataset doesn't widen it;
  * otherwise the hand-set defaults below.

Severity of one measurement, with `tol` = the check's tolerance:
    ok      inside the normal range (+ tol)
    minor   up to 2 x tol further out    -> rep counted, with a tip
    major   further out than that        -> rep NOT counted (COUNT_ONLY_GOOD_FORM)

The messages say what to fix and by how much (measured vs typical value).
The same thresholds give live cues during a rep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import angle, tilt_from_vertical
from .types import ExerciseSpec, Feedback, Keypoint

VISIBLE = 0.3           # keypoint score needed to use it
BOTH_SIDES = 0.5        # both sides must be this clear for the symmetry check
MIN_PROFILE_REPS = 15   # fewer dataset reps than this: use the defaults instead
PENALTY = {"minor": 15, "major": 40}

JOINT_NAMES = {"knee": "knee", "elbow": "elbow", "hip": "hip", "shoulder": "shoulder"}


# ---------------------------------------------------------------- frame measurements
def _vis(kp: dict[str, Keypoint], *names: str, min_score: float = VISIBLE) -> bool:
    return all(n in kp and kp[n].score >= min_score for n in names)


def _mirror(name: str, side: str) -> str:
    if side == "left" or not name.startswith(("left_", "right_")):
        return name
    return ("right_" + name[5:]) if name.startswith("left_") else ("left_" + name[6:])


def frame_metrics(spec: ExerciseSpec, kp: dict[str, Keypoint], side: str) -> dict[str, float]:
    """Form measurements of one frame, on the body side the rep counter tracks."""
    s = lambda n: f"{side}_{n}"  # noqa: E731
    out: dict[str, float] = {}
    if _vis(kp, s("hip"), s("shoulder")):
        out["torso_lean"] = tilt_from_vertical(kp[s("hip")], kp[s("shoulder")])
    foot = s("ankle") if _vis(kp, s("ankle")) else s("knee")
    if _vis(kp, s("shoulder"), s("hip"), foot):
        sh, hip, ft = kp[s("shoulder")], kp[s("hip")], kp[foot]
        out["body_line"] = angle(sh, hip, ft)
        # which way the hips leave the shoulder-foot line: +1 below it (sag), -1 above it (pike)
        dx, dy = ft.x - sh.x, ft.y - sh.y
        t = ((hip.x - sh.x) * dx + (hip.y - sh.y) * dy) / max(dx * dx + dy * dy, 1e-9)
        out["body_dir"] = 1.0 if hip.y > sh.y + t * dy else -1.0
    if _vis(kp, s("hip"), s("shoulder"), s("elbow")):
        out["elbow_drift"] = angle(kp[s("hip")], kp[s("shoulder")], kp[s("elbow")])
    a, b, c = spec.angle_points
    left = (a, b, c)
    right = tuple(_mirror(p, "right") for p in left)
    if left != right and _vis(kp, *left, *right, min_score=BOTH_SIDES):
        out["symmetry"] = abs(angle(*(kp[p] for p in left)) - angle(*(kp[p] for p in right)))
    return out


# Per rep: smallest / largest value of each frame measurement.
def accumulate(acc: dict, angle_value: float | None, metrics: dict[str, float]) -> None:
    values = dict(metrics)
    if angle_value is not None:
        values["angle"] = angle_value
    for k, v in values.items():
        if k == "body_dir":
            continue
        lo, hi = acc.get(k, [v, v])[:2]
        acc[k] = [min(lo, v), max(hi, v)]
        if k == "body_line" and v <= lo:
            acc["body_dir"] = [metrics.get("body_dir", 1.0)] * 2


def rep_metrics(spec: ExerciseSpec, acc: dict, tempo_s: float | None) -> dict[str, float]:
    """One number per check from a rep's accumulated min/max values."""
    out: dict[str, float] = {}
    if "angle" in acc:
        lo, hi = acc["angle"]
        out["target"], out["start"] = (lo, hi) if spec.start_phase == "high" else (hi, lo)
    if tempo_s is not None:
        out["tempo"] = tempo_s
    for k, pick in (("torso_lean", 1), ("elbow_drift", 1), ("symmetry", 1), ("body_line", 0)):
        if k in acc:
            out[k] = acc[k][pick]
    if "body_dir" in acc:
        out["body_dir"] = acc["body_dir"][0]
    return out


# ---------------------------------------------------------------- checks
@dataclass(frozen=True)
class Check:
    metric: str
    bad: str                 # "high" | "low" | "short_of_target" | "short_of_start"
    tol: float               # units of the metric (degrees, seconds)
    limit: float | None      # default edge of the normal range (None: derived from the spec)
    typical: float | None    # default typical value of a good rep
    minor: str
    major: str
    live: bool = False       # also check every frame while moving
    major_by_direction: dict = field(default_factory=dict)  # body_line: {"sag": ..., "pike": ...}


TARGET_PHRASE = {
    "squat": "squat deeper", "lunge": "lower your back knee further", "push_up": "lower your chest further",
    "bench_press": "lower the bar to your chest", "incline_bench_press": "lower the bar to your chest",
    "decline_bench_press": "lower the bar to your chest", "bicep_curl": "curl the weight higher",
    "hammer_curl": "curl the weight higher", "shoulder_press": "press all the way up",
    "pull_up": "pull higher, chin over the bar", "lat_pulldown": "pull the bar lower, to your upper chest",
    "t_bar_row": "row the weight closer to your chest", "tricep_pushdown": "push down until your arms are straight",
    "tricep_dips": "dip lower", "deadlift": "lower the bar further", "romanian_deadlift": "hinge further",
    "lateral_raise": "raise your arms to shoulder height", "leg_extension": "straighten your legs fully",
    "hip_thrust": "push your hips higher", "leg_raises": "raise your legs higher",
    "chest_fly_machine": "bring your arms further together", "jumping_jack": "bring your arms all the way up",
}
START_PHRASE = {
    "squat": "stand up fully", "lunge": "stand up fully", "push_up": "straighten your arms at the top",
    "bench_press": "press to full lockout", "incline_bench_press": "press to full lockout",
    "decline_bench_press": "press to full lockout", "bicep_curl": "lower the weight all the way",
    "hammer_curl": "lower the weight all the way", "shoulder_press": "lower the weight to shoulder height",
    "pull_up": "lower to a full hang", "lat_pulldown": "let your arms extend fully at the top",
    "deadlift": "stand tall and lock your hips", "romanian_deadlift": "stand tall at the top",
    "tricep_dips": "straighten your arms at the top",
}


def _joint(spec: ExerciseSpec) -> str:
    mid = spec.angle_points[1].split("_", 1)[-1]
    return JOINT_NAMES.get(mid, mid)


ROM_CHECKS = [
    Check("target", "short_of_target", 0, None, None,
          minor="Almost: {phrase} ({joint} {value:.0f}°, aim for about {typical:.0f}°).",
          major="{Phrase}: your {joint} only reached {value:.0f}° (good reps reach about {typical:.0f}°)."),
    Check("start", "short_of_start", 0, None, None,
          minor="{Phrase} between reps ({joint} {value:.0f}°, aim for about {typical:.0f}°).",
          major="{Phrase}: your {joint} only got back to {value:.0f}° (about {typical:.0f}° at the start)."),
    Check("tempo", "low", 0.4, 0.8, 2.0,
          minor="A bit fast ({value:.1f} s): control the movement.",
          major="Much too fast ({value:.1f} s per rep): slow down, good reps take about {typical:.1f} s."),
]
SYMMETRY = Check("symmetry", "high", 6, 15, 5,
                 minor="Slightly uneven: keep both sides moving together ({value:.0f}° apart).",
                 major="Very uneven left/right ({value:.0f}° apart): move both sides together.")
LEAN_FORWARD = dict(minor="Lean a little less ({value:.0f}° from upright): keep your chest up.",
                    major="Too much forward lean ({value:.0f}°, about {typical:.0f}° is normal): chest up, "
                          "brace your core.")
NO_SWING = dict(minor="Keep your body still ({value:.0f}° lean): don't swing.",
                major="You're swinging your body ({value:.0f}° lean): use a lighter weight and stay upright.")
BODY_LINE = Check("body_line", "low", 6, 160, 175, live=True,
                  minor="Keep a straight line from shoulders to feet ({value:.0f}° at the hips).",
                  major="Body not straight ({value:.0f}° at the hips, about {typical:.0f}° is normal).",
                  major_by_direction={
                      "sag": "Hips are sagging ({value:.0f}° at the hips): tighten your core and glutes.",
                      "pike": "Hips are too high ({value:.0f}° at the hips): lower them into one straight line."})
ELBOWS = Check("elbow_drift", "high", 6, 30, 15, live=True,
               minor="Keep your elbows a bit closer to your sides ({value:.0f}°).",
               major="Elbows are swinging forward ({value:.0f}°, about {typical:.0f}° is normal): pin them to "
                     "your sides.")

EXTRA_CHECKS: dict[str, list[Check]] = {
    "squat": [Check("torso_lean", "high", 5, 40, 25, live=True, **LEAN_FORWARD), SYMMETRY],
    "lunge": [Check("torso_lean", "high", 5, 25, 10, live=True, **LEAN_FORWARD)],
    "push_up": [BODY_LINE, SYMMETRY],
    "plank": [BODY_LINE],
    "bicep_curl": [ELBOWS, Check("torso_lean", "high", 4, 12, 4, live=True, **NO_SWING)],
    "hammer_curl": [ELBOWS, Check("torso_lean", "high", 4, 12, 4, live=True, **NO_SWING)],
    "tricep_pushdown": [ELBOWS],
    "shoulder_press": [Check("torso_lean", "high", 4, 15, 5, live=True,
                             minor="Lean back a little less ({value:.0f}°): brace your core.",
                             major="Leaning back too much ({value:.0f}°): brace your core and keep your "
                                   "ribs down."), SYMMETRY],
    "lateral_raise": [Check("torso_lean", "high", 4, 12, 4, live=True, **NO_SWING), SYMMETRY],
    "bench_press": [SYMMETRY], "incline_bench_press": [SYMMETRY], "decline_bench_press": [SYMMETRY],
    "pull_up": [SYMMETRY], "lat_pulldown": [SYMMETRY], "deadlift": [SYMMETRY], "romanian_deadlift": [SYMMETRY],
}


def is_static(spec: ExerciseSpec) -> bool:
    """Holds such as the plank: thresholds that can never be crossed, so no reps."""
    return spec.high_angle - spec.low_angle < 5


def checks_for(spec: ExerciseSpec) -> list[Check]:
    extra = EXTRA_CHECKS.get(spec.slug, [])
    return extra if is_static(spec) else ROM_CHECKS + extra


# ---------------------------------------------------------------- grading
@dataclass
class Issue:
    code: str
    severity: str  # "minor" | "major"
    message: str

    def feedback(self, prefix: str = "") -> Feedback:
        return Feedback("warning" if self.severity == "major" else "info", self.code, prefix + self.message)


@dataclass
class RepGrade:
    severity: str            # "good" | "minor" | "major"
    score: int               # 0..100
    issues: list[Issue]

    @property
    def counted(self) -> bool:
        return self.severity != "major"


def _range(check: Check, spec: ExerciseSpec):
    """(limit, typical, bad direction, tol) for this check: dataset profile first, defaults otherwise."""
    span = max(spec.high_angle - spec.low_angle, 1.0)
    bad, tol = check.bad, check.tol
    if bad in ("short_of_target", "short_of_start"):
        tol = max(5.0, 0.08 * span)
        to_target = bad == "short_of_target"
        at_low = (spec.start_phase == "high") == to_target  # this end of the movement is the low angle
        bad = "high" if at_low else "low"
        typical = spec.low_angle if at_low else spec.high_angle
        limit = typical + 0.1 * span if at_low else typical - 0.1 * span
    else:
        limit, typical = check.limit, check.typical
    prof = ((spec.form_profile or {}).get("metrics") or {}).get(check.metric)
    if prof and prof.get("n", 0) >= MIN_PROFILE_REPS:
        # Tukey's fence (p75 + 1.5 IQR), but never wider than p5..p95: robust to the share of
        # sloppy reps every dataset has, which would otherwise widen the "normal" range
        typical = prof["p50"]
        p25, p75 = prof.get("p25", prof["p5"]), prof.get("p75", prof["p95"])
        iqr = p75 - p25
        if bad == "high":
            limit = min(prof["p95"], p75 + 1.5 * iqr) + tol
        else:
            limit = max(prof["p5"], p25 - 1.5 * iqr) - tol
    return limit, typical, bad, tol


def _message(check: Check, spec: ExerciseSpec, severity: str, value: float, typical: float,
             direction: float | None = None) -> Issue:
    phrase = (TARGET_PHRASE if check.metric == "target" else START_PHRASE).get(spec.slug, (
        "use your full range of motion" if check.metric == "target" else "return fully to the start position"))
    template = check.minor if severity == "minor" else check.major
    if severity == "major" and check.major_by_direction and direction is not None:
        template = check.major_by_direction["sag" if direction > 0 else "pike"]
    msg = template.format(value=value, typical=typical, joint=_joint(spec), phrase=phrase,
                          Phrase=phrase[:1].upper() + phrase[1:])
    return Issue(f"{check.metric}_{severity}", severity, msg)


def grade_value(check: Check, spec: ExerciseSpec, value: float, direction: float | None = None) -> Issue | None:
    limit, typical, bad, tol = _range(check, spec)
    if limit is None:
        return None
    excess = value - limit if bad == "high" else limit - value
    if excess <= 0:
        return None
    severity = "minor" if excess <= 2 * max(tol, 1e-6) else "major"
    return _message(check, spec, severity, value, typical, direction)


def grade_rep(spec: ExerciseSpec, metrics: dict[str, float], *, reached_target: bool = True) -> RepGrade:
    """Grade one rep. reached_target=False is a partial rep: always a major depth fault, never counted."""
    issues = []
    for check in checks_for(spec):
        if check.metric not in metrics:
            continue
        issue = grade_value(check, spec, metrics[check.metric], metrics.get("body_dir"))
        if check.metric == "target" and not reached_target and (issue is None or issue.severity != "major"):
            issue = _message(check, spec, "major", metrics["target"], _range(check, spec)[1])
        if issue:
            issues.append(issue)
    issues.sort(key=lambda i: i.severity != "major")
    severity = "major" if any(i.severity == "major" for i in issues) else "minor" if issues else "good"
    score = max(0, 100 - sum(PENALTY[i.severity] for i in issues)) if reached_target else 0
    return RepGrade(severity, score, issues)


def live_issues(spec: ExerciseSpec, metrics: dict[str, float]) -> list[Issue]:
    """Cues for the current frame (while moving, or always for holds like the plank)."""
    out = []
    for check in checks_for(spec):
        if check.live and check.metric in metrics:
            issue = grade_value(check, spec, metrics[check.metric], metrics.get("body_dir"))
            if issue:
                out.append(issue)
    return out
