"""
Process 5 — repetition counting.

Tracks one joint angle per exercise (ExerciseSpec.angle_points) and maps it to
`progress` in [0, 1] (0 = start position, 1 = target position). A rep is a
full excursion start -> target -> start. Excursions that get past
PARTIAL_MIN but not to the target are reported as partial reps (not counted).

Camera input is noisy and only arrives at a few frames per second, so:
  * the angle is read from whichever body side the camera sees best
    (angle_points name the left side; the right side is the mirror image);
  * frames where those keypoints are unreliable are skipped, not counted;
  * "reached the target" and "back at the start" use tolerance bands
    (hysteresis) instead of the exact low/high angles, which a smoothed,
    sparsely sampled signal rarely hits.
"""

from dataclasses import asdict, dataclass, field, fields

from .geometry import angle
from .types import ExerciseSpec, Keypoint

TARGET_REACHED = 0.75  # progress that counts as reaching the target position
RETURNED = 0.25        # progress at or below this counts as back at the start
START_LEAVE = 0.35     # progress needed to consider a rep "started"
PARTIAL_MIN = 0.5      # minimum excursion to report a partial rep
EMA_ALPHA = 0.7        # smoothing for the raw angle (higher = less lag)
MIN_KP_SCORE = 0.3     # keypoints below this confidence are not trusted
SIDE_MARGIN = 0.15     # how much better the other side must be before switching


@dataclass
class RepState:
    count: int = 0
    good_count: int = 0
    moving: bool = False
    reached_target: bool = False
    peak: float = 0.0
    ema_angle: float | None = None
    side: str = "left"
    rep_issues: list[str] = field(default_factory=list)  # posture codes seen during current rep

    @classmethod
    def from_dict(cls, d: dict | None) -> "RepState":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepUpdate:
    angle: float | None
    progress: float
    completed: bool = False
    partial: bool = False
    partial_peak: float = 0.0
    good_form: bool = False
    tracked: bool = True  # False when the joints weren't visible enough this frame


def _mirror(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def side_points(spec: ExerciseSpec, side: str) -> tuple[str, str, str]:
    pts = tuple(spec.angle_points)
    return pts if side == "left" else tuple(_mirror(p) for p in pts)


def _side_score(kp: dict[str, Keypoint], names) -> float:
    return min((kp[n].score if n in kp else 0.0) for n in names)


def pick_side(spec: ExerciseSpec, kp: dict[str, Keypoint], current: str = "left") -> tuple[str, float]:
    """Side whose three keypoints are most confidently detected (sticky, to avoid flip-flopping)."""
    other = "right" if current == "left" else "left"
    cur_score = _side_score(kp, side_points(spec, current))
    other_score = _side_score(kp, side_points(spec, other))
    if other_score > cur_score + SIDE_MARGIN or (cur_score < MIN_KP_SCORE <= other_score):
        return other, other_score
    return current, cur_score


def measure(spec: ExerciseSpec, kp: dict[str, Keypoint], side: str = "left") -> float:
    a, b, c = side_points(spec, side)
    return angle(kp[a], kp[b], kp[c])


def progress_for(spec: ExerciseSpec, value: float) -> float:
    span = max(spec.high_angle - spec.low_angle, 1e-6)
    if spec.start_phase == "high":
        return (spec.high_angle - value) / span
    return (value - spec.low_angle) / span


def update(spec: ExerciseSpec, state: RepState, kp: dict[str, Keypoint]) -> RepUpdate:
    side, score = pick_side(spec, kp, state.side)
    if score < MIN_KP_SCORE:
        # Joints not visible enough: hold the state rather than feed garbage in.
        q = progress_for(spec, state.ema_angle) if state.ema_angle is not None else 0.0
        return RepUpdate(angle=state.ema_angle, progress=max(0.0, min(1.0, q)), tracked=False)

    raw = measure(spec, kp, side)
    if side != state.side:
        state.side, state.ema_angle = side, raw  # don't blend angles from two different limbs
    state.ema_angle = raw if state.ema_angle is None else EMA_ALPHA * raw + (1 - EMA_ALPHA) * state.ema_angle
    q = progress_for(spec, state.ema_angle)
    out = RepUpdate(angle=state.ema_angle, progress=max(0.0, min(1.0, q)))

    if not state.moving:
        if q > START_LEAVE:
            state.moving, state.peak, state.reached_target = True, q, q >= TARGET_REACHED
        return out

    state.peak = max(state.peak, q)
    if q >= TARGET_REACHED:
        state.reached_target = True
    if q <= RETURNED:  # back at the start position
        if state.reached_target:
            state.count += 1
            out.completed = True
            out.good_form = not state.rep_issues
            if out.good_form:
                state.good_count += 1
        elif state.peak >= PARTIAL_MIN:
            out.partial, out.partial_peak = True, min(state.peak, 1.0)
        state.moving, state.reached_target, state.peak = False, False, 0.0
        state.rep_issues = []
    return out
