"""
Process 5 — repetition counting.

Tracks one joint angle per exercise (ExerciseSpec.angle_points) and maps it to
`progress` in [0, 1] (0 = start position, 1 = target position). A rep is a
full excursion start -> target -> start. Excursions that reach >= 40 % but
not the target are reported as partial reps (not counted).
"""

from dataclasses import asdict, dataclass, field

from .geometry import angle
from .types import ExerciseSpec, Keypoint

START_LEAVE = 0.2      # progress needed to consider a rep "started"
PARTIAL_MIN = 0.4      # minimum excursion to report a partial rep
EMA_ALPHA = 0.6        # smoothing for the raw angle


@dataclass
class RepState:
    count: int = 0
    good_count: int = 0
    moving: bool = False
    reached_target: bool = False
    peak: float = 0.0
    ema_angle: float | None = None
    rep_issues: list[str] = field(default_factory=list)  # posture codes seen during current rep

    @classmethod
    def from_dict(cls, d: dict | None) -> "RepState":
        return cls(**d) if d else cls()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepUpdate:
    angle: float
    progress: float
    completed: bool = False
    partial: bool = False
    partial_peak: float = 0.0
    good_form: bool = False


def measure(spec: ExerciseSpec, kp: dict[str, Keypoint]) -> float:
    a, b, c = spec.angle_points
    return angle(kp[a], kp[b], kp[c])


def progress_for(spec: ExerciseSpec, value: float) -> float:
    span = max(spec.high_angle - spec.low_angle, 1e-6)
    if spec.start_phase == "high":
        return (spec.high_angle - value) / span
    return (value - spec.low_angle) / span


def update(spec: ExerciseSpec, state: RepState, kp: dict[str, Keypoint]) -> RepUpdate:
    raw = measure(spec, kp)
    state.ema_angle = raw if state.ema_angle is None else EMA_ALPHA * raw + (1 - EMA_ALPHA) * state.ema_angle
    q = progress_for(spec, state.ema_angle)
    out = RepUpdate(angle=state.ema_angle, progress=max(0.0, min(1.0, q)))

    if not state.moving:
        if q > START_LEAVE:
            state.moving, state.peak, state.reached_target = True, q, q >= 1.0
        return out

    state.peak = max(state.peak, q)
    if q >= 1.0:
        state.reached_target = True
    if q <= 0.0:  # back at the start position
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
