from dataclasses import dataclass, field

# COCO-17 keypoint layout (same names MoveNet / YOLO-pose use; MediaPipe's 33
# landmarks are a superset and can be mapped onto these).
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]


@dataclass
class Keypoint:
    x: float  # normalised 0..1, image width
    y: float  # normalised 0..1, image height (y grows downwards)
    score: float = 1.0

    def to_dict(self):
        return {"x": round(self.x, 4), "y": round(self.y, 4), "score": round(self.score, 3)}


@dataclass
class PoseResult:
    keypoints: dict[str, Keypoint]
    score: float
    # Free-form extra info from the detector. The dummy detector stores the
    # exercise it simulated here; real detectors can leave it empty.
    meta: dict = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return self.score >= 0.3 and bool(self.keypoints)


@dataclass
class ExerciseSpec:
    """Rep-counter / posture configuration for one exercise (mirrors workouts.Exercise)."""
    slug: str
    name: str
    angle_points: tuple[str, str, str]
    low_angle: float
    high_angle: float
    start_phase: str = "high"  # "high" = starts extended, "low" = starts flexed


@dataclass
class Feedback:
    level: str  # "good" | "info" | "warning"
    code: str
    message: str

    def to_dict(self):
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class FrameResult:
    pose_detected: bool
    keypoints: dict[str, dict]
    exercise: str | None          # smoothed slug, None if uncertain
    confidence: float             # smoothed confidence of `exercise`
    raw_exercise: str | None      # un-smoothed classifier output for this frame
    raw_confidence: float
    exercise_changed: bool
    reps: int                     # reps for the current exercise block
    good_reps: int
    rep_completed: bool
    angle: float | None
    progress: float               # 0 = start position, 1 = target position
    live_feedback: list[Feedback] # issues visible in this frame
    events: list[Feedback]        # things worth persisting (rep done, partial rep, new issue)
    simulated: bool

    def to_dict(self):
        return {
            "pose_detected": self.pose_detected,
            "keypoints": self.keypoints,
            "exercise": self.exercise,
            "confidence": round(self.confidence, 3),
            "raw_exercise": self.raw_exercise,
            "raw_confidence": round(self.raw_confidence, 3),
            "exercise_changed": self.exercise_changed,
            "reps": self.reps,
            "good_reps": self.good_reps,
            "rep_completed": self.rep_completed,
            "angle": None if self.angle is None else round(self.angle, 1),
            "progress": round(self.progress, 3),
            "live_feedback": [f.to_dict() for f in self.live_feedback],
            "events": [f.to_dict() for f in self.events],
            "simulated": self.simulated,
        }
