"""
Orchestrates one frame through the recognition pipeline.

    frame ─► 4.1 preprocess ─► 4.2 pose + features ─► 4.3 classify ─► 4.4 refine
                                                   └─► 5 rep counter ─► 6/7 posture feedback

The engine is stateless; everything that must survive between frames is in
the JSON-serialisable `state` dict, which the caller persists (on the
WorkoutSession row) and passes back on the next frame.
"""

from __future__ import annotations

import logging

from . import posture, rep_counter
from .classifier import ExerciseClassifier
from .features import extract_features
from .pose_detector import PoseDetector
from .preprocess import preprocess_frame
from .refine import refine
from .types import ExerciseSpec, Feedback, FrameResult

logger = logging.getLogger("recognition")

FEATURE_WINDOW = 16
SWITCH_FRAMES = 6  # frames a new exercise label must hold before the rep counter switches to it


class RecognitionEngine:
    def __init__(
        self,
        specs: dict[str, ExerciseSpec],
        pose_detector: PoseDetector,
        classifier: ExerciseClassifier,
        *,
        input_size: int = 256,
        smoothing_window: int = 8,
        min_confidence: float = 0.6,
        count_only_good_form: bool = True,
    ):
        self.count_only_good_form = count_only_good_form
        self.specs = specs
        self.pose_detector = pose_detector
        self.classifier = classifier
        self.input_size = input_size
        self.smoothing_window = smoothing_window
        self.min_confidence = min_confidence

    def process(
        self,
        frame_bytes: bytes | None,
        *,
        timestamp_ms: float,
        state: dict,
        context: dict | None = None,
    ) -> tuple[FrameResult, dict]:
        context = dict(context or {})
        state = dict(state or {})

        # 4.1 Preprocess
        image = preprocess_frame(frame_bytes, self.input_size) if frame_bytes else None

        # 4.2 Pose / keypoints
        pose = self.pose_detector.detect_pose(image, timestamp_ms=timestamp_ms, context=context)
        simulated = bool(pose.meta.get("dummy"))
        if not pose.detected:
            return self._no_pose(state, simulated), state

        features = extract_features(pose.keypoints)
        window = (state.get("feature_window", []) + [features])[-FEATURE_WINDOW:]
        state["feature_window"] = window

        # 4.3 Classification
        raw_label, raw_conf = self.classifier.predict(window, context={**context, "pose_meta": pose.meta})

        # 4.4 Validate & refine
        label, conf, history = refine(
            state.get("prediction_history", []), raw_label, raw_conf,
            window=self.smoothing_window, min_confidence=self.min_confidence,
        )
        state["prediction_history"] = history

        # Decide which exercise the rep counter follows. Switching resets the
        # count, so it must not happen on a brief misclassification:
        #  * if the user picked a target exercise, always count that one;
        #  * in auto-detect mode, switch only after the new label has held for
        #    SWITCH_FRAMES frames in a row and no rep is in progress.
        current = state.get("current_exercise")
        target = context.get("target_exercise")
        candidate = None
        if target in self.specs:
            candidate = target if target != current else None
            state.pop("pending_exercise", None)
        elif label and label != current and label in self.specs:
            pending = state.get("pending_exercise") or {}
            n = pending.get("frames", 0) + 1 if pending.get("label") == label else 1
            state["pending_exercise"] = {"label": label, "frames": n}
            mid_rep = (state.get("rep") or {}).get("moving", False)
            if current not in self.specs or (n >= SWITCH_FRAMES and not mid_rep):
                candidate = label
        else:
            state.pop("pending_exercise", None)

        changed = False
        if candidate:
            current, changed = candidate, True
            state["current_exercise"] = candidate
            state["rep"] = rep_counter.RepState().to_dict()
            state.pop("pending_exercise", None)

        events: list[Feedback] = []
        live: list[Feedback] = []
        reps = good = rejected = 0
        points = 0.0
        completed = False
        angle = None
        progress = 0.0
        last_rep = None

        if current in self.specs:
            spec = self.specs[current]
            rs = rep_counter.RepState.from_dict(state.get("rep"))

            # 5 Repetition counting + 6/7 form grading of each finished rep
            upd = rep_counter.update(spec, rs, pose.keypoints, timestamp_ms, strict=self.count_only_good_form)
            angle, progress, completed = upd.angle, upd.progress, upd.completed

            # live cues while moving (or always, for holds like the plank)
            if upd.moving or posture.is_static(spec):
                for issue in posture.live_issues(spec, upd.frame):
                    live.append(issue.feedback())
                    if issue.code not in rs.live_codes:  # log each problem once per rep
                        rs.live_codes.append(issue.code)
                        events.append(issue.feedback())

            g = upd.grade
            if g is not None:
                last_rep = {"score": g.score, "severity": g.severity, "counted": upd.completed}
                if upd.completed and g.severity == "good":
                    events.append(Feedback("good", "good_rep", f"Rep {rs.count}: good form! ({g.score}/100)"))
                elif upd.completed:
                    tips = " ".join(i.message for i in g.issues[:2])
                    events.append(Feedback("info", "rep_with_issues", f"Rep {rs.count} counted ({g.score}/100). {tips}"))
                else:
                    fixes = " ".join(i.message for i in g.issues if i.severity == "major")
                    what = "Partial rep" if upd.partial else "Rep"
                    events.append(Feedback("warning", "rep_rejected", f"{what} not counted. {fixes}"))
                    live = [i.feedback() for i in g.issues[:2]] + live
            reps, good, rejected, points = rs.count, rs.good_count, rs.rejected, rs.points
            state["rep"] = rs.to_dict()

        if target and label and label != target:
            live.append(Feedback("info", "mismatch",
                                 f"Detected {self.specs[label].name} but you selected {self.specs[target].name}."
                                 if target in self.specs and label in self.specs else "Detected a different exercise."))

        result = FrameResult(
            pose_detected=True,
            keypoints={k: v.to_dict() for k, v in pose.keypoints.items()},
            exercise=current if current in self.specs else None,
            confidence=conf if label == current else 0.0,
            raw_exercise=raw_label,
            raw_confidence=raw_conf,
            exercise_changed=changed,
            reps=reps,
            good_reps=good,
            rep_completed=completed,
            angle=angle,
            progress=progress,
            live_feedback=live,
            events=events,
            simulated=simulated,
            rejected_reps=rejected,
            form_points=points,
            last_rep=last_rep,
        )
        return result, state

    def _no_pose(self, state: dict, simulated: bool) -> FrameResult:
        rs = rep_counter.RepState.from_dict(state.get("rep"))
        return FrameResult(
            pose_detected=False, keypoints={}, exercise=state.get("current_exercise"), confidence=0.0,
            raw_exercise=None, raw_confidence=0.0, exercise_changed=False, reps=rs.count,
            good_reps=rs.good_count, rep_completed=False, angle=None, progress=0.0,
            live_feedback=[Feedback("info", "no_pose", "Step into the frame so your whole body is visible.")],
            events=[], simulated=simulated, rejected_reps=rs.rejected, form_points=rs.points,
        )
