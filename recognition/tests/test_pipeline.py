"""Unit tests for the recognition pipeline (test plan: unit testing — rep counter, pose processing)."""

import base64
import io

from django.test import SimpleTestCase
from PIL import Image

from recognition.pipeline import rep_counter
from recognition.pipeline.classifier import DummyExerciseClassifier
from recognition.pipeline.engine import RecognitionEngine
from recognition.pipeline.features import FEATURE_SIZE, extract_features
from recognition.pipeline.geometry import angle
from recognition.pipeline.pose_detector import DummyPoseDetector
from recognition.pipeline.preprocess import FrameDecodeError, decode_data_url, preprocess_frame
from recognition.pipeline.refine import refine
from recognition.pipeline.types import ExerciseSpec, Keypoint

SQUAT = ExerciseSpec("squat", "Squat", ("left_hip", "left_knee", "left_ankle"), 100, 160, "high")
PRESS = ExerciseSpec("shoulder_press", "Shoulder Press", ("left_shoulder", "left_elbow", "left_wrist"), 100, 155, "low")


def leg(knee_angle):
    """Keypoints whose hip-knee-ankle angle equals `knee_angle`."""
    import math
    a = math.radians(knee_angle)
    knee = Keypoint(0.5, 0.5)
    ankle = Keypoint(0.5, 0.7)
    # ankle points straight down from the knee; rotate by `knee_angle` to place the hip
    hip = Keypoint(0.5 + 0.2 * math.sin(a), 0.5 + 0.2 * math.cos(a))
    return {"left_hip": hip, "left_knee": knee, "left_ankle": ankle}


class GeometryTests(SimpleTestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(angle(Keypoint(0, 1), Keypoint(0, 0), Keypoint(1, 0)), 90, places=4)

    def test_straight_line(self):
        self.assertAlmostEqual(angle(Keypoint(0, 0), Keypoint(0, 1), Keypoint(0, 2)), 180, places=4)

    def test_leg_helper(self):
        for a in (60, 90, 170):
            self.assertAlmostEqual(angle(*leg(a).values()), a, places=3)


class RepCounterTests(SimpleTestCase):
    def run_angles(self, spec, angles):
        state = rep_counter.RepState()
        updates = []
        for a in angles:
            kp = leg(a) if spec is SQUAT else {
                "left_shoulder": leg(a)["left_hip"], "left_elbow": leg(a)["left_knee"], "left_wrist": leg(a)["left_ankle"],
            }
            # feed each angle several times so the EMA settles
            for _ in range(4):
                updates.append(rep_counter.update(spec, state, kp))
        return state, updates

    def test_counts_full_reps(self):
        state, _ = self.run_angles(SQUAT, [170, 130, 90, 130, 170] * 3)
        self.assertEqual(state.count, 3)
        self.assertEqual(state.good_count, 3)

    def test_partial_rep_not_counted(self):
        state, updates = self.run_angles(SQUAT, [170, 125, 115, 125, 170])
        self.assertEqual(state.count, 0)
        self.assertTrue(any(u.partial for u in updates))

    def test_rep_with_issue_is_not_good_form(self):
        state = rep_counter.RepState()
        for a in (170, 130):
            for _ in range(4):
                rep_counter.update(SQUAT, state, leg(a))
        state.rep_issues.append("chest_up")
        for a in (90, 130, 170):
            for _ in range(4):
                rep_counter.update(SQUAT, state, leg(a))
        self.assertEqual((state.count, state.good_count), (1, 0))

    def test_low_start_exercise(self):
        state, _ = self.run_angles(PRESS, [85, 130, 170, 130, 85] * 2)
        self.assertEqual(state.count, 2)

    def test_sparse_noisy_camera_input(self):
        """One sample per frame at ~5 fps, squat only to parallel, no full lockout."""
        import math
        import random
        rnd, state = random.Random(0), rep_counter.RepState()
        for i in range(105):
            t = i / 5
            a = 158 - 53 * (1 - math.cos(2 * math.pi * min(t / 2, 10))) / 2 + rnd.gauss(0, 4)
            rep_counter.update(SQUAT, state, leg(a))
        self.assertEqual(state.count, 10)

    def test_uses_side_facing_camera(self):
        state = rep_counter.RepState()
        for a in [170, 130, 90, 130, 170] * 2:
            right = {k.replace("left_", "right_"): v for k, v in leg(a).items()}
            hidden = {k: Keypoint(v.x, v.y, 0.1) for k, v in leg(175).items()}  # occluded left leg
            for _ in range(2):
                rep_counter.update(SQUAT, state, {**hidden, **right})
        self.assertEqual(state.count, 2)

    def test_invisible_joints_are_skipped(self):
        state = rep_counter.RepState()
        blind = {k: Keypoint(v.x, v.y, 0.05) for k, v in leg(90).items()}
        for _ in range(10):
            self.assertFalse(rep_counter.update(SQUAT, state, blind).tracked)
        self.assertEqual((state.count, state.moving), (0, False))

    def test_half_reps_and_standing_still_not_counted(self):
        state, _ = self.run_angles(SQUAT, [170, 150, 135, 150, 170] * 3 + [168, 172] * 5)
        self.assertEqual(state.count, 0)

    def test_old_saved_state_still_loads(self):
        s = rep_counter.RepState.from_dict({"count": 2, "moving": False, "unknown_old_key": 1})
        self.assertEqual(s.count, 2)

    def test_state_roundtrip(self):
        s = rep_counter.RepState(count=4, moving=True, rep_issues=["x"])
        self.assertEqual(rep_counter.RepState.from_dict(s.to_dict()), s)


class _ScriptedClassifier:
    """Returns the labels of a script, one per frame."""
    def __init__(self, labels):
        self.labels, self.i = labels, 0

    def predict(self, window, context=None):
        label = self.labels[min(self.i, len(self.labels) - 1)]
        self.i += 1
        return label, 0.95


class _ScriptedPose:
    def __init__(self, angles):
        self.angles, self.i = angles, 0

    def detect_pose(self, image, timestamp_ms=0, context=None):
        from recognition.pipeline.types import PoseResult
        a = self.angles[min(self.i, len(self.angles) - 1)]
        self.i += 1
        from recognition.pipeline.types import KEYPOINT_NAMES
        kp = {n: Keypoint(0.5, 0.3, 0.2) for n in KEYPOINT_NAMES}  # rest of the body: low confidence
        kp.update(leg(a))
        return PoseResult(keypoints=kp, score=0.9)


class ExerciseSwitchTests(SimpleTestCase):
    SPECS = {"squat": SQUAT, "lunge": ExerciseSpec("lunge", "Lunge", ("left_hip", "left_knee", "left_ankle"), 105, 160, "high")}
    ANGLES = [170, 140, 100, 140, 170] * 6

    def run_engine(self, labels, context=None):
        engine = RecognitionEngine(self.SPECS, _ScriptedPose(self.ANGLES), _ScriptedClassifier(labels),
                                   smoothing_window=1, min_confidence=0.5)
        state, result = {}, None
        for t in range(len(self.ANGLES)):
            result, state = engine.process(None, timestamp_ms=t * 200, state=state, context=context or {})
        return result

    def test_brief_misclassification_does_not_reset_count(self):
        labels = ["squat"] * 12 + ["lunge"] * 2 + ["squat"] * 16  # 2-frame glitch
        result = self.run_engine(labels)
        self.assertEqual((result.exercise, result.reps), ("squat", 6))

    def test_target_exercise_is_always_counted(self):
        labels = ["squat", "lunge"] * 15  # classifier can't decide
        result = self.run_engine(labels, {"target_exercise": "squat"})
        self.assertEqual((result.exercise, result.reps), ("squat", 6))

    def test_sustained_change_switches_exercise(self):
        labels = ["squat"] * 10 + ["lunge"] * 20
        result = self.run_engine(labels)
        self.assertEqual(result.exercise, "lunge")
        self.assertGreaterEqual(result.reps, 2)


class RefineTests(SimpleTestCase):
    def test_majority_vote_smooths_noise(self):
        hist = []
        for label in ["squat"] * 6 + ["lunge"]:
            out, conf, hist = refine(hist, label, 0.9, window=8, min_confidence=0.6)
        self.assertEqual(out, "squat")

    def test_low_confidence_is_uncertain(self):
        out, conf, _ = refine([], "squat", 0.3, window=8, min_confidence=0.6)
        self.assertIsNone(out)


class PreprocessTests(SimpleTestCase):
    def test_decode_and_resize(self):
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), "red").save(buf, "JPEG")
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        img = preprocess_frame(decode_data_url(url), 256)
        self.assertEqual(img.size, (256, 256))

    def test_rejects_garbage(self):
        with self.assertRaises(FrameDecodeError):
            decode_data_url("data:image/jpeg;base64,@@@")
        with self.assertRaises(FrameDecodeError):
            preprocess_frame(b"not an image")


class EngineTests(SimpleTestCase):
    def test_simulated_workout_counts_reps(self):
        specs = {"squat": SQUAT}
        engine = RecognitionEngine(specs, DummyPoseDetector(), DummyExerciseClassifier(["squat"]))
        state, result = {}, None
        for t in range(0, 15000, 120):
            result, state = engine.process(None, timestamp_ms=t, state=state,
                                           context={"simulate_exercise": "squat", "seed": 1})
        self.assertEqual(result.exercise, "squat")
        self.assertGreaterEqual(result.reps, 3)
        self.assertEqual(len(extract_features(DummyPoseDetector().detect_pose(
            None, timestamp_ms=0, context={}).keypoints)), FEATURE_SIZE)


class ExtraFeatureTests(SimpleTestCase):
    def test_extra_features_shape_and_tilt(self):
        import numpy as np

        from recognition.pipeline.extra_features import EXTENDED_SIZE, extend
        from recognition.pipeline.types import KEYPOINT_NAMES

        def pose(lying):
            kp = {n: Keypoint(0.5, 0.5, 0.9) for n in KEYPOINT_NAMES}
            for side, dx in (("left", -0.02), ("right", 0.02)):
                kp[f"{side}_hip"] = Keypoint(0.5 + dx, 0.6)
                kp[f"{side}_shoulder"] = Keypoint(0.3, 0.6 + dx) if lying else Keypoint(0.5 + dx, 0.3)
            return extract_features(kp)

        up, flat = extend(np.array([pose(False), pose(True)]))
        self.assertEqual(up.shape, (EXTENDED_SIZE,))
        self.assertAlmostEqual(float(up[FEATURE_SIZE]), 0.0, places=2)   # torso upright
        self.assertAlmostEqual(float(flat[FEATURE_SIZE]), 0.5, places=2)  # torso horizontal
