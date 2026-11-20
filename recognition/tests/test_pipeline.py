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

    def test_state_roundtrip(self):
        s = rep_counter.RepState(count=4, moving=True, rep_issues=["x"])
        self.assertEqual(rep_counter.RepState.from_dict(s.to_dict()), s)


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
