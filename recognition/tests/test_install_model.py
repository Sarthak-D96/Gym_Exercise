"""`manage.py install_model` with a model zip shaped like training/select_best.py's output."""

import importlib.util
import io
import json
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from recognition.models import MLModel
from recognition.pipeline import model_store
from workouts.models import Exercise

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("ultralytics") is not None
CLASSES = ["lateral_raise", "pull_up", "squat"]
LATERAL = {"name": "Lateral Raise", "icon": "🙆", "muscle_group": "Shoulders", "met_value": 3.5,
           "angle_points": ["left_hip", "left_shoulder", "left_elbow"], "low_angle": 30, "high_angle": 75,
           "start_phase": "low"}


def make_bundle(root: Path, model_id="cnn-20261125-1200") -> Path:
    import torch
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(42, len(CLASSES))

        def forward(self, x):
            return self.fc(x.mean(1))

    folder = root / f"gymrec_model_{model_id}"
    folder.mkdir()
    torch.jit.trace(Tiny().eval(), torch.zeros(1, 16, 42)).save(str(folder / "classifier.pt"))
    (folder / "yolo11n-pose.pt").write_bytes(b"fake weights")
    (folder / "classes.json").write_text(json.dumps(CLASSES))
    (folder / "metrics_test.json").write_text(json.dumps({"accuracy": 0.91, "macro_f1": 0.9, "split": "test",
                                                          "model": "cnn", "size_mb": 0.1}))
    (folder / "model_card.json").write_text(json.dumps({
        "id": model_id,
        "classifier": {"model": "cnn", "name": "1D-CNN / TCN", "file": "classifier.pt", "backend": "torchscript"},
        "pose": {"file": "yolo11n-pose.pt", "name": "yolo11n-pose.pt"},
        "classes": CLASSES,
        "exercises": {"lateral_raise": LATERAL},
        "preprocessing": {"imgsz": settings.RECOGNITION["INPUT_SIZE"], "window": 16},
        "metrics": {"accuracy": 0.91, "macro_f1": 0.9, "split": "test"},
        "library_versions": {},
    }))
    zip_path = root / f"{folder.name}.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in folder.iterdir():
            z.write(f, f"{folder.name}/{f.name}")
    return zip_path


def make_two_stage_bundle(root: Path, model_id="two_stage-20261126-1000") -> Path:
    """Router (TorchScript) -> groups 'arms' (lateral_raise only, no model) and 'body' (sklearn specialist)."""
    import joblib
    import numpy as np
    import torch
    from sklearn.tree import DecisionTreeClassifier
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fc = nn.Linear(55, n)

        def forward(self, x):
            return self.fc(x.mean(1))

    folder = root / f"gymrec_model_{model_id}"
    folder.mkdir()
    torch.jit.trace(Tiny(2).eval(), torch.zeros(1, 16, 55)).save(str(folder / "router.pt"))
    tree = DecisionTreeClassifier().fit(np.random.default_rng(0).random((20, 275)), [0, 1] * 10)
    joblib.dump({"model": tree}, folder / "body.joblib")
    (folder / "two_stage.json").write_text(json.dumps({
        "features": "extended",
        "router": {"file": "router.pt", "model": "cnn", "backend": "torchscript", "classes": ["arms", "body"]},
        "groups": {"arms": {"file": None, "model": None, "backend": None, "classes": ["lateral_raise"]},
                   "body": {"file": "body.joblib", "model": "rf", "backend": "sklearn",
                            "classes": ["pull_up", "squat"]}},
    }))
    (folder / "yolo11n-pose.pt").write_bytes(b"fake weights")
    (folder / "classes.json").write_text(json.dumps(CLASSES))
    (folder / "metrics_test.json").write_text(json.dumps({"accuracy": 0.8, "macro_f1": 0.75, "split": "test",
                                                          "model": "two_stage"}))
    (folder / "model_card.json").write_text(json.dumps({
        "id": model_id,
        "classifier": {"model": "two_stage", "name": "Two-stage (router + specialists)", "file": "two_stage.json",
                       "backend": "two_stage", "backends": ["sklearn", "torchscript"], "models": ["cnn", "rf"]},
        "stages": {"router": {"model": "cnn", "classes": ["arms", "body"]},
                   "body": {"model": "rf", "classes": ["pull_up", "squat"]}},
        "pose": {"file": "yolo11n-pose.pt", "name": "yolo11n-pose.pt"},
        "classes": CLASSES,
        "exercises": {"lateral_raise": LATERAL},
        "preprocessing": {"imgsz": settings.RECOGNITION["INPUT_SIZE"], "window": 16},
        "metrics": {"accuracy": 0.8, "macro_f1": 0.75, "split": "test"},
        "library_versions": {},
    }))
    zip_path = root / f"{folder.name}.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in folder.iterdir():
            z.write(f, f"{folder.name}/{f.name}")
    return zip_path


class InstallModelTests(TestCase):
    def setUp(self):
        if not HAS_TORCH:
            self.skipTest("torch / ultralytics not installed")
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "ml_models").mkdir()
        self.src = root / "downloads"
        self.src.mkdir()
        cfg = {**settings.RECOGNITION, "MODEL_DIR": root / "ml_models"}
        self.settings_override = override_settings(RECOGNITION=cfg)
        self.settings_override.enable()
        # the fake pose weights can't be loaded by ultralytics
        self.pose_patch = mock.patch("recognition.pipeline.trained_models.YoloPoseDetector.load")
        self.pose_patch.start()
        model_store.clear_cache()

    def tearDown(self):
        self.pose_patch.stop()
        self.settings_override.disable()
        self.tmp.cleanup()
        model_store.clear_cache()

    def test_install_activates_model_and_syncs_exercises(self):
        zip_path = make_bundle(self.src)
        call_command("install_model", str(zip_path), stdout=io.StringIO())

        clf = MLModel.objects.get(kind="classifier", is_active=True)
        self.assertFalse(clf.is_dummy)
        self.assertEqual(clf.name, "torchscript")
        self.assertEqual(clf.version, "cnn-20261125-1200")
        self.assertEqual(clf.model_path, "gymrec_model_cnn-20261125-1200/classifier.pt")
        self.assertEqual(clf.classes, CLASSES)
        self.assertEqual(clf.metrics["accuracy"], 0.91)
        self.assertEqual(MLModel.objects.get(kind="pose", is_active=True).name, "yolo_pose")

        # new class added from the card, only trained classes shown
        self.assertEqual(Exercise.objects.get(slug="lateral_raise").low_angle, 30)
        self.assertEqual(set(Exercise.objects.filter(is_active=True).values_list("slug", flat=True)), set(CLASSES))

        # the website now predicts with it
        engine = model_store.build_engine()
        probs = engine.classifier.predict_proba([[0.1] * 42] * 5, context={})
        self.assertEqual(set(probs), set(CLASSES))

    def test_reinstalling_older_model_switches_back(self):
        old = make_bundle(self.src, "cnn-20261101-0900")
        (self.src / "b").mkdir()
        new = make_bundle(self.src / "b", "lstm-20261125-1300")
        call_command("install_model", str(old), stdout=io.StringIO())
        call_command("install_model", str(new), stdout=io.StringIO())
        call_command("install_model", str(old), stdout=io.StringIO())
        self.assertEqual(MLModel.objects.get(kind="classifier", is_active=True).version, "cnn-20261101-0900")

    def test_rejects_unsafe_zip(self):
        bad = self.src / "bad.zip"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("../evil.txt", "x")
        with self.assertRaises(CommandError):
            call_command("install_model", str(bad), stdout=io.StringIO())

    def test_install_two_stage_bundle(self):
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        zip_path = make_two_stage_bundle(self.src)
        call_command("install_model", str(zip_path), stdout=io.StringIO())

        clf = MLModel.objects.get(kind="classifier", is_active=True)
        self.assertEqual(clf.name, "two_stage")
        self.assertEqual(clf.model_path, "gymrec_model_two_stage-20261126-1000/two_stage.json")

        engine = model_store.build_engine()
        probs = engine.classifier.predict_proba([[0.1] * 42] * 5, context={})
        self.assertEqual(set(probs), set(CLASSES))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
        label, conf = engine.classifier.predict([[0.1] * 42] * 16, context={})
        self.assertIn(label, CLASSES)


class InstallFormProfilesTests(TestCase):
    def test_installs_profile_and_calibrated_angles(self):
        squat = Exercise.objects.get(slug="squat")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "form_profiles.json"
            path.write_text(json.dumps({"created": "2026-11-26T10:00:00", "exercises": {
                "squat": {"reps": 120, "videos": 30,
                          "angles": {"applied": True, "low": 90, "high": 170},
                          "metrics": {"torso_lean": {"p5": 8, "p25": 14, "p50": 17, "p75": 20, "p95": 26, "n": 120}},
                          "test": {"attempts": 40, "good": 30, "minor": 6}},
                "not_on_site": {"reps": 5, "metrics": {}},
            }}))
            out = io.StringIO()
            call_command("install_form_profiles", str(path), stdout=out)
        squat.refresh_from_db()
        self.assertEqual((squat.low_angle, squat.high_angle), (90, 170))
        self.assertEqual(squat.form_profile["metrics"]["torso_lean"]["p50"], 17)
        self.assertIn("held-out reps counted 36/40", out.getvalue())

        spec = model_store.build_engine().specs["squat"]
        from recognition.pipeline import posture
        lean = posture.EXTRA_CHECKS["squat"][0]
        self.assertIsNone(posture.grade_value(lean, spec, 30))          # fence 20+1.5*6=29, +5 tol = 34
        self.assertEqual(posture.grade_value(lean, spec, 40).severity, "minor")
        self.assertEqual(posture.grade_value(lean, spec, 50).severity, "major")

    def test_keep_angles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps({"exercises": {"squat": {"angles": {"applied": True, "low": 90, "high": 170},
                                                                 "metrics": {}}}}))
            call_command("install_form_profiles", str(path), keep_angles=True, stdout=io.StringIO())
        self.assertEqual(Exercise.objects.get(slug="squat").low_angle, 100)


class EnsembleStageTests(TestCase):
    def test_two_stage_with_ensemble_stage(self):
        if not HAS_TORCH or importlib.util.find_spec("sklearn") is None:
            self.skipTest("torch / scikit-learn not installed")
        import joblib
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier

        from recognition.pipeline.trained_models import TwoStageClassifier

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(make_two_stage_bundle(Path(tmp))).with_suffix("")
            rng = np.random.default_rng(1)
            joblib.dump({"model": DecisionTreeClassifier(random_state=1).fit(rng.random((20, 275)), [0, 1] * 10)},
                        folder / "body_xgb.joblib")
            manifest = json.loads((folder / "two_stage.json").read_text())
            body = manifest["groups"]["body"]
            body["ensemble"] = [{"file": "body.joblib", "model": "rf", "backend": "sklearn"},
                                {"file": "body_xgb.joblib", "model": "xgb", "backend": "sklearn"}]
            (folder / "two_stage.json").write_text(json.dumps(manifest))

            clf = TwoStageClassifier(classes=CLASSES, model_path=str(folder / "two_stage.json"))
            clf.load()
            window = [[0.1] * 42] * 16
            probs = clf.predict_proba(window, context={})
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
            # the body stage is the average of its two members
            members = clf.groups["body"][1].members
            sub = [m.predict_proba(window, context={}) for m in members]
            p_body = clf.router.predict_proba(window, context={})["body"]
            self.assertAlmostEqual(probs["squat"], p_body * (sub[0]["squat"] + sub[1]["squat"]) / 2, places=6)
