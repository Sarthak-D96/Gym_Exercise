"""Tests for the glue around the trained models (the models themselves are tested in training/)."""

import importlib.util
import io
import tempfile

from django.test import SimpleTestCase
from PIL import Image

from recognition.pipeline.features import FEATURE_SIZE
from recognition.pipeline.preprocess import preprocess_frame
from recognition.pipeline.trained_models import WINDOW, SklearnClassifier, _pad

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None and importlib.util.find_spec("joblib") is not None


def jpeg(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "JPEG")
    return buf.getvalue()


class PreprocessContentBoxTests(SimpleTestCase):
    def test_landscape_frame_box(self):
        img = preprocess_frame(jpeg(320, 240), 256)
        self.assertEqual(img.size, (256, 256))
        self.assertEqual(img.info["content_box"], (0, 32, 256, 192))


class PadTests(SimpleTestCase):
    def test_short_window_repeats_last_frame(self):
        w = _pad([[1.0], [2.0]])
        self.assertEqual(len(w), WINDOW)
        self.assertEqual(w[-1], [2.0])

    def test_long_window_keeps_newest(self):
        w = _pad([[float(i)] for i in range(20)])
        self.assertEqual(w[0], [4.0])


class SklearnClassifierTests(SimpleTestCase):
    def setUp(self):
        if not HAS_SKLEARN:
            self.skipTest("scikit-learn not installed")

    def test_probabilities_map_to_class_names(self):
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, FEATURE_SIZE * 5))
        y = np.arange(40) % 2
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".joblib") as f:
            joblib.dump({"model": model, "classes": ["push_up", "squat"]}, f.name)
            clf = SklearnClassifier(classes=["push_up", "squat"], model_path=f.name)
            clf.load()
            probs = clf.predict_proba([[0.1] * FEATURE_SIZE] * 3, context={})
        self.assertEqual(set(probs), {"push_up", "squat"})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
