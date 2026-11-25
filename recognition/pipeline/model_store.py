"""
Model Storage (DFD store D2) — loads the active model for each pipeline stage.

The `recognition.MLModel` table says which implementation/version is active;
this module maps that to a Python class and caches loaded instances per process.

To plug in a real model:
  1. Put the weights under settings.RECOGNITION["MODEL_DIR"].
  2. Implement a subclass of PoseDetector / ExerciseClassifier.
  3. Add it to POSE_BACKENDS / CLASSIFIER_BACKENDS below.
  4. Create an MLModel row (Django admin) with is_dummy=False, name=<backend key>,
     and activate it.
"""

from __future__ import annotations

import threading

from django.conf import settings

from .classifier import DummyExerciseClassifier, ExerciseClassifier
from .engine import RecognitionEngine
from .pose_detector import DummyPoseDetector, PoseDetector
from .trained_models import SklearnClassifier, TorchScriptClassifier, YoloPoseDetector
from .types import ExerciseSpec

POSE_BACKENDS: dict[str, type[PoseDetector]] = {
    "dummy": DummyPoseDetector,
    "yolo_pose": YoloPoseDetector,
}
CLASSIFIER_BACKENDS: dict[str, type[ExerciseClassifier]] = {
    "dummy": DummyExerciseClassifier,
    "torchscript": TorchScriptClassifier,   # LSTM / GRU / 1D-CNN (.pt from training/train_dl.py)
    "sklearn": SklearnClassifier,           # Random Forest / XGBoost (.joblib from training/train_ml.py)
}

_lock = threading.Lock()
_cache: dict[tuple, object] = {}


def _active(kind: str):
    from recognition.models import MLModel

    return MLModel.objects.filter(kind=kind, is_active=True).first()


def _model_path(row) -> str | None:
    if row and row.model_path:
        return str(settings.RECOGNITION["MODEL_DIR"] / row.model_path)
    return None


def get_pose_detector() -> PoseDetector:
    row = _active("pose")
    backend = "dummy" if (row is None or row.is_dummy) else row.name
    key = ("pose", backend, row.version if row else None)
    with _lock:
        if key not in _cache:
            det = POSE_BACKENDS.get(backend, DummyPoseDetector)(model_path=_model_path(row))
            det.load()
            _cache[key] = det
        return _cache[key]


def get_classifier(classes: list[str]) -> ExerciseClassifier:
    row = _active("classifier")
    backend = "dummy" if (row is None or row.is_dummy) else row.name
    key = ("classifier", backend, row.version if row else None, tuple(classes))
    with _lock:
        if key not in _cache:
            clf = CLASSIFIER_BACKENDS.get(backend, DummyExerciseClassifier)(
                classes=(row.classes if row and row.classes else classes), model_path=_model_path(row)
            )
            clf.load()
            _cache[key] = clf
        return _cache[key]


def clear_cache():
    with _lock:
        _cache.clear()


def build_engine() -> RecognitionEngine:
    from workouts.models import Exercise

    specs = {
        e.slug: ExerciseSpec(
            slug=e.slug, name=e.name, angle_points=tuple(e.angle_points),
            low_angle=e.low_angle, high_angle=e.high_angle, start_phase=e.start_phase,
        )
        for e in Exercise.objects.filter(is_active=True)
    }
    cfg = settings.RECOGNITION
    return RecognitionEngine(
        specs,
        get_pose_detector(),
        get_classifier(sorted(specs)),
        input_size=cfg["INPUT_SIZE"],
        smoothing_window=cfg["SMOOTHING_WINDOW"],
        min_confidence=cfg["MIN_CONFIDENCE"],
    )
