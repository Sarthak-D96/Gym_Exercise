"""
4.3 Classification — ML exercise recognition.

`ExerciseClassifier` is the interface (class diagram: predict() / confidence()).
Input is a sliding window of feature vectors (see features.py), newest last,
so both per-frame models (MLP, random forest) and sequence models
(LSTM / 1D-CNN / transformer) fit the same signature.

TODO(model): implement the trained classifier as a subclass, load weights in
`load()` from `self.model_path`, and register it in `model_store.CLASSIFIER_BACKENDS`.
"""

from __future__ import annotations

import random


class ExerciseClassifier:
    def __init__(self, classes: list[str], model_path: str | None = None):
        self.classes = list(classes)
        self.model_path = model_path
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def predict_proba(self, feature_window: list[list[float]], *, context: dict) -> dict[str, float]:
        """Return a probability for every class (values sum to 1)."""
        raise NotImplementedError

    def predict(self, feature_window: list[list[float]], *, context: dict) -> tuple[str | None, float]:
        probs = self.predict_proba(feature_window, context=context)
        if not probs:
            return None, 0.0
        label = max(probs, key=probs.get)
        return label, probs[label]

    @staticmethod
    def confidence(probs: dict[str, float], label: str) -> float:
        return probs.get(label, 0.0)


class DummyExerciseClassifier(ExerciseClassifier):
    """
    Placeholder classifier — does not run any model.

    It "cheats" by reading the exercise the DummyPoseDetector simulated
    (`context["pose_meta"]["simulated_exercise"]`) and returns it with a random
    confidence. About 10 % of frames return a wrong label with low confidence,
    so the Validate & Refine stage (refine.py) has noise to smooth out.
    """

    def predict_proba(self, feature_window, *, context):
        if not self.classes or not feature_window:
            return {}
        rng = random.Random()
        truth = (context.get("pose_meta") or {}).get("simulated_exercise")
        if truth not in self.classes or rng.random() < 0.1:
            truth = rng.choice(self.classes)
            top = rng.uniform(0.35, 0.55)
        else:
            top = rng.uniform(0.78, 0.97)
        others = [c for c in self.classes if c != truth]
        weights = [rng.random() for _ in others]
        total = sum(weights) or 1.0
        probs = {c: (1 - top) * w / total for c, w in zip(others, weights)}
        probs[truth] = top
        return probs
