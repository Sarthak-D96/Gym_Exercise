"""
Real (trained) models for stages 4.2 and 4.3.

    YoloPoseDetector       YOLO-pose (ultralytics), the same model training/extract_keypoints.py used
    TorchScriptClassifier  LSTM / GRU / 1D-CNN exported by training/train_dl.py   (*.pt)
    SklearnClassifier      Random Forest / XGBoost saved by training/train_ml.py  (*.joblib)
    TwoStageClassifier     router + per-group specialists from training/select_two_stage.py (two_stage.json)

They are registered in model_store.py and chosen through the MLModel table
(Django admin, or `python manage.py register_models`). Heavy libraries are
imported inside load(), so the site still runs without them while the dummy
models are active.
"""

from __future__ import annotations

import json
from pathlib import Path

from .classifier import ExerciseClassifier
from .pose_detector import PoseDetector
from .types import KEYPOINT_NAMES, Keypoint, PoseResult

WINDOW = 16  # frames per window the classifiers were trained on


def _pad(window: list[list[float]]) -> list[list[float]]:
    """Use the newest 16 frames; repeat the last one if fewer (as build_dataset.py does for short clips)."""
    window = window[-WINDOW:]
    return window + [window[-1]] * (WINDOW - len(window))


def _inputs(window: list[list[float]], features: str):
    """(16, 42) array, or (16, 55) with the two-stage extra features."""
    import numpy as np

    X = np.asarray(_pad(window), dtype=np.float32)
    if features == "extended":
        from .extra_features import extend

        X = extend(X)
    return X


class YoloPoseDetector(PoseDetector):
    """COCO-17 keypoints from an ultralytics YOLO-pose model, e.g. ml_models/yolo11n-pose.pt."""

    def load(self) -> None:
        from ultralytics import YOLO

        path = self.model_path or "yolo11n-pose.pt"
        if not Path(path).exists():
            raise FileNotFoundError(f"Pose model not found: {path}. Run `python manage.py register_models`.")
        self.model = YOLO(path)
        self.loaded = True

    def detect_pose(self, frame, *, timestamp_ms, context):
        if frame is None:  # simulation mode sends no image; a real detector has nothing to look at
            return PoseResult(keypoints={}, score=0.0)
        result = self.model.predict(frame, verbose=False)[0]
        kps = result.keypoints
        if kps is None or kps.conf is None or len(kps.xy) == 0:
            return PoseResult(keypoints={}, score=0.0)

        # biggest person in the frame (same rule as training)
        boxes = result.boxes.xywh.cpu().numpy()
        i = int((boxes[:, 2] * boxes[:, 3]).argmax())
        xy = kps.xy[i].cpu().numpy()      # pixels in the letterboxed square
        conf = kps.conf[i].cpu().numpy()

        left, top, w, h = frame.info.get("content_box", (0, 0, frame.width, frame.height))
        keypoints = {
            name: Keypoint(x=float((xy[j, 0] - left) / w), y=float((xy[j, 1] - top) / h), score=float(conf[j]))
            for j, name in enumerate(KEYPOINT_NAMES)
        }
        return PoseResult(keypoints=keypoints, score=float(conf.mean()))


class TorchScriptClassifier(ExerciseClassifier):
    """TorchScript sequence model: input (1, 16, 42) features, output logits in `classes` order."""

    features = "base"  # "extended" = 42 + 13 extra features (two-stage models)

    def load(self) -> None:
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.net = torch.jit.load(self.model_path, map_location="cpu").eval()
        self.loaded = True

    def predict_proba(self, feature_window, *, context):
        if not feature_window:
            return {}
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(_inputs(feature_window, self.features)[None])
            probs = self.net(x).softmax(-1)[0].tolist()
        return dict(zip(self.classes, probs))


class SklearnClassifier(ExerciseClassifier):
    """Random Forest / XGBoost on the 210 window statistics (mean, std, min, max, range)."""

    features = "base"

    def load(self) -> None:
        import joblib
        import numpy as np

        self.np = np
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"] if isinstance(bundle, dict) else bundle
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1  # one window per call: thread start-up would cost more than it saves
        self.loaded = True

    def predict_proba(self, feature_window, *, context):
        if not feature_window:
            return {}
        np = self.np
        X = _inputs(feature_window, self.features)
        mn, mx = X.min(0), X.max(0)
        stats = np.concatenate([X.mean(0), X.std(0), mn, mx, mx - mn])[None]
        probs = self.model.predict_proba(stats)[0]
        # model.classes_ are label indices 0..K-1 into the training classes.json
        return {self.classes[int(c)]: float(p) for c, p in zip(self.model.classes_, probs)}


class TwoStageClassifier(ExerciseClassifier):
    """
    Router + specialists (training/select_two_stage.py). `model_path` is the bundle's two_stage.json.

    P(exercise) = P(group | window) x P(exercise | group, window). Specialists of
    groups the router gives less than SKIP_BELOW are not run (saves time); their
    exercises share the group's small probability equally.
    """

    SKIP_BELOW = 0.05
    BACKENDS = {"torchscript": TorchScriptClassifier, "sklearn": SklearnClassifier}

    def load(self) -> None:
        path = Path(self.model_path)
        manifest = json.loads(path.read_text())
        features = manifest.get("features", "extended")

        def make(entry):
            if not entry or not entry.get("file"):
                return None
            clf = self.BACKENDS[entry["backend"]](classes=entry["classes"], model_path=str(path.parent / entry["file"]))
            clf.features = features
            clf.load()
            return clf

        self.router = make(manifest.get("router"))
        self.groups = {g: (e["classes"], make(e)) for g, e in manifest["groups"].items()}
        self.loaded = True

    def predict_proba(self, feature_window, *, context):
        if not feature_window:
            return {}
        if self.router is not None:
            p_group = self.router.predict_proba(feature_window, context=context)
        else:
            p_group = {g: 1.0 for g in self.groups}
        out: dict[str, float] = {}
        for g, (members, model) in self.groups.items():
            pg = p_group.get(g, 0.0)
            if model is None or pg < self.SKIP_BELOW:
                out.update({c: pg / len(members) for c in members})
            else:
                sub = model.predict_proba(feature_window, context=context)
                out.update({c: pg * sub.get(c, 0.0) for c in members})
        return {c: out.get(c, 0.0) for c in self.classes}
