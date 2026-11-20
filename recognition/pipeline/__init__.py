"""
Exercise-recognition pipeline (DFD level 2, process 4.0 + processes 5-7).

    4.1 preprocess.py      decode + resize camera frames
    4.2 pose_detector.py   body keypoints            <-- DUMMY (plug real model here)
        features.py        keypoints -> feature vector
    4.3 classifier.py      exercise classification   <-- DUMMY (plug real model here)
    4.4 refine.py          smoothing / validation of predictions
    5   rep_counter.py     repetition counting (joint-angle state machine)
    6/7 posture.py         posture analysis + feedback
        engine.py          orchestrates one frame through all stages
        model_store.py     loads active models from Model Storage (D2)

Everything in this package is framework-agnostic (no Django imports except in
model_store.py) so it can be unit-tested or moved to a separate ML service.
"""

from .engine import RecognitionEngine  # noqa: F401
from .types import ExerciseSpec, FrameResult  # noqa: F401
