# Model Storage (DFD store D2)

Trained model files go in this folder. Files here are ignored by git; only metadata is
tracked, in the `recognition.MLModel` table (Django admin → Recognition → ML models).

## Plugging in the real models

The prototype runs with **placeholder** models:

| Stage | Interface | Placeholder | File |
|---|---|---|---|
| 4.2 Pose detection | `PoseDetector.detect_pose(frame, timestamp_ms, context) -> PoseResult` | `DummyPoseDetector` (synthetic skeleton) | `recognition/pipeline/pose_detector.py` |
| 4.3 Classification | `ExerciseClassifier.predict_proba(feature_window, context) -> {label: prob}` | `DummyExerciseClassifier` | `recognition/pipeline/classifier.py` |

Steps:

1. Put the weights here, e.g. `ml_models/classifier_v1.pt`.
2. Subclass the interface and load the weights in `load()` (it is called once per process):
   ```python
   class LSTMExerciseClassifier(ExerciseClassifier):
       def load(self):
           self.net = torch.jit.load(self.model_path).eval()
           self.loaded = True

       def predict_proba(self, feature_window, *, context):
           x = torch.tensor([feature_window], dtype=torch.float32)  # (1, T, 42)
           probs = self.net(x).softmax(-1)[0].tolist()
           return dict(zip(self.classes, probs))
   ```
3. Register it in `recognition/pipeline/model_store.py` (`CLASSIFIER_BACKENDS["lstm"] = LSTMExerciseClassifier`).
4. In Django admin, add an ML model row: kind = classifier, name = `lstm`, version = `1.0`,
   model path = `classifier_v1.pt`, classes = the label order the network was trained with,
   **is_dummy unchecked**. Then use the "Activate" action.

Input/output contracts:

* **Pose detector input:** a 256×256 RGB `PIL.Image` (letterboxed), or `None` in simulation mode.
  **Output:** 17 COCO keypoints (`nose`, `left_shoulder`, …, `right_ankle`), with coordinates
  normalised to 0..1 relative to the *original* frame.
* **Classifier input:** up to 16 most recent feature vectors, oldest first. Each vector has 42 floats:
  17 keypoints × (x, y), normalised to the hip centre and torso length, followed by 8 joint angles / 180.
  (See `recognition/pipeline/features.py`.)
  **Output:** a probability per class slug. Slugs must match `workouts.Exercise.slug`.
