# Model Storage (DFD store D2)

Trained model files go in this folder. Files here are ignored by git; only metadata is
tracked, in the `recognition.MLModel` table (Django admin → Recognition → ML models).

## Plugging in the trained models

The code for the real models is already in `recognition/pipeline/trained_models.py`:

| Stage | Backend name | Class | File |
|---|---|---|---|
| 4.2 Pose detection | `yolo_pose` | `YoloPoseDetector` | `yolo11n-pose.pt` (same weights as training) |
| 4.3 Classification | `torchscript` | `TorchScriptClassifier` | `lstm.pt` / `gru.pt` / `cnn.pt` from `training/train_dl.py` |
| 4.3 Classification | `sklearn` | `SklearnClassifier` | `rf.joblib` / `xgb.joblib` from `training/train_ml.py` |

**Easiest:** train with `training/run_all.py` (or `training/kaggle_train.ipynb`), then run

```bash
pip install -r requirements.txt
python manage.py install_model ~/Downloads/gymrec_model_<id>.zip
```

It unpacks the zip into `ml_models/<id>/`, adds any missing exercises, activates the classifier
and its pose weights, stores the scores, and shows only the trained exercises. Installing an older
zip again switches back to that model.

Manual steps (for a single model file):

1. Copy into this folder from the training run (Kaggle → notebook → Output, `gym/runs/`):
   * the chosen classifier, e.g. `lstm.pt`
   * `classes.json`
   * optionally `yolo11n-pose.pt` (downloaded automatically if missing)
2. Install the model libraries: `pip install -r requirements.txt`
3. Register and activate them:
   ```bash
   python manage.py migrate                      # adds the bench_press and pull_up exercises
   python manage.py register_models --classifier lstm.pt --sync-exercises
   ```
   `--sync-exercises` shows only the exercises the model knows on the website.
   Add `--metrics lstm_test.json` to store the test scores, and `--model-version 1.1` for a newer model.
4. In `.env`: `RECOGNITION_INPUT_SIZE=640` and `RECOGNITION_ALLOW_SIMULATION=false`, then restart the server.

To switch back to the placeholders, activate the `dummy` rows in Django admin.

## Input/output contracts

* **Pose detector input:** a square RGB `PIL.Image` (letterboxed to `RECOGNITION_INPUT_SIZE`), whose
  `info["content_box"]` says where the real picture sits, or `None` in simulation mode.
  **Output:** 17 COCO keypoints (`nose`, `left_shoulder`, …, `right_ankle`), with coordinates
  normalised to 0..1 relative to the *original* frame.
* **Classifier input:** up to 16 most recent feature vectors, oldest first (padded to 16 by
  repeating the newest). Each vector has 42 floats: 17 keypoints × (x, y), normalised to the hip
  centre and torso length, followed by 8 joint angles / 180. (See `recognition/pipeline/features.py`.)
  **Output:** a probability per class slug. Slugs must match `workouts.Exercise.slug`.
