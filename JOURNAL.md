# Project Journal — Gym Exercise Recognition System

UCS503 Software Engineering Project · TIET, Patiala
Team: Sarthak Duvedi (1024031013), Samar Goyal (1024030992)

This journal records the project phase by phase: what was planned, what was done, the decisions made
along the way, and the results. Phase 2 is complete. Phases 1, 3 and 4 are written as plans. Fields marked
`☐ TBD` are to be filled in once each phase is carried out.

| # | Phase | Status | Dates |
|---|---|---|---|
| 1 | Dataset exploration | Planned | ☐ TBD |
| 2 | Website making | **Done** (prototype) | 2026-10-09 → 2026-11-20 |
| 3 | Data preprocessing | Planned | ☐ TBD |
| 4 | Model training | Planned | ☐ TBD |

---

## Phase 1 — Dataset exploration

**Goal:** understand the dataset well enough to choose the exercise classes, the train/validation/test split
and the preprocessing settings.
**Status:** Planned · **Start:** ☐ TBD · **End:** ☐ TBD

### Dataset source (external, not created by the team)

| | |
|---|---|
| Name | *Gym Workout/Exercises Video* |
| Author | vm7608 (Kaggle user `philosopher0808`) |
| Link | https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video |
| Built on | Hasyim Abdillah, *Workout/Exercises Video* — https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video |
| Licence | Apache 2.0 |
| Size | ≈ 10.4 GB, `.mp4` videos, one folder per exercise |
| Last updated | 2024-05-12 (version 1) |

According to the dataset's own description, it has three parts:

| Part | Contents | Videos |
|---|---|---|
| `raw_data/data-btc` | Videos from Abdillah's dataset | 652 |
| `raw_data/data-crawl` | Videos crawled from YouTube | 334 |
| `verified_data/data_btc_10s` | Cleaned and cut into clips of ≤ ~10–13 s | 817 |
| `verified_data/data_crawl_10s` | Cleaned and cut into clips of ≤ ~10–13 s | 754 |
| `test` | Kept noisy on purpose, for evaluation | 61 |

It covers 22 exercise classes: barbell biceps curl, bench press, chest fly machine, deadlift, decline bench
press, hammer curl, hip thrust, incline bench press, lat pulldown, lateral raise, leg extension, leg raises,
plank, pull up, push-up, romanian deadlift, russian twist, shoulder press, squat, t bar row, tricep dips,
tricep pushdown.

### Planned checks

- [ ] Count videos per class in each part (`verified_data`, `test`), and check for class imbalance.
- [ ] Record resolution, frame rate, duration and orientation (portrait or landscape) for every video, and save them to a manifest CSV.
- [ ] Find corrupt or unreadable files, and duplicates (the same clip in both `data_btc` and `data_crawl`).
- [ ] Watch a sample of about 5 videos per class. Note the camera angle (front, side, overhead), whether the whole body is visible, how many people are in frame, and any text overlays.
- [ ] Run a pose detector on the sample. Note which classes are detected badly, for example lying-down exercises like bench press, or people partly hidden by machines.
- [ ] Check whether the person can be identified per video. This is needed so no person ends up in more than one split.

### Decision to make: which exercises to support

The website's prototype uses 6 exercises. Only some of them appear in this dataset:

| Website exercise | In dataset? | Matching dataset class |
|---|---|---|
| Squat | Yes | `squat` |
| Push-up | Yes | `push-up` |
| Shoulder press | Yes | `shoulder press` |
| Bicep curl | Yes | `barbell biceps curl` (and maybe `hammer curl`) |
| Lunge | **No** | — |
| Jumping jack | **No** | — |

There are two options:
- **(a)** Change the website's exercise list to match a subset of the dataset's classes. Exercises can be added in the admin panel without code changes.
- **(b)** Find extra data for lunge and jumping jack.

Prefer exercises where the whole body is visible to a camera.

**Decision:** ☐ TBD — final class list: ☐ TBD

### Findings (fill in)

| Item | Value |
|---|---|
| Videos per class (min / median / max) | ☐ TBD |
| Typical resolution and frame rate | ☐ TBD |
| Typical clip length | ☐ TBD |
| Corrupt or duplicate files removed | ☐ TBD |
| Classes where pose detection works poorly | ☐ TBD |
| Can people be identified per video? | ☐ TBD |

**Figures to include:** class distribution bar chart, histograms of resolution and frame rate,
histogram of clip length, and sample frames per class with the detected skeleton drawn on top.

---

## Phase 2 — Website making

**Goal:** build a working prototype of the system described in the UCS503 presentation (use cases, DFDs, class,
sequence, state and component diagrams), with space left for the ML model.
**Status:** Done · **Start:** 2026-10-09 · **Code committed:** 2026-11-20

### Requirements taken from the presentation

FR1–FR8: register and log in; start and stop a workout; capture camera video; detect pose; recognise the
exercise; count reps; give posture feedback; store and show workout history.
User stories US-01 to US-06, including the admin usage report.
Test cases TC01–TC06.

### Technology choices

| Layer | Choice | Reason |
|---|---|---|
| Backend | Django | Required. Built-in login, admin panel and database layer. |
| Database | PostgreSQL | Required. JSON fields hold the per-session pipeline state. |
| Frontend | Django templates, plain JavaScript, Chart.js | No separate build step. The browser camera API (`getUserMedia`) captures frames. |
| ML pipeline | A plain Python package with no Django code inside it | Can be unit-tested on its own, or moved to a separate ML server later (as in the deployment diagram). |

### What was built

- **Apps:**
  - `accounts`: registration, login, profile.
  - `workouts`: exercises, sessions, records, feedback, pages and JSON API.
  - `recognition`: model metadata, inference logs and the pipeline.
  - `reports`: admin usage report.
- **Database tables:**
  - `User` and `Profile`: data store D1 in the DFD.
  - `MLModel`: data store D2.
  - `WorkoutSession`, `WorkoutRecord`, `FeedbackEvent`: data store D3.
  - `Exercise`: the exercise catalogue, including each exercise's rep-counting settings.
  - `InferenceLog`: per-frame speed and usage data.
- **Recognition pipeline**, following DFD level 2:
  1. Preprocess frames (4.1).
  2. Detect pose (4.2). **Placeholder.**
  3. Extract features: 42 numbers per frame (17 keypoints plus 8 joint angles).
  4. Classify the exercise (4.3). **Placeholder.**
  5. Validate and refine (4.4): majority vote over the last 8 frames, plus a minimum confidence.
  6. Count reps from a joint angle (process 5).
  7. Check posture with rules (processes 6/7).
- **State chart:** each workout goes Ready → Active ⇄ Paused → Completed, enforced by `WorkoutSession.transition()`.
- **Pages:**
  - Login and register.
  - Dashboard: stats and an exercise picker, including auto-detect.
  - Live workout: camera view with skeleton overlay, exercise and confidence, rep counter, posture feedback, activity log.
  - History: charts plus a list of workouts.
  - Workout detail: per-exercise table and feedback timeline.
  - Admin reports (staff only).
  - Django admin.
- **Simulation mode:** the live page still works without a camera, so the whole flow can be demoed anywhere.
- **Demo data:** the `seed_demo` command creates a demo user, an admin user and about 4 weeks of sample history, for screenshots.
- **Tests:** 22 automated tests. They cover the rep counter, smoothing, preprocessing, the state chart, and full workflows matching TC01–TC06.

### Space left for the model

`DummyPoseDetector` and `DummyExerciseClassifier` stand in for the real models. They don't look at the image;
they return a simulated skeleton and label. The real models plug in by subclassing `PoseDetector` and
`ExerciseClassifier`, registering them in `recognition/pipeline/model_store.py`, and activating them in the admin
panel. The full steps and the input/output contract are in `ml_models/README.md`.

### Problems found and fixed

| Problem | Cause | Fix |
|---|---|---|
| Live-page timer showed `NaN:NaN` | Session data was turned into JSON twice before reaching the page | Pass the data to `json_script` once |
| Pause/Resume buttons showed while hidden | The button CSS overrode the `hidden` attribute | Added a global `[hidden] { display: none !important }` rule |
| Saving a frame crashed with `NotSupportedError` | PostgreSQL can't lock rows across an outer join | Lock only the session row (`select_for_update(of=("self",))`) |
| One bad simulated rep was flagged twice | The smoothed angle lags past the end of a rep, so the fault leaked into the next rep | The simulated fault now grows and shrinks with the movement |
| Push-up "hips in line" warning fired on good reps | Threshold too close to the normal body-line angle | Lowered the threshold from 165° to 160° |

### Verification

- All 22 tests pass against PostgreSQL.
- A full workout was run in the browser: log in → start a squat workout → reps counted, posture warnings shown, skeleton overlay drawn → end workout → workout detail, history charts and admin reports all checked.
- **Not yet tested:** the real-camera path. The test browser blocks camera access, so only simulation mode was used.

### Deliverables

The codebase (see `README.md`), the demo data command, and screenshots for slide 4.3: Login, Dashboard,
Live Workout, History.

---

## Phase 3 — Data preprocessing

**Goal:** turn the videos into keypoint sequences in the same format the website's classifier will receive.
**Status:** Planned · **Start:** ☐ TBD · **End:** ☐ TBD

### Planned steps

1. **Choose the input data:** use `verified_data` for training and validation. Keep the dataset's own `test` folder untouched for the final evaluation.
2. **Split by person or video, never by frame:** about 70 / 15 / 15. Clips cut from the same source video must stay in the same split. Save the split to `splits.json` so every experiment uses the same one.
3. **Extract keypoints:**
   - Sample each video at the same frame rate the website sends frames (about 6 fps). If the website's frame rate changes, re-extract.
   - Run the pose detector on each sampled frame and save 17 keypoints (COCO layout) with confidence scores, one `.npy` file per video.
   - Use the same pose model here that will run in the website.
4. **Clean the keypoints:**
   - Drop frames with average confidence below about 0.3.
   - Fill short gaps (one or two missing frames) by interpolating.
   - Drop videos that are mostly missing.
   - Trim idle frames at the start and end.
5. **Turn frames into features:** 42 numbers per frame, using the website's own `recognition/pipeline/features.py`. This guarantees training and the live app compute features the same way.
6. **Cut windows:** 16 frames per window, starting a new window every 4 frames. Output arrays `X: (N, 16, 42)`, `y: (N,)`, plus a person/video id per window.
7. **Augment (training split only):**
   - Mirror left and right, swapping the left_ and right_ keypoint names.
   - Add small random noise to the keypoints.
   - Scale the skeleton slightly and rotate it by about ±10°.
   - Stretch or squeeze time by about ±20%.
8. **Check the result:** draw the skeleton back onto about 10 videos to confirm the keypoints are right. Report how many windows each class has.

### Settings (fill in final values)

| Setting | Planned | Final |
|---|---|---|
| Pose model | YOLO-pose (COCO-17) | ☐ TBD |
| Sampling rate | 6 fps | ☐ TBD |
| Window length / step | 16 / 4 frames | ☐ TBD |
| Minimum keypoint confidence | 0.3 | ☐ TBD |
| Train / val / test split | 70 / 15 / 15 by person or video | ☐ TBD |

### Results (fill in)

| Item | Value |
|---|---|
| Videos processed / dropped | ☐ TBD |
| Windows per split (train / val / test) | ☐ TBD |
| Windows per class | ☐ TBD |
| Time taken to extract keypoints | ☐ TBD |

**Outputs:** `keypoints/*.npy`, `splits.json`, `train.npz`, `val.npz`, `test.npz`, `classes.json`.

---

## Phase 4 — Model training

**Goal:** train and compare classical ML and deep learning models, then put the best one into the website.
**Status:** Planned · **Start:** ☐ TBD · **End:** ☐ TBD

### Planned experiments

| Round | Model | Input |
|---|---|---|
| 1 | Random Forest | 5 statistics per feature per window (mean, std, min, max, range): about 210 numbers |
| 2 | XGBoost | Same as round 1 |
| 3 | LSTM / GRU (2 layers, 64–128 units) | The raw `(16, 42)` sequences |
| 4 | 1D-CNN / TCN | The raw `(16, 42)` sequences |
| 5 (optional) | ST-GCN, or a video model (VideoMAE / X3D) on raw clips | Comparison for the report |

**Training setup:**
- Adam optimiser with learning rate 1e-3, batch size 64.
- Early stopping on validation F1.
- Class weights if the classes are imbalanced.
- Tune only on the validation set; the test set is used once, at the end.

### Evaluation plan

- **Scores** on the test set, which contains only unseen people: accuracy, F1 per class, macro F1, and a confusion matrix.
- **Speed:** how long the model takes per window on the target machine.
- **Live check:** replay full test videos through the website's actual pipeline, including the smoothing step. Measure how long it takes to recognise the exercise, and how often the label flickers.

### Results (fill in)

| Model | Accuracy | Macro F1 | Time per window | Size |
|---|---|---|---|---|
| Random Forest | ☐ | ☐ | ☐ | ☐ |
| XGBoost | ☐ | ☐ | ☐ | ☐ |
| LSTM / GRU | ☐ | ☐ | ☐ | ☐ |
| 1D-CNN / TCN | ☐ | ☐ | ☐ | ☐ |

Classes most often confused: ☐ TBD
Model chosen and why: ☐ TBD

### Putting it into the website

- [ ] Export the chosen model (TorchScript or `.joblib`) to `ml_models/`, together with `classes.json`.
- [ ] Write subclasses of `PoseDetector` (YOLO-pose) and `ExerciseClassifier` (the trained model), and register them in `model_store.py`.
- [ ] Add both models in the admin panel with their scores, and activate them.
- [ ] Update the exercise catalogue to the final class list from Phase 1.
- [ ] Turn off simulation mode (`RECOGNITION_ALLOW_SIMULATION=false`) and test with a real webcam.
- [ ] Update the tests, and rerun TC01–TC06.

---

## References

1. vm7608, *Gym Workout/Exercises Video*, Kaggle, 2024. https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video (Apache 2.0)
2. H. Abdillah, *Workout/Exercises Video*, Kaggle. https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video
3. UCS503 project presentation: *Gym Exercise Recognition System* (team document).
