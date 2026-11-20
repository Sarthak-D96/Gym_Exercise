# GymRec — Gym Exercise Recognition System (prototype)

UCS503 Software Engineering project prototype: a web app that recognises the exercise you're doing
from the camera, counts repetitions, gives posture feedback, and stores your workout history.

**Stack:** Django (backend and server-rendered UI), PostgreSQL (database), vanilla JS with Chart.js (frontend).

> **ML status:** the **pose detector** and the **exercise classifier** are *placeholders*
> (`DummyPoseDetector`, `DummyExerciseClassifier`). They return a simulated skeleton and label, so
> the whole flow can be demoed without trained models. The rest of the pipeline is real code that
> works on whatever keypoints the detector returns: frame preprocessing, feature extraction,
> prediction smoothing, rep counting and posture rules. See [`ml_models/README.md`](ml_models/README.md)
> to plug in real models.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # optional, the defaults work with a local PostgreSQL
createdb gymrec                   # or: docker compose up -d  (then set POSTGRES_USER/PASSWORD=gymrec)
python manage.py migrate          # creates tables and seeds 6 exercises and model metadata
python manage.py seed_demo        # optional: demo + admin users and 4 weeks of sample history
python manage.py runserver
```

Open http://localhost:8000. You can register a new account, or use the demo accounts that
`seed_demo` prints (defined in `workouts/management/commands/seed_demo.py`).
Staff users also see **Reports** and **/admin**.

**Camera vs simulation:** the live page uses your webcam if you allow it. If not, it falls back to
**Simulation** mode, where frames are sent with no image and the dummy detector generates the skeleton.
You can switch between the two at any time.

Run the tests:

```bash
python manage.py test
```

## How it maps to the design documents

| PPT artefact | Where in code |
|---|---|
| Use cases UC-01 *Recognize Exercise*, UC-02 *View History* | `workouts/views.py` → `live_workout`, `api_session_frame`, `history`, `api_history` |
| DFD L1 stores D1 / D2 / D3 | `accounts.Profile` and `auth.User` / `recognition.MLModel` and `ml_models/` / `workouts.WorkoutSession`, `WorkoutRecord`, `FeedbackEvent` |
| DFD L2: 4.1 → 4.4 | `recognition/pipeline/`: `preprocess.py` → `pose_detector.py` and `features.py` → `classifier.py` → `refine.py` |
| Processes 5–7: reps, posture, feedback | `recognition/pipeline/rep_counter.py`, `posture.py` |
| Class diagram (User, GymSystem, MLModel, Database, Camera, PoseDetector, ExerciseClassifier, WorkoutRecord) | `auth.User`/`Profile`, `workouts/services.py`, `recognition.MLModel` and `model_store.py`, Django ORM, browser `getUserMedia` (`static/js/live.js`), `PoseDetector`, `ExerciseClassifier`, `WorkoutRecord` |
| Sequence / collaboration diagram | `static/js/live.js` → `POST /api/sessions/<id>/frames/` → `services.process_frame` → `RecognitionEngine.process` → DB |
| State chart (Ready → Active ⇄ Paused → Completed) | `WorkoutSession.transition()` |
| Component diagram | UI = `templates/`, `static/`; Workout Controller = `workouts/services.py`; Video Processing and ML Recognition = `recognition/pipeline/`; Database Service = Django ORM / PostgreSQL; Model Storage = `MLModel` and `ml_models/` |
| US-06 Admin usage reports | `reports/` (staff only) and Django admin |
| Test cases TC01–TC06 | `workouts/tests/test_workflow.py`, `recognition/tests/test_pipeline.py` |

## Project layout

```
gymrec/                 Django settings / root URLs
accounts/               registration, login, profile (D1)
workouts/               exercises, sessions, records, feedback, pages and JSON API (D3)
  services.py           Workout Controller: runs a frame through the engine, persists the results
  management/commands/seed_demo.py
recognition/            ML side
  models.py             MLModel (D2 metadata), InferenceLog (latency/usage telemetry)
  pipeline/             framework-agnostic pipeline (see pipeline/__init__.py)
reports/                admin usage & performance report
templates/, static/     UI (Dashboard, Live workout, History, Session detail, Reports)
ml_models/              place trained weights here
```

## API (JSON, session-authenticated, CSRF header required for POST)

| Method | URL | Purpose |
|---|---|---|
| GET | `/api/exercises/` | exercise catalogue |
| GET | `/api/sessions/<id>/` | session summary |
| POST | `/api/sessions/<id>/begin` · `pause` · `resume` · `end` | state-chart transitions (409 if not allowed) |
| POST | `/api/sessions/<id>/frames/` | body `{"frame": "<jpeg data URL or null>", "t": <active ms>}` returns exercise, confidence, reps, keypoints, feedback |
| GET | `/api/history/?days=30` | daily reps / form score and per-exercise totals |
| GET | `/reports/data/` | (staff) sessions per day, inference latency |

## Adding an exercise

Add one in Django admin. You don't need to change any code. Set the three keypoints whose angle
should be tracked, the low and high thresholds, and whether the movement starts extended (`high`)
or flexed (`low`). Then add the slug to the classifier's classes. Posture rules for new exercises
go in `recognition/pipeline/posture.py`.
