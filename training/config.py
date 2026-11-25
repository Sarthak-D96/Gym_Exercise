"""
Training configuration — the one file to edit (Phases 3–4 in JOURNAL.md).

`run_all.py` reads everything from here. In a notebook you can also override any
value before running, e.g.

    import config
    config.CLASSES = ["squat", "push_up", "pull_up"]
    import run_all; run_all.main()
"""

# ---------------------------------------------------------------------------
# 1. Which exercises to train on
# ---------------------------------------------------------------------------
# Slugs from EXERCISES below. Add or remove lines to change what the model learns.
# Only these classes are extracted, trained on, and shown on the website after
# `python manage.py install_model`.
CLASSES = [
    "squat",
    "push_up",
    "shoulder_press",
    "bicep_curl",
    "bench_press",
    "pull_up",
    # "lateral_raise",
    # "deadlift",
    # "hammer_curl",
]

# ---------------------------------------------------------------------------
# 2. Exercise catalogue
# ---------------------------------------------------------------------------
# For each class:
#   folders   dataset folder names that belong to this class. Case, spaces and
#             dashes are ignored ("Push-Up" == "push up" == "push_up"). Several
#             folders are merged into one class. The slug itself always matches.
#   website   settings copied into the website's exercise catalogue when the model
#             is installed: the joint angle used to count reps (three COCO
#             keypoints, angle measured at the middle one), the flexed (low) and
#             extended (high) thresholds in degrees, and which end a rep starts at.
#             Thresholds marked ROUGH are first guesses; tune them in Django admin
#             (Workouts → Exercises) after watching real reps.
def _ex(name, folders, angle_points, low, high, start="high", icon="🏋️", muscles="", met=5.0):
    return {
        "folders": folders,
        "website": {
            "name": name, "icon": icon, "muscle_group": muscles, "met_value": met,
            "angle_points": list(angle_points), "low_angle": low, "high_angle": high, "start_phase": start,
        },
    }


ARM = ("left_shoulder", "left_elbow", "left_wrist")      # elbow angle
KNEE = ("left_hip", "left_knee", "left_ankle")           # knee angle
HIP = ("left_shoulder", "left_hip", "left_knee")         # hip angle
SHOULDER = ("left_hip", "left_shoulder", "left_elbow")   # arm raised away from the body

EXERCISES = {
    # --- already on the website ---
    "squat": _ex("Squat", ["squat"], KNEE, 100, 160, icon="🏋️", muscles="Quads, glutes"),
    "push_up": _ex("Push-up", ["push-up", "push up", "pushup"], ARM, 95, 155, icon="💪",
                   muscles="Chest, triceps", met=8.0),
    "shoulder_press": _ex("Shoulder Press", ["shoulder press", "overhead press"], ARM, 100, 155, start="low",
                          icon="🙌", muscles="Shoulders", met=4.0),
    "bicep_curl": _ex("Bicep Curl", ["barbell biceps curl", "bicep curl", "biceps curl", "dumbbell curl"],
                      ARM, 60, 150, icon="🦾", muscles="Biceps", met=3.5),
    "bench_press": _ex("Bench Press", ["bench press"], ARM, 95, 150, icon="🏋️", muscles="Chest, triceps"),
    "pull_up": _ex("Pull-up", ["pull up", "pull-up", "pullup", "chin up"], ARM, 90, 150, icon="🧗",
                   muscles="Back, biceps", met=8.0),
    "lunge": _ex("Lunge", ["lunge", "lunges"], KNEE, 105, 160, icon="🦵", muscles="Quads, glutes", met=4.0),
    "jumping_jack": _ex("Jumping Jack", ["jumping jack", "jumping jacks"], SHOULDER, 45, 140, start="low",
                        icon="⭐", muscles="Full body", met=8.0),
    # --- in the Kaggle dataset, not yet on the website ---
    "hammer_curl": _ex("Hammer Curl", ["hammer curl"], ARM, 60, 150, icon="🦾", muscles="Biceps, forearms", met=3.5),
    "incline_bench_press": _ex("Incline Bench Press", ["incline bench press"], ARM, 95, 150, muscles="Upper chest"),
    "decline_bench_press": _ex("Decline Bench Press", ["decline bench press"], ARM, 95, 150, muscles="Lower chest"),
    "lateral_raise": _ex("Lateral Raise", ["lateral raise", "lateral raises"], SHOULDER, 30, 75, start="low",
                         icon="🙆", muscles="Shoulders", met=3.5),
    "deadlift": _ex("Deadlift", ["deadlift"], HIP, 110, 165, muscles="Back, glutes, hamstrings", met=6.0),
    "romanian_deadlift": _ex("Romanian Deadlift", ["romanian deadlift", "rdl"], HIP, 110, 165,
                             muscles="Hamstrings, glutes", met=5.0),
    "lat_pulldown": _ex("Lat Pulldown", ["lat pulldown"], ARM, 80, 150, muscles="Back, biceps", met=4.0),
    "t_bar_row": _ex("T-Bar Row", ["t bar row", "t-bar row"], ARM, 90, 150, muscles="Back", met=5.0),
    "tricep_pushdown": _ex("Tricep Pushdown", ["tricep pushdown", "triceps pushdown"], ARM, 80, 150, start="low",
                           muscles="Triceps", met=3.5),
    "tricep_dips": _ex("Tricep Dips", ["tricep dips", "triceps dips", "dips"], ARM, 95, 150,
                       muscles="Triceps, chest", met=5.0),
    "leg_extension": _ex("Leg Extension", ["leg extension"], KNEE, 100, 155, start="low", icon="🦵",
                         muscles="Quads", met=3.5),
    "hip_thrust": _ex("Hip Thrust", ["hip thrust"], HIP, 120, 165, start="low", muscles="Glutes", met=4.0),  # ROUGH
    "leg_raises": _ex("Leg Raises", ["leg raises", "leg raise"], HIP, 100, 160, muscles="Abs", met=3.5),  # ROUGH
    "chest_fly_machine": _ex("Chest Fly Machine", ["chest fly machine", "chest fly"],
                             ("left_elbow", "left_shoulder", "right_shoulder"), 110, 160,
                             muscles="Chest", met=3.5),  # ROUGH
    # Static hold: the thresholds can never be crossed, so no reps are counted.
    "plank": _ex("Plank", ["plank"], HIP, 0, 1, muscles="Core", met=3.0),
    "russian_twist": _ex("Russian Twist", ["russian twist"], ("left_shoulder", "left_hip", "right_hip"), 70, 100,
                         muscles="Obliques", met=3.5),  # ROUGH
}

# ---------------------------------------------------------------------------
# 3. Where the data is
# ---------------------------------------------------------------------------
# Folders that hold <class>/<video>.mp4 (or <class>/<video>/frame_*.jpg).
# Several folders are combined. Run `run_all.preview()` to see what was found.
DATA_DIRS = [
    "/kaggle/input/gym-workoutexercises-video/verified_data/verified_data",
]
# Optional separate test set (e.g. the dataset's own noisy `test` folder).
# Empty list = split DATA_DIRS 70 / 15 / 15 by video instead.
TEST_DIRS = [
    "/kaggle/input/gym-workoutexercises-video/test/test",
]
# Everything the pipeline writes (keypoints, prepared data, runs, the model zip).
WORK_DIR = "/kaggle/working/gym"

# ---------------------------------------------------------------------------
# 4. Preprocessing — must match the website
# ---------------------------------------------------------------------------
POSE_MODEL = "yolo11n-pose.pt"  # the website runs the same weights (shipped inside the model zip)
FPS = 6                          # the website sends ~6 frames per second (static/js/live.js)
FRAMES_FPS = 30                  # only for datasets of frame folders: the rate they were extracted at
IMGSZ = 640                      # YOLO input size; the website uses RECOGNITION_INPUT_SIZE (640)
WINDOW = 16                      # frames per sample; the website's classifier expects 16
STRIDE = 4
MIN_CONF = 0.3                   # frames with lower average keypoint confidence are dropped
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15             # only used when TEST_DIRS is empty
# Regex whose group(1) is the source-video id, so clips cut from one video stay
# in the same split. Check the file names first. None = every file is its own video.
GROUP_REGEX = None
SEED = 42

# ---------------------------------------------------------------------------
# 5. Models and how the best one is chosen
# ---------------------------------------------------------------------------
MODELS = ["rf", "xgb", "lstm", "gru", "cnn"]  # remove any you don't want to train
SELECT_BY = "macro_f1"   # "macro_f1" or "accuracy", measured on the VALIDATION set
FORCE_MODEL = None        # e.g. "lstm" to ship that model regardless of scores

ML_PARAMS = {"n_estimators": 500}
DL_PARAMS = {"hidden": 128, "epochs": 100, "patience": 10, "batch": 64, "lr": 1e-3}
