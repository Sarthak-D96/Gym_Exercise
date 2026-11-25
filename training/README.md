# Training on another computer (Phases 3–4)

## Quick start: one config, one command, best model to the website

1. **Choose the classes** in `training/gym_config.py` (`CLASSES`). Each class lists the dataset folder names
   that belong to it, and the rep-counting settings the website will use.
2. **Run everything** with `training/kaggle_train.ipynb` on Kaggle, `training/colab_train.ipynb` on Google Colab, or with
   `python training/run_all.py` on any machine:
   `extract → build → train (rf, xgb, lstm, gru, cnn) → select`.
   * `python training/run_all.py --preview` shows which folders map to which class before you start.
   * `--steps train select` re-runs only some steps.
3. **Which model ships:** the one with the best **validation** macro-F1 (`config.SELECT_BY`).
   Ties go to higher accuracy, then the faster model. Every model is then scored once on the test set,
   for the report. The winner is packed as `WORK_DIR/gymrec_model_<model>-<date>.zip`, containing:
   the model, `classes.json`, the pose weights, the scores, `results.md` and `model_card.json`.
4. **On the laptop:** `python manage.py install_model path/to/gymrec_model_<id>.zip`. That's all.

## Two-stage model (router + specialists)

`config.TWO_STAGE = True` (the default) trains, instead of one model for all classes:

```
16-frame window ─► router: which group?  (config.GROUPS: legs, press, pull, arms, core, ...)
                     └─► that group's specialist: which exercise?
P(exercise) = P(group) × P(exercise | group)
```

* Every stage (the router and each specialist) is trained with every type in `config.MODELS` and keeps
  its best on **validation**, so the router might be XGBoost and the legs specialist a CNN.
* All stages also get 13 extra features computed from the 42 base ones
  (`recognition/pipeline/extra_features.py`, shared with the website): torso tilt, shoulder/hip width,
  wrist and elbow height, wrist reach, shin angle, body aspect, grip width.
* `select_two_stage.py` scores each stage and the whole chain on validation and test. If a single-stage
  run on the same classes exists in `WORK_DIR/runs/`, it is compared too (`results.md`).
* Output: `WORK_DIR/gymrec_model_two_stage-<date>.zip`, installed with the same `install_model` command.
  The website loads it through `TwoStageClassifier` (`recognition/pipeline/trained_models.py`).
* Notebook: `training/colab_two_stage.ipynb`.
* Training resumes after a disconnect: models already trained on the current data are skipped
  (`config.RETRAIN = True` retrains them).

Set `TWO_STAGE = False` for the original single-model setup.

The sections below run the same scripts by hand.

---
Everything heavy (the ~10 GB dataset, keypoint extraction, training) happens on the
training machine. The only thing that comes back to the laptop is the trained model
(a few MB) plus `classes.json`.

```
videos / frame folders ──extract_keypoints.py──► keypoints/*.npz      (YOLO-pose, 6 fps)
keypoints/*.npz        ──build_dataset.py─────► prepared/{train,val,test}.npz, classes.json, splits.json
prepared/              ──train_ml.py──────────► runs/rf.joblib, runs/xgb.joblib
prepared/              ──train_dl.py──────────► runs/lstm.pt, gru.pt, cnn.pt   (TorchScript)
runs/                  ──summarize.py─────────► results table for JOURNAL.md
```

Features are computed with the same code as the website (`build_dataset.py` checks
that its NumPy version gives identical numbers to `recognition/pipeline/features.py`).

**Needs:** Python 3.10–3.12, ~25 GB free disk, an NVIDIA GPU if possible
(everything also runs on CPU, just slower). Commands below are for Linux/macOS;
Windows differences are noted.

---

## 1. On the laptop: push the code

```bash
cd ~/Desktop/Gym_Exercise
git add training
git commit -m "Add training scripts"
git push
```

## 2. On the training machine: set up

```bash
git clone https://github.com/Sarthak-D96/Gym_Exercise.git
cd Gym_Exercise
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# PyTorch first, with the right CUDA build (pick the command for your GPU at https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# no NVIDIA GPU:  pip install torch

pip install -r requirements.txt -r training/requirements-train.txt
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

## 3. Download the dataset

Get an API token: kaggle.com → Settings → API → *Create New Token*. Put the
downloaded `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
export DATA=~/gym_data                 # Windows PowerShell: $env:DATA="$HOME\gym_data"
kaggle datasets download -d philosopher0808/gym-workoutexercises-video -p $DATA --unzip
find $DATA -maxdepth 3 -type d | head -40   # check the layout
```

The scripts expect one folder per class, holding either videos or folders of frames:
`<root>/<class>/<video>.mp4` or `<root>/<class>/<video>/frame_00001.jpg`.
Adjust the paths below if the unzipped folder names differ.

## 4. Extract keypoints (the slow step)

Run long jobs inside `tmux` or `screen` (or with `nohup ... &`) so they survive a closed terminal.
Safe to stop and restart: finished videos are skipped.

```bash
python training/extract_keypoints.py --data $DATA/verified_data --out $DATA/keypoints      --device 0
python training/extract_keypoints.py --data $DATA/test          --out $DATA/keypoints_test --device 0
```

* `--device 0` = first GPU; use `--device cpu` without one (`mps` on an Apple-silicon Mac).
* Videos are sampled at 6 fps (`--fps`), the rate the website sends frames.
* **If your data is frame folders** (like the `bench_press` sample) instead of videos, say what
  frame rate they were extracted at so every Nth frame is taken, e.g. `--frames-fps 30`.
* `--pose-model yolo11n-pose.pt` is the default (fast enough for the website). Whichever you pick,
  use the same one in the website later.

## 5. Build the dataset

```bash
python training/build_dataset.py --keypoints $DATA/keypoints --test-keypoints $DATA/keypoints_test \
    --out $DATA/prepared
```

Useful options:

* Only some classes, with website slugs:
  `--classes squat push_up shoulder_press bicep_curl bench_press --alias barbell_biceps_curl=bicep_curl`
  (class folder names are turned into slugs: `push-up` → `push_up`.)
* If clips cut from one source video share a name pattern, keep them together:
  `--group-regex '^(.*)_\d+$'` (group 1 = the source id). Check the file names first.
* Without `--test-keypoints`, it splits 70/15/15 by video instead.

Read `$DATA/prepared/summary.json` for windows per class and dropped videos (Phase 3 results).

## 6. Train

Tune on the validation set; these print validation scores only.

```bash
python training/train_ml.py --data $DATA/prepared --model rf
python training/train_ml.py --data $DATA/prepared --model xgb --device cuda   # or --device cpu
python training/train_dl.py --data $DATA/prepared --model lstm
python training/train_dl.py --data $DATA/prepared --model gru
python training/train_dl.py --data $DATA/prepared --model cnn
python training/summarize.py --runs runs
```

DL options: `--hidden 64|128`, `--epochs`, `--patience`, `--batch`, `--lr`, `--no-augment`.
Each run writes `runs/<model>_val.json` (scores, per-class F1, confusion matrix, speed, size)
and `runs/<model>_val_confusion.csv`.

When you've settled on settings, score the test set **once**:

```bash
python training/train_dl.py --data $DATA/prepared --model lstm --eval-test   # same flags as your best run
python training/train_ml.py --data $DATA/prepared --model xgb --device cuda --eval-test
python training/summarize.py --runs runs --test
```

## 7. Bring the model back to the laptop

Copy the chosen model and `classes.json` into the laptop's `ml_models/` (git ignores that folder),
e.g. with `scp`, a USB stick or Google Drive:

```bash
# run on the laptop
scp user@training-machine:~/Gym_Exercise/runs/lstm.pt ~/Desktop/Gym_Exercise/ml_models/classifier_v1.pt
scp user@training-machine:~/Gym_Exercise/runs/classes.json       ~/Desktop/Gym_Exercise/ml_models/
```

* `.pt` files are TorchScript: input `(batch, T, 42)` features, output logits, works with T < 16.
  Load with `torch.jit.load(path)` exactly as in `ml_models/README.md`.
* `.joblib` files (RF/XGBoost) hold `{"model", "classes"}`; their input is
  `common.window_stats(window)` (210 numbers), not the raw window.

Then follow `ml_models/README.md` to plug it into the website.
