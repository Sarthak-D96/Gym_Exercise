"""
Register trained models from ml_models/ and make them the active ones.

    python manage.py register_models --classifier lstm.pt --sync-exercises

What it does:
  1. Checks the files exist in ml_models/ (downloads the YOLO-pose weights if missing).
  2. Loads the classifier once and checks it outputs one score per class in classes.json.
  3. Creates/updates the MLModel rows (pose + classifier), marks them not-dummy, activates them.
  4. With --sync-exercises: shows only the exercises the model knows in the website.
"""

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from recognition.models import MLModel
from recognition.pipeline import model_store
from recognition.pipeline.features import FEATURE_SIZE
from workouts.models import Exercise


class Command(BaseCommand):
    help = "Register trained pose + classifier files from ml_models/ and activate them."

    def add_arguments(self, parser):
        parser.add_argument("--classifier", required=True,
                            help="file in ml_models/: .pt (LSTM/GRU/CNN), .joblib (RF/XGBoost) "
                                 "or two_stage.json (two-stage bundle)")
        parser.add_argument("--classes", default="classes.json", help="classes.json from training (in ml_models/)")
        parser.add_argument("--pose", default="yolo11n-pose.pt", help="YOLO-pose weights in ml_models/")
        parser.add_argument("--model-version", default="1.0", dest="model_version")
        parser.add_argument("--metrics", help="optional runs/<model>_test.json from training, to store the scores")
        parser.add_argument("--sync-exercises", action="store_true",
                            help="activate exercises in classes.json and deactivate the rest")

    def handle(self, *args, **opts):
        model_dir = Path(settings.RECOGNITION["MODEL_DIR"])
        clf_file = model_dir / opts["classifier"]
        classes_file = model_dir / opts["classes"]
        pose_file = model_dir / opts["pose"]

        if not clf_file.exists():
            raise CommandError(f"{clf_file} not found. Copy your trained model into ml_models/.")
        if not classes_file.exists():
            raise CommandError(f"{classes_file} not found. Copy classes.json from the training run into ml_models/.")
        classes = json.loads(classes_file.read_text())

        if clf_file.suffix == ".pt":
            backend = "torchscript"
        elif clf_file.suffix == ".joblib":
            backend = "sklearn"
        elif clf_file.suffix == ".json":
            backend = "two_stage"
        else:
            raise CommandError("Classifier must be a .pt (TorchScript), .joblib or two_stage.json file.")

        if not pose_file.exists():
            self.stdout.write(f"Downloading {pose_file.name} ...")
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(str(pose_file))
            if not pose_file.exists():
                raise CommandError(f"Could not download {pose_file.name}; copy it into ml_models/ by hand.")

        # sanity check: the classifier loads and gives one probability per class
        clf = model_store.CLASSIFIER_BACKENDS[backend](classes=classes, model_path=str(clf_file))
        clf.load()
        probs = clf.predict_proba([[0.0] * FEATURE_SIZE] * 16, context={})
        if len(probs) != len(classes):
            raise CommandError(f"Model gives {len(probs)} outputs but classes.json has {len(classes)} classes.")
        pose = model_store.POSE_BACKENDS["yolo_pose"](model_path=str(pose_file))
        pose.load()

        metrics = {}
        if opts["metrics"]:
            m = json.loads(Path(opts["metrics"]).read_text())
            metrics = {k: m[k] for k in ("accuracy", "macro_f1", "ms_per_window", "ms_per_window_cpu", "size_mb",
                                         "split", "val_accuracy", "val_macro_f1", "model")
                       if k in m}

        rows = [
            ("pose", "yolo_pose", opts["pose"], [], {}, "YOLO-pose (COCO-17), same weights as training."),
            ("classifier", backend, opts["classifier"], classes, metrics, f"Trained classifier {clf_file.name}."),
        ]
        for kind, name, path, cls, met, notes in rows:
            row, _ = MLModel.objects.update_or_create(
                kind=kind, version=opts["model_version"],
                defaults=dict(name=name, model_path=path, classes=cls, metrics=met, is_dummy=False, notes=notes),
            )
            row.activate()
            self.stdout.write(self.style.SUCCESS(f"Active {kind}: {name} v{row.version} ({path})"))
        model_store.clear_cache()

        known = set(Exercise.objects.values_list("slug", flat=True))
        missing = [c for c in classes if c not in known]
        if missing:
            self.stdout.write(self.style.WARNING(
                f"No exercise in the catalogue for: {', '.join(missing)}. Add them in Django admin "
                "(Workouts → Exercises) with these slugs, or reps won't be counted for them."))
        if opts["sync_exercises"]:
            Exercise.objects.filter(slug__in=classes).update(is_active=True)
            off = Exercise.objects.exclude(slug__in=classes).filter(is_active=True)
            names = list(off.values_list("slug", flat=True))
            off.update(is_active=False)
            self.stdout.write(f"Exercises shown: {', '.join(c for c in classes if c in known)}")
            if names:
                self.stdout.write(f"Hidden (model can't recognise them): {', '.join(names)}")
