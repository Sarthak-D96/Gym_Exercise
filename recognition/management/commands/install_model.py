"""
Install a model zip produced by training/run_all.py and make it the active model.

    python manage.py install_model ~/Downloads/gymrec_model_lstm-20261125-1430.zip

What it does:
  1. Unpacks the zip into ml_models/<model id>/ (a folder works too).
  2. Warns if this machine's library versions or input size differ from training.
  3. Adds/updates the website exercises for the trained classes (rep-counting
     settings from training/gym_config.py), so every class the model knows can be counted.
  4. Runs `register_models`: checks the model loads and outputs one score per class,
     activates it together with its pose weights, stores its scores, and shows
     only the trained exercises on the website.

Installing an older zip again switches back to it.
"""

import json
import shutil
import sys
import zipfile
from importlib import metadata
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from workouts.models import Exercise

EXERCISE_FIELDS = {"name", "icon", "muscle_group", "met_value", "angle_points", "low_angle", "high_angle",
                   "start_phase", "description"}


class Command(BaseCommand):
    help = "Install a gymrec_model_*.zip from training and activate it."

    def add_arguments(self, parser):
        parser.add_argument("bundle", help="path to gymrec_model_<id>.zip (or the unzipped folder)")
        parser.add_argument("--keep-exercise-settings", action="store_true",
                            help="don't overwrite rep-counting settings of exercises that already exist")
        parser.add_argument("--no-sync", action="store_true",
                            help="keep showing exercises the model doesn't know")

    def handle(self, *args, bundle, keep_exercise_settings, no_sync, **opts):
        model_dir = Path(settings.RECOGNITION["MODEL_DIR"])
        model_dir.mkdir(parents=True, exist_ok=True)
        folder = self._unpack(Path(bundle).expanduser(), model_dir)

        card_file = folder / "model_card.json"
        if not card_file.exists():
            raise CommandError(f"{card_file} not found: is this a zip from training/run_all.py?")
        card = json.loads(card_file.read_text())
        rel = folder.relative_to(model_dir).as_posix()
        self.stdout.write(f"Model {card['id']}: {card['classifier']['name']}, "
                          f"{len(card['classes'])} classes ({', '.join(card['classes'])})")
        for name, st in (card.get("stages") or {}).items():
            self.stdout.write(f"  stage {name}: {st['model']} -> {', '.join(st['classes'])}")

        self._check_environment(card)
        self._sync_exercises(card, keep_exercise_settings)

        pose = card.get("pose") or {}
        pose_arg = f"{rel}/{pose['file']}" if pose.get("file") else Path(pose.get("name") or "yolo11n-pose.pt").name
        call_command(
            "register_models",
            classifier=f"{rel}/{card['classifier']['file']}",
            classes=f"{rel}/classes.json",
            pose=pose_arg,
            model_version=card["id"][:30],
            metrics=str(folder / "metrics_test.json"),
            sync_exercises=not no_sync,
            stdout=self.stdout,
        )

        if (folder / "form_profiles.json").exists():
            self.stdout.write("Form grading learned from the dataset:")
            call_command("install_form_profiles", str(folder / "form_profiles.json"), stdout=self.stdout,
                         keep_angles=keep_exercise_settings)

        m = card.get("metrics", {})
        if "accuracy" in m:
            self.stdout.write(self.style.SUCCESS(
                f"Installed. {m.get('split', 'test')} accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}."))
        if settings.RECOGNITION.get("ALLOW_SIMULATION"):
            self.stdout.write("Note: set RECOGNITION_ALLOW_SIMULATION=false in .env so frames without a camera "
                              "image are rejected instead of returning 'no person'.")
        self.stdout.write("Restart the server (runserver reloads by itself) and start a workout to try it.")

    # ------------------------------------------------------------------ steps
    def _unpack(self, src: Path, model_dir: Path) -> Path:
        if src.is_dir():
            if src.resolve().parent == model_dir.resolve():
                return src.resolve()
            dest = model_dir / src.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            return dest
        if not src.exists():
            raise CommandError(f"{src} not found.")
        if not zipfile.is_zipfile(src):
            raise CommandError(f"{src} is not a zip file.")
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if not n.startswith("__MACOSX/")]
            tops = {n.split("/", 1)[0] for n in names}
            if len(tops) != 1 or any(".." in Path(n).parts or Path(n).is_absolute() for n in names):
                raise CommandError("Unexpected zip layout: expected one gymrec_model_<id>/ folder.")
            top = tops.pop()
            dest = model_dir / top
            if dest.exists():
                shutil.rmtree(dest)
            for n in names:
                z.extract(n, model_dir)
        self.stdout.write(f"Unpacked to {dest}")
        return dest

    def _check_environment(self, card: dict):
        trained = card.get("library_versions", {})
        clf = card["classifier"]
        backends = clf.get("backends") or [clf["backend"]]  # a two-stage bundle lists every stage's backend
        models = clf.get("models") or [clf["model"]]
        needed = (["torch"] if "torchscript" in backends else []) + (["scikit-learn"] if "sklearn" in backends else [])
        if "xgb" in models:
            needed.append("xgboost")
        for pkg in needed + ["ultralytics"]:
            # "+cu130" etc. is the GPU build used for training; pip needs the plain version here
            want = (trained.get(pkg) or "").split("+")[0] or None
            try:
                have = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                raise CommandError(
                    f"{pkg} is not installed for {sys.executable}.\n"
                    "Use the project's virtual environment (source .venv/bin/activate), "
                    "or install the model libraries: pip install -r requirements.txt")
            if want and have.split(".")[:2] != want.split(".")[:2]:
                self.stdout.write(self.style.WARNING(
                    f"{pkg} {have} here, {want} in training. If loading fails: pip install {pkg}=={want}"))
        pre = card.get("preprocessing", {})
        size = settings.RECOGNITION.get("INPUT_SIZE")
        if pre.get("imgsz") and size != pre["imgsz"]:
            self.stdout.write(self.style.WARNING(
                f"Training used {pre['imgsz']}px frames but RECOGNITION_INPUT_SIZE is {size}. "
                f"Set RECOGNITION_INPUT_SIZE={pre['imgsz']} in .env for matching results."))
        if pre.get("window") and pre["window"] != 16:
            self.stdout.write(self.style.WARNING(
                f"Model was trained on {pre['window']}-frame windows; the website feeds 16 "
                "(recognition/pipeline/trained_models.py WINDOW)."))

    def _sync_exercises(self, card: dict, keep_existing: bool):
        specs = card.get("exercises") or {}
        for slug in card["classes"]:
            spec = {k: v for k, v in (specs.get(slug) or {}).items() if k in EXERCISE_FIELDS}
            existing = Exercise.objects.filter(slug=slug).first()
            if existing is None:
                if not spec.get("angle_points"):
                    self.stdout.write(self.style.WARNING(
                        f"No rep-counting settings for '{slug}'; add it in Django admin (Workouts → Exercises)."))
                    continue
                Exercise.objects.create(slug=slug, **spec)
                self.stdout.write(f"Added exercise {spec.get('name', slug)}")
            elif spec and not keep_existing:
                for k, v in spec.items():
                    setattr(existing, k, v)
                existing.is_active = True
                existing.save()
