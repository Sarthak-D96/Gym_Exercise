"""
Install form_profiles.json (training/build_form_profiles.py): what correct reps look like.

    python manage.py install_form_profiles ~/Downloads/form_profiles.json

For every exercise in the file that exists on the website:
  * stores the learned normal ranges in Exercise.form_profile, so reps are graded
    against the dataset instead of the built-in defaults (recognition/pipeline/posture.py);
  * sets the rep-counting angles (low/high) to the values calibrated from the
    dataset, when calibration succeeded (--keep-angles to skip).

install_model runs this automatically when the model zip contains form_profiles.json.
Undo for one exercise: clear its "Form profile" in Django admin (Workouts → Exercises).
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from recognition.pipeline import model_store
from workouts.models import Exercise


class Command(BaseCommand):
    help = "Install dataset-learned form profiles (form grading) and calibrated rep angles."

    def add_arguments(self, parser):
        parser.add_argument("profiles", help="form_profiles.json")
        parser.add_argument("--keep-angles", action="store_true", help="don't change the rep-counting angles")

    def handle(self, *args, profiles, keep_angles, **opts):
        path = Path(profiles).expanduser()
        if not path.exists():
            raise CommandError(f"{path} not found.")
        data = json.loads(path.read_text())
        entries = data.get("exercises")
        if not isinstance(entries, dict):
            raise CommandError(f"{path} is not a form_profiles.json from training/build_form_profiles.py.")

        for slug, entry in entries.items():
            ex = Exercise.objects.filter(slug=slug).first()
            if ex is None:
                self.stdout.write(f"  {slug}: not on the website, skipped")
                continue
            ex.form_profile = {"metrics": entry.get("metrics", {}), "reps": entry.get("reps", 0),
                               "videos": entry.get("videos", 0), "source": path.name,
                               "created": data.get("created")}
            line = f"  {ex.name}: {entry.get('reps', 0)} dataset reps"
            ang = entry.get("angles") or {}
            if ang.get("applied") and not keep_angles:
                old = (ex.low_angle, ex.high_angle)
                ex.low_angle, ex.high_angle = float(ang["low"]), float(ang["high"])
                if old != (ex.low_angle, ex.high_angle):
                    line += f", rep angles {old[0]:.0f}/{old[1]:.0f}° -> {ex.low_angle:.0f}/{ex.high_angle:.0f}°"
            test = entry.get("test") or {}
            if test.get("attempts"):
                line += (f", held-out reps counted {test.get('good', 0) + test.get('minor', 0)}"
                         f"/{test['attempts']}")
            ex.save()
            self.stdout.write(line)
        model_store.clear_cache()
        self.stdout.write(self.style.SUCCESS("Form profiles installed. Restart the server to use them."))
