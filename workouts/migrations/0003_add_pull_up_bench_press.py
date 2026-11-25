"""Adds the two exercises the trained classifier knows that the prototype catalogue didn't have."""

from django.db import migrations

EXERCISES = [
    dict(slug="bench_press", name="Bench Press", icon="🏋️", muscle_group="Chest, triceps", met_value=5.0,
         description="Barbell bench press. Tracked by elbow angle, starting with arms extended.",
         angle_points=["left_shoulder", "left_elbow", "left_wrist"], low_angle=95, high_angle=150, start_phase="high"),
    dict(slug="pull_up", name="Pull-up", icon="🧗", muscle_group="Back, biceps", met_value=8.0,
         description="Pull-up from a dead hang. Tracked by elbow angle.",
         angle_points=["left_shoulder", "left_elbow", "left_wrist"], low_angle=90, high_angle=150, start_phase="high"),
]


def seed(apps, schema_editor):
    Exercise = apps.get_model("workouts", "Exercise")
    for data in EXERCISES:
        Exercise.objects.update_or_create(slug=data["slug"], defaults=data)


def unseed(apps, schema_editor):
    apps.get_model("workouts", "Exercise").objects.filter(slug__in=[e["slug"] for e in EXERCISES]).delete()


class Migration(migrations.Migration):
    dependencies = [("workouts", "0002_seed_exercises")]
    operations = [migrations.RunPython(seed, unseed)]
