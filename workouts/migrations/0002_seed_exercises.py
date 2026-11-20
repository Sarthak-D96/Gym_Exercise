from django.db import migrations

EXERCISES = [
    dict(slug="squat", name="Squat", icon="🏋️", muscle_group="Quads, glutes", met_value=5.0,
         description="Bodyweight or barbell squat. Tracked by knee angle.",
         angle_points=["left_hip", "left_knee", "left_ankle"], low_angle=100, high_angle=160, start_phase="high"),
    dict(slug="push_up", name="Push-up", icon="💪", muscle_group="Chest, triceps", met_value=8.0,
         description="Standard push-up. Tracked by elbow angle.",
         angle_points=["left_shoulder", "left_elbow", "left_wrist"], low_angle=95, high_angle=155, start_phase="high"),
    dict(slug="bicep_curl", name="Bicep Curl", icon="🦾", muscle_group="Biceps", met_value=3.5,
         description="Dumbbell curl. Tracked by elbow angle.",
         angle_points=["left_shoulder", "left_elbow", "left_wrist"], low_angle=60, high_angle=150, start_phase="high"),
    dict(slug="shoulder_press", name="Shoulder Press", icon="🙌", muscle_group="Shoulders", met_value=4.0,
         description="Overhead press. Tracked by elbow angle, starting from the rack position.",
         angle_points=["left_shoulder", "left_elbow", "left_wrist"], low_angle=100, high_angle=155, start_phase="low"),
    dict(slug="lunge", name="Lunge", icon="🦵", muscle_group="Quads, glutes", met_value=4.0,
         description="Forward / static lunge. Tracked by knee angle.",
         angle_points=["left_hip", "left_knee", "left_ankle"], low_angle=105, high_angle=160, start_phase="high"),
    dict(slug="jumping_jack", name="Jumping Jack", icon="⭐", muscle_group="Full body", met_value=8.0,
         description="Jumping jacks. Tracked by shoulder abduction angle.",
         angle_points=["left_hip", "left_shoulder", "left_elbow"], low_angle=45, high_angle=140, start_phase="low"),
]


def seed(apps, schema_editor):
    Exercise = apps.get_model("workouts", "Exercise")
    for data in EXERCISES:
        Exercise.objects.update_or_create(slug=data["slug"], defaults=data)


def unseed(apps, schema_editor):
    apps.get_model("workouts", "Exercise").objects.filter(slug__in=[e["slug"] for e in EXERCISES]).delete()


class Migration(migrations.Migration):
    dependencies = [("workouts", "0001_initial")]
    operations = [migrations.RunPython(seed, unseed)]
