from django.db import migrations

MODELS = [
    dict(name="dummy", kind="pose", version="0.0-dummy", is_active=True, is_dummy=True,
         notes="Placeholder: synthesises keypoints. Replace with MediaPipe/MoveNet/YOLO-pose."),
    dict(name="dummy", kind="classifier", version="0.0-dummy", is_active=True, is_dummy=True,
         classes=["bicep_curl", "jumping_jack", "lunge", "push_up", "shoulder_press", "squat"],
         notes="Placeholder: echoes the simulated exercise. Replace with the trained classifier."),
    dict(name="rules", kind="posture", version="1.0", is_active=True, is_dummy=False,
         notes="Rule-based posture analyser (recognition/pipeline/posture.py)."),
]


def seed(apps, schema_editor):
    MLModel = apps.get_model("recognition", "MLModel")
    for data in MODELS:
        MLModel.objects.update_or_create(kind=data["kind"], version=data["version"], defaults=data)


def unseed(apps, schema_editor):
    apps.get_model("recognition", "MLModel").objects.filter(version__in=["0.0-dummy", "1.0"]).delete()


class Migration(migrations.Migration):
    dependencies = [("recognition", "0001_initial")]
    operations = [migrations.RunPython(seed, unseed)]
