from django.db import models
from django.db.models import Q


class MLModel(models.Model):
    """
    Metadata for a trained model kept in Model Storage (DFD store D2).

    Class diagram: MLModel(modelPath, version) with load()/predict()/update().
    The weights themselves live on disk under settings.RECOGNITION["MODEL_DIR"];
    this table records which file/version is active for each pipeline stage.
    """

    class Kind(models.TextChoices):
        POSE = "pose", "Pose detector"
        CLASSIFIER = "classifier", "Exercise classifier"
        POSTURE = "posture", "Posture model"

    name = models.CharField(max_length=100)
    kind = models.CharField(max_length=20, choices=Kind.choices)
    version = models.CharField(max_length=30)
    model_path = models.CharField(max_length=255, blank=True, help_text="Path relative to the model directory.")
    classes = models.JSONField(default=list, blank=True, help_text="Output labels (classifier only).")
    metrics = models.JSONField(default=dict, blank=True, help_text="e.g. accuracy / F1 from evaluation.")
    is_active = models.BooleanField(default=False)
    is_dummy = models.BooleanField(default=True, help_text="True while the placeholder implementation is used.")
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["kind", "-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["kind", "version"], name="unique_model_version_per_kind"),
            models.UniqueConstraint(fields=["kind"], condition=Q(is_active=True), name="one_active_model_per_kind"),
        ]

    def __str__(self):
        return f"{self.get_kind_display()} {self.name} v{self.version}"

    def activate(self):
        """Make this the active model for its kind (deactivates the others)."""
        MLModel.objects.filter(kind=self.kind, is_active=True).exclude(pk=self.pk).update(is_active=False)
        self.is_active = True
        self.save(update_fields=["is_active"])


class InferenceLog(models.Model):
    """Per-frame inference telemetry — feeds the admin performance/usage report (US-06)."""

    session = models.ForeignKey("workouts.WorkoutSession", on_delete=models.CASCADE, related_name="inference_logs")
    predicted_exercise = models.CharField(max_length=50, blank=True)
    confidence = models.FloatField(default=0)
    latency_ms = models.FloatField(default=0)
    simulated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
