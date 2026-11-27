from django.conf import settings
from django.db import models
from django.utils import timezone


class Exercise(models.Model):
    """
    Exercise catalogue (deployment diagram: "Exercise details").

    The rep-counter configuration lives here so new exercises can be added
    without code changes: the angle at `angle_joint` (formed by the three
    keypoints in `angle_points`) is tracked, and a repetition is counted each
    time the angle leaves the start phase, reaches the opposite threshold and
    returns.
    """

    class StartPhase(models.TextChoices):
        HIGH = "high", "Starts extended (angle high)"
        LOW = "low", "Starts flexed (angle low)"

    slug = models.SlugField(unique=True)
    name = models.CharField(max_length=60)
    muscle_group = models.CharField(max_length=60, blank=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=8, blank=True, help_text="Emoji shown in the UI.")
    angle_points = models.JSONField(help_text='Three keypoint names, e.g. ["left_hip", "left_knee", "left_ankle"].')
    low_angle = models.FloatField(help_text="Angle (deg) that marks the flexed position.")
    high_angle = models.FloatField(help_text="Angle (deg) that marks the extended position.")
    start_phase = models.CharField(max_length=4, choices=StartPhase.choices, default=StartPhase.HIGH)
    met_value = models.FloatField(default=5.0, help_text="Metabolic equivalent, used for calorie estimate.")
    form_profile = models.JSONField(
        default=dict, blank=True,
        help_text="What correct reps look like, learned from the dataset (training/build_form_profiles.py). "
                  "Empty: built-in defaults.")
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class InvalidTransition(Exception):
    pass


class WorkoutSession(models.Model):
    """
    One workout. Implements the state chart from the design phase:
    Ready -> Active <-> Paused -> Completed (Idle / Logged-in are auth states).
    """

    class Status(models.TextChoices):
        READY = "ready", "Ready"
        ACTIVE = "active", "Workout active"
        PAUSED = "paused", "Paused"
        COMPLETED = "completed", "Completed"

    TRANSITIONS = {
        "begin": ({Status.READY}, Status.ACTIVE),
        "pause": ({Status.ACTIVE}, Status.PAUSED),
        "resume": ({Status.PAUSED}, Status.ACTIVE),
        "end": ({Status.READY, Status.ACTIVE, Status.PAUSED}, Status.COMPLETED),
    }

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="workout_sessions")
    target_exercise = models.ForeignKey(
        Exercise, null=True, blank=True, on_delete=models.SET_NULL,
        help_text="Optional: exercise the user chose. Empty means auto-detect.",
    )
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.READY, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    last_resumed_at = models.DateTimeField(null=True, blank=True)
    active_seconds = models.FloatField(default=0, help_text="Time spent in ACTIVE state (pauses excluded).")
    frames_processed = models.PositiveIntegerField(default=0)
    # Serialized state of the recognition engine (prediction window, rep counter
    # phase, ...). Kept in the DB so any worker process can handle the next frame.
    pipeline_state = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Session #{self.pk} ({self.user}) — {self.status}"

    # --- state machine ------------------------------------------------------
    def transition(self, event: str):
        if event not in self.TRANSITIONS:
            raise InvalidTransition(f"Unknown event '{event}'")
        allowed_from, target = self.TRANSITIONS[event]
        if self.status not in allowed_from:
            raise InvalidTransition(f"Cannot '{event}' a session that is {self.get_status_display().lower()}")
        now = timezone.now()
        if self.status == self.Status.ACTIVE and self.last_resumed_at:
            self.active_seconds += (now - self.last_resumed_at).total_seconds()
            self.last_resumed_at = None
        if target == self.Status.ACTIVE:
            self.last_resumed_at = now
            self.started_at = self.started_at or now
        if target == self.Status.COMPLETED:
            self.ended_at = now
            self.records.filter(ended_at__isnull=True).update(ended_at=now)
        self.status = target
        self.save()

    # --- derived values -----------------------------------------------------
    @property
    def duration_seconds(self) -> float:
        extra = 0.0
        if self.status == self.Status.ACTIVE and self.last_resumed_at:
            extra = (timezone.now() - self.last_resumed_at).total_seconds()
        return self.active_seconds + extra

    @property
    def total_reps(self) -> int:
        return sum(r.repetitions for r in self.records.all())

    @property
    def estimated_calories(self) -> float:
        """Rough estimate: MET * 3.5 * kg / 200 per minute (70 kg default)."""
        weight = 70.0
        profile = getattr(self.user, "profile", None)
        if profile and profile.weight_kg:
            weight = float(profile.weight_kg)
        total = 0.0
        for r in self.records.select_related("exercise"):
            total += r.exercise.met_value * 3.5 * weight / 200 * (r.duration_seconds / 60)
        return round(total, 1)


def form_score(reps, good, rejected, points) -> int | None:
    reps, good, rejected, points = reps or 0, good or 0, rejected or 0, points or 0
    if not reps + rejected:
        return None
    if points or rejected:
        return round(points / (reps + rejected))
    return round(100 * good / reps)  # records from before rep scores existed


class WorkoutRecord(models.Model):
    """
    One continuous block of a single recognised exercise inside a session
    (class diagram: WorkoutRecord(exerciseName, repetitions); DFD store D3).
    """

    session = models.ForeignKey(WorkoutSession, on_delete=models.CASCADE, related_name="records")
    exercise = models.ForeignKey(Exercise, on_delete=models.PROTECT, related_name="records")
    repetitions = models.PositiveIntegerField(default=0, help_text="Counted reps (no major form fault).")
    good_form_reps = models.PositiveIntegerField(default=0, help_text="Counted reps without any form issue.")
    rejected_reps = models.PositiveIntegerField(default=0, help_text="Attempts not counted: major fault or partial.")
    form_points = models.FloatField(default=0, help_text="Sum of rep scores (0-100) over all attempts.")
    avg_confidence = models.FloatField(default=0)
    confidence_samples = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(default=timezone.now)
    ended_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["started_at"]

    def __str__(self):
        return f"{self.exercise.name}: {self.repetitions} reps"

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or timezone.now()
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def attempts(self) -> int:
        return self.repetitions + self.rejected_reps

    @property
    def form_score(self) -> int | None:
        """Average rep score (0-100) over all attempts, counted or not."""
        return form_score(self.repetitions, self.good_form_reps, self.rejected_reps, self.form_points)

    def add_confidence(self, value: float):
        self.confidence_samples += 1
        self.avg_confidence += (value - self.avg_confidence) / self.confidence_samples


class FeedbackEvent(models.Model):
    """Posture feedback produced during a session (FR7)."""

    class Level(models.TextChoices):
        GOOD = "good", "Good"
        INFO = "info", "Info"
        WARNING = "warning", "Warning"

    session = models.ForeignKey(WorkoutSession, on_delete=models.CASCADE, related_name="feedback")
    record = models.ForeignKey(WorkoutRecord, null=True, blank=True, on_delete=models.CASCADE, related_name="feedback")
    level = models.CharField(max_length=10, choices=Level.choices, default=Level.INFO)
    code = models.CharField(max_length=40, blank=True)
    message = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
