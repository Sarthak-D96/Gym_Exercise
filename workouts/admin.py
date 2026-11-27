from django.contrib import admin

from .models import Exercise, FeedbackEvent, WorkoutRecord, WorkoutSession


@admin.register(Exercise)
class ExerciseAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "muscle_group", "angle_points", "low_angle", "high_angle", "start_phase", "is_active")
    list_editable = ("is_active",)
    prepopulated_fields = {"slug": ("name",)}


class WorkoutRecordInline(admin.TabularInline):
    model = WorkoutRecord
    extra = 0


class FeedbackInline(admin.TabularInline):
    model = FeedbackEvent
    extra = 0
    fields = ("created_at", "level", "code", "message")
    readonly_fields = ("created_at",)


@admin.register(WorkoutSession)
class WorkoutSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "status", "target_exercise", "created_at", "active_seconds", "frames_processed")
    list_filter = ("status", "target_exercise")
    search_fields = ("user__username",)
    inlines = [WorkoutRecordInline, FeedbackInline]
    readonly_fields = ("pipeline_state",)


@admin.register(WorkoutRecord)
class WorkoutRecordAdmin(admin.ModelAdmin):
    list_display = ("session", "exercise", "repetitions", "good_form_reps", "rejected_reps", "avg_confidence", "started_at")
    list_filter = ("exercise",)


@admin.register(FeedbackEvent)
class FeedbackEventAdmin(admin.ModelAdmin):
    list_display = ("created_at", "session", "level", "code", "message")
    list_filter = ("level", "code")
