from django.contrib import admin, messages

from .models import InferenceLog, MLModel
from .pipeline import model_store


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ("name", "kind", "version", "is_active", "is_dummy", "model_path", "created_at")
    list_filter = ("kind", "is_active", "is_dummy")
    actions = ["activate"]

    @admin.action(description="Activate selected model (one per kind)")
    def activate(self, request, queryset):
        for model in queryset.order_by("kind", "-created_at").distinct("kind"):
            model.activate()
        model_store.clear_cache()
        messages.success(request, "Model(s) activated. Pipeline cache cleared.")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        model_store.clear_cache()


@admin.register(InferenceLog)
class InferenceLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "session", "predicted_exercise", "confidence", "latency_ms", "simulated")
    list_filter = ("simulated", "predicted_exercise")
    date_hierarchy = "created_at"
