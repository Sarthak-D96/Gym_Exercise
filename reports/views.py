"""Admin usage / performance reports (US-06)."""

from datetime import timedelta

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.db.models import Avg, Count, Max, Sum
from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone

from recognition.models import InferenceLog, MLModel
from workouts.models import WorkoutRecord, WorkoutSession


@staff_member_required
def usage_report(request):
    since = timezone.now() - timedelta(days=30)
    logs = InferenceLog.objects.filter(created_at__gte=since)
    perf = logs.aggregate(avg=Avg("latency_ms"), worst=Max("latency_ms"), n=Count("id"), conf=Avg("confidence"))
    context = {
        "totals": {
            "users": User.objects.count(),
            "active_users": WorkoutSession.objects.filter(created_at__gte=since).values("user").distinct().count(),
            "sessions": WorkoutSession.objects.filter(created_at__gte=since).count(),
            "completed": WorkoutSession.objects.filter(created_at__gte=since, status="completed").count(),
            "reps": WorkoutRecord.objects.filter(session__created_at__gte=since).aggregate(s=Sum("repetitions"))["s"] or 0,
            "frames": perf["n"],
            "avg_latency": round(perf["avg"] or 0, 1),
            "max_latency": round(perf["worst"] or 0, 1),
            "avg_confidence": round(100 * (perf["conf"] or 0), 1),
            "simulated_share": round(100 * logs.filter(simulated=True).count() / perf["n"]) if perf["n"] else 0,
        },
        "models": MLModel.objects.all(),
        "top_users": (
            WorkoutSession.objects.filter(created_at__gte=since)
            .values("user__username")
            .annotate(sessions=Count("id", distinct=True), reps=Sum("records__repetitions"))
            .order_by("-sessions")[:10]
        ),
    }
    return render(request, "reports/usage.html", context)


@staff_member_required
def usage_data(request):
    since = timezone.localdate() - timedelta(days=29)
    sessions = dict(
        WorkoutSession.objects.filter(created_at__date__gte=since)
        .annotate(day=TruncDate("created_at")).values("day").annotate(n=Count("id")).values_list("day", "n")
    )
    latency = dict(
        InferenceLog.objects.filter(created_at__date__gte=since)
        .annotate(day=TruncDate("created_at")).values("day").annotate(a=Avg("latency_ms")).values_list("day", "a")
    )
    days = [since + timedelta(days=i) for i in range(30)]
    exercises = list(
        WorkoutRecord.objects.filter(session__created_at__date__gte=since)
        .values("exercise__name").annotate(reps=Sum("repetitions"), conf=Avg("avg_confidence"))
        .order_by("-reps")
    )
    return JsonResponse({
        "days": [d.isoformat() for d in days],
        "sessions": [sessions.get(d, 0) for d in days],
        "latency": [round(latency[d], 1) if d in latency else None for d in days],
        "exercises": exercises,
    })
