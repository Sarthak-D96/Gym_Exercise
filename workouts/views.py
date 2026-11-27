import json
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .models import Exercise, InvalidTransition, WorkoutRecord, WorkoutSession, form_score
from .services import FrameRejected, process_frame, session_summary


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

@login_required
def dashboard(request):
    user = request.user
    week_ago = timezone.now() - timedelta(days=7)
    sessions = WorkoutSession.objects.filter(user=user)
    completed = sessions.filter(status=WorkoutSession.Status.COMPLETED)
    records = WorkoutRecord.objects.filter(session__user=user)
    unfinished = sessions.exclude(status=WorkoutSession.Status.COMPLETED).first()
    context = {
        "exercises": Exercise.objects.filter(is_active=True),
        "unfinished": unfinished,
        "recent": completed.prefetch_related("records__exercise")[:5],
        "stats": {
            "sessions_week": completed.filter(created_at__gte=week_ago).count(),
            "weekly_target": user.profile.weekly_target_sessions,
            "total_sessions": completed.count(),
            "total_reps": records.aggregate(s=Sum("repetitions"))["s"] or 0,
            "total_minutes": round(sum(s.active_seconds for s in completed) / 60),
        },
    }
    return render(request, "workouts/dashboard.html", context)


@login_required
@require_POST
def start_workout(request):
    unfinished = WorkoutSession.objects.filter(user=request.user).exclude(
        status=WorkoutSession.Status.COMPLETED
    )
    for s in unfinished:  # only one open session per user
        s.transition("end")
    slug = request.POST.get("exercise") or None
    target = Exercise.objects.filter(slug=slug, is_active=True).first() if slug else None
    session = WorkoutSession.objects.create(user=request.user, target_exercise=target)
    return redirect("workouts:live", pk=session.pk)


@login_required
def live_workout(request, pk):
    session = get_object_or_404(WorkoutSession, pk=pk, user=request.user)
    if session.status == WorkoutSession.Status.COMPLETED:
        return redirect("workouts:session_detail", pk=pk)
    return render(request, "workouts/live.html", {
        "session": session,
        "summary": session_summary(session),
    })


@login_required
def history(request):
    sessions = (
        WorkoutSession.objects.filter(user=request.user, status=WorkoutSession.Status.COMPLETED)
        .prefetch_related("records__exercise")
    )
    return render(request, "workouts/history.html", {"sessions": sessions[:50]})


@login_required
def session_detail(request, pk):
    session = get_object_or_404(
        WorkoutSession.objects.prefetch_related("records__exercise", "feedback"), pk=pk, user=request.user
    )
    return render(request, "workouts/session_detail.html", {"session": session})


@login_required
@require_POST
def delete_session(request, pk):
    session = get_object_or_404(WorkoutSession, pk=pk, user=request.user)
    session.delete()
    messages.success(request, "Workout deleted.")
    return redirect("workouts:history")


# --------------------------------------------------------------------------
# JSON API (used by static/js/*.js)
# --------------------------------------------------------------------------

def _json_body(request) -> dict:
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return {}


def _api_login_required(view):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
        return view(request, *args, **kwargs)
    wrapper.__name__ = view.__name__
    return wrapper


@_api_login_required
@require_GET
def api_exercises(request):
    data = [
        {"slug": e.slug, "name": e.name, "muscle_group": e.muscle_group, "icon": e.icon}
        for e in Exercise.objects.filter(is_active=True)
    ]
    return JsonResponse({"exercises": data})


@_api_login_required
@require_GET
def api_session(request, pk):
    session = get_object_or_404(WorkoutSession, pk=pk, user=request.user)
    return JsonResponse(session_summary(session))


@_api_login_required
@require_POST
def api_session_transition(request, pk, event):
    session = get_object_or_404(WorkoutSession, pk=pk, user=request.user)
    try:
        session.transition(event)
    except InvalidTransition as exc:
        return JsonResponse({"error": str(exc)}, status=409)
    if event == "end":
        session.records.filter(repetitions=0).delete()
    return JsonResponse(session_summary(session))


@_api_login_required
@require_POST
def api_session_frame(request, pk):
    body = _json_body(request)
    try:
        t = float(body.get("t", 0))
    except (TypeError, ValueError):
        return JsonResponse({"error": "'t' must be a number (ms of active workout time)"}, status=400)
    try:
        result = process_frame(pk, request.user, body.get("frame"), t)
    except WorkoutSession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)
    except FrameRejected as exc:
        return JsonResponse({"error": str(exc)}, status=409)
    return JsonResponse(result)


@_api_login_required
@require_GET
def api_history(request):
    """Aggregates for the History / Progress charts (UC-02)."""
    try:
        days = max(1, min(365, int(request.GET.get("days", 30))))
    except ValueError:
        days = 30
    since = timezone.localdate() - timedelta(days=days - 1)
    records = WorkoutRecord.objects.filter(
        session__user=request.user,
        session__status=WorkoutSession.Status.COMPLETED,
        session__created_at__date__gte=since,
    )
    daily = {
        row["day"]: row
        for row in records.annotate(day=TruncDate("session__created_at"))
        .values("day")
        .annotate(reps=Sum("repetitions"), good=Sum("good_form_reps"), rejected=Sum("rejected_reps"),
                  points=Sum("form_points"), sessions=Count("session", distinct=True))
    }
    series = []
    for i in range(days):
        d = since + timedelta(days=i)
        row = daily.get(d, {})
        reps = row.get("reps") or 0
        series.append({
            "date": d.isoformat(),
            "reps": reps,
            "sessions": row.get("sessions") or 0,
            "form_score": form_score(reps, row.get("good"), row.get("rejected"), row.get("points")),
            "rejected": row.get("rejected") or 0,
        })
    by_exercise = list(
        records.values("exercise__slug", "exercise__name")
        .annotate(reps=Sum("repetitions"), good=Sum("good_form_reps"), rejected=Sum("rejected_reps"),
                  blocks=Count("id"))
        .order_by("-reps")
    )
    return JsonResponse({"days": days, "daily": series, "by_exercise": by_exercise})
