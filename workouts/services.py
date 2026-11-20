"""
Workout Controller (component diagram) — glue between HTTP views, the
recognition engine and the database.
"""

from __future__ import annotations

import time

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from recognition.models import InferenceLog
from recognition.pipeline import model_store
from recognition.pipeline.preprocess import FrameDecodeError, decode_data_url

from .models import Exercise, FeedbackEvent, WorkoutRecord, WorkoutSession


class FrameRejected(Exception):
    pass


def process_frame(session_id: int, user, frame_data: str | None, timestamp_ms: float) -> dict:
    """Run one frame through the pipeline and persist the outcome (sequence diagram main loop)."""
    raw = None
    if frame_data:
        try:
            raw = decode_data_url(frame_data)
        except FrameDecodeError as exc:
            raise FrameRejected(str(exc)) from exc
    elif not settings.RECOGNITION["ALLOW_SIMULATION"]:
        raise FrameRejected("No frame supplied and simulation mode is disabled.")

    engine = model_store.build_engine()

    with transaction.atomic():
        session = (
            WorkoutSession.objects.select_for_update(of=("self",))
            .select_related("target_exercise")
            .get(pk=session_id, user=user)
        )
        if session.status != WorkoutSession.Status.ACTIVE:
            raise FrameRejected(f"Session is {session.get_status_display().lower()}, not active.")

        target = session.target_exercise.slug if session.target_exercise else None
        context = {"target_exercise": target, "seed": session.pk}
        # Only consumed by the dummy pose detector, which needs to know what to simulate.
        if target:
            context["simulate_exercise"] = target

        started = time.perf_counter()
        try:
            result, new_state = engine.process(
                raw, timestamp_ms=timestamp_ms, state=session.pipeline_state, context=context
            )
        except FrameDecodeError as exc:
            raise FrameRejected(str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        record = _current_record(session, result)
        if record is not None:
            record.repetitions = result.reps
            record.good_form_reps = result.good_reps
            if result.confidence:
                record.add_confidence(result.confidence)
            record.save()

        if result.events:
            FeedbackEvent.objects.bulk_create(
                FeedbackEvent(session=session, record=record, level=e.level, code=e.code, message=e.message)
                for e in result.events
            )

        session.pipeline_state = new_state
        session.frames_processed += 1
        session.save(update_fields=["pipeline_state", "frames_processed"])

        InferenceLog.objects.create(
            session=session,
            predicted_exercise=result.exercise or "",
            confidence=result.confidence,
            latency_ms=latency_ms,
            simulated=result.simulated,
        )

    payload = result.to_dict()
    payload.update(
        latency_ms=round(latency_ms, 1),
        exercise_name=record.exercise.name if record else None,
        session=session_summary(session),
    )
    return payload


def _current_record(session: WorkoutSession, result) -> WorkoutRecord | None:
    """Open a new WorkoutRecord whenever the recognised exercise changes."""
    if not result.exercise:
        return session.records.select_related("exercise").filter(ended_at__isnull=True).last()
    open_record = session.records.select_related("exercise").filter(ended_at__isnull=True).last()
    if open_record and open_record.exercise.slug == result.exercise and not result.exercise_changed:
        return open_record
    now = timezone.now()
    if open_record:
        open_record.ended_at = now
        open_record.save(update_fields=["ended_at"])
        if open_record.repetitions == 0:
            open_record.delete()  # drop blocks where nothing was done (e.g. brief misclassification)
    exercise = Exercise.objects.get(slug=result.exercise)
    return WorkoutRecord.objects.create(session=session, exercise=exercise, started_at=now)


def session_summary(session: WorkoutSession) -> dict:
    records = list(session.records.select_related("exercise"))
    return {
        "id": session.pk,
        "status": session.status,
        "status_label": session.get_status_display(),
        "target_exercise": session.target_exercise.slug if session.target_exercise else None,
        "duration_seconds": round(session.duration_seconds, 1),
        "frames_processed": session.frames_processed,
        "total_reps": sum(r.repetitions for r in records),
        "records": [
            {
                "exercise": r.exercise.slug,
                "exercise_name": r.exercise.name,
                "repetitions": r.repetitions,
                "good_form_reps": r.good_form_reps,
                "form_score": r.form_score,
                "avg_confidence": round(r.avg_confidence, 3),
                "duration_seconds": round(r.duration_seconds, 1),
            }
            for r in records
        ],
    }
