"""
Create demo accounts and a few weeks of past workouts so History and Reports
have data to show (useful for screenshots / evaluation).

    python manage.py seed_demo            # adds demo data
    python manage.py seed_demo --reset    # wipes the demo user's workouts first
"""

import random
from datetime import timedelta

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from recognition.models import InferenceLog
from workouts.models import Exercise, FeedbackEvent, WorkoutRecord, WorkoutSession

DEMO_USER = ("demo", "GymDemo#2026", "Demo User")
ADMIN_USER = ("admin", "GymAdmin#2026", "Admin")

ISSUES = {
    "squat": ("chest_up", "Keep your chest up — you're leaning too far forward."),
    "lunge": ("torso_upright", "Keep your torso upright."),
    "push_up": ("hips_line", "Keep your hips in line with your shoulders."),
    "bicep_curl": ("elbows_pinned", "Keep your elbows pinned to your sides."),
    "shoulder_press": ("no_lean_back", "Avoid arching / leaning back while pressing."),
    "jumping_jack": ("partial_rep", "Partial rep — bring your arms all the way overhead."),
}


class Command(BaseCommand):
    help = "Create demo user, admin user and sample workout history."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="Delete the demo user's existing workouts first.")
        parser.add_argument("--days", type=int, default=28)

    def handle(self, *args, reset, days, **opts):
        demo = self._user(*DEMO_USER)
        admin = self._user(*ADMIN_USER, staff=True)
        demo.profile.weight_kg = 72
        demo.profile.height_cm = 175
        demo.profile.weekly_target_sessions = 4
        demo.profile.save()
        if reset:
            WorkoutSession.objects.filter(user=demo).delete()

        rng = random.Random(42)
        exercises = list(Exercise.objects.filter(is_active=True))
        now = timezone.now()
        created = 0
        for day in range(days, 0, -1):
            if rng.random() > 0.55:
                continue
            start = (now - timedelta(days=day)).replace(hour=rng.choice([7, 8, 18, 19]), minute=rng.randint(0, 59))
            created += self._session(demo, start, rng.sample(exercises, rng.randint(2, 4)), rng, skill=1 - day / (days * 1.6))
        self.stdout.write(self.style.SUCCESS(
            f"Created {created} sessions.\n"
            f"  Demo login : {DEMO_USER[0]} / {DEMO_USER[1]}\n"
            f"  Admin login: {ADMIN_USER[0]} / {ADMIN_USER[1]} (staff — can see Reports and /admin)"
        ))

    def _user(self, username, password, name, staff=False):
        user, created = User.objects.get_or_create(
            username=username, defaults={"first_name": name, "email": f"{username}@gymrec.local"}
        )
        if created:
            user.set_password(password)
            user.is_staff = user.is_superuser = staff
            user.save()
        return user

    def _session(self, user, start, exercises, rng, skill):
        session = WorkoutSession.objects.create(user=user, status=WorkoutSession.Status.COMPLETED)
        t = start
        total = 0.0
        for ex in exercises:
            reps = rng.randint(8, 20)
            good = min(reps, round(reps * min(0.98, skill + rng.uniform(-0.1, 0.1))))
            secs = reps * rng.uniform(2.4, 3.4)
            rec = WorkoutRecord.objects.create(
                session=session, exercise=ex, repetitions=reps, good_form_reps=max(0, good),
                avg_confidence=rng.uniform(0.78, 0.95), confidence_samples=int(secs * 6),
                started_at=t, ended_at=t + timedelta(seconds=secs),
            )
            code, msg = ISSUES.get(ex.slug, ("form", "Watch your form."))
            events = [FeedbackEvent(session=session, record=rec, level="warning", code=code, message=msg)
                      for _ in range(reps - rec.good_form_reps)]
            events.append(FeedbackEvent(session=session, record=rec, level="good", code="good_rep",
                                        message=f"{ex.name}: {rec.good_form_reps} reps with good form."))
            FeedbackEvent.objects.bulk_create(events)
            InferenceLog.objects.bulk_create(
                InferenceLog(session=session, predicted_exercise=ex.slug, confidence=rec.avg_confidence,
                             latency_ms=rng.uniform(4, 18), simulated=True)
                for _ in range(10)
            )
            t += timedelta(seconds=secs + rng.uniform(40, 90))
            total += secs
        WorkoutSession.objects.filter(pk=session.pk).update(
            created_at=start, started_at=start, ended_at=t, active_seconds=total,
            frames_processed=int(total * 6),
        )
        FeedbackEvent.objects.filter(session=session).update(created_at=start)
        InferenceLog.objects.filter(session=session).update(created_at=start)
        return 1
