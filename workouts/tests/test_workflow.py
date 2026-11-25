"""Integration / system tests mirroring the test cases TC01-TC06 from the presentation."""

import json

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from workouts.models import Exercise, InvalidTransition, WorkoutSession


class StateMachineTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("sam", password="pw-12345-xyz")

    def test_valid_flow(self):
        s = WorkoutSession.objects.create(user=self.user)
        for event, expected in [("begin", "active"), ("pause", "paused"), ("resume", "active"), ("end", "completed")]:
            s.transition(event)
            self.assertEqual(s.status, expected)
        self.assertIsNotNone(s.ended_at)

    def test_invalid_transition(self):
        s = WorkoutSession.objects.create(user=self.user)
        with self.assertRaises(InvalidTransition):
            s.transition("pause")


class WorkflowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("sam", email="sam@example.com", password="pw-12345-xyz")

    def post_json(self, url, data=None):
        return self.client.post(url, json.dumps(data or {}), content_type="application/json")

    def test_tc01_login_opens_dashboard(self):
        res = self.client.post(reverse("accounts:login"), {"username": "sam", "password": "pw-12345-xyz"}, follow=True)
        self.assertContains(res, "Start a workout")

    def test_register_creates_profile(self):
        res = self.client.post(reverse("accounts:register"), {
            "first_name": "Riya", "username": "riya", "email": "riya@example.com",
            "password1": "Str0ng-pass-123", "password2": "Str0ng-pass-123",
        })
        self.assertRedirects(res, reverse("workouts:dashboard"))
        self.assertTrue(User.objects.get(username="riya").profile)

    def test_api_requires_login(self):
        self.assertEqual(self.client.get(reverse("workouts:api_history")).status_code, 401)

    def test_tc02_to_tc06_full_workout(self):
        self.client.force_login(self.user)
        # TC02 start workout
        res = self.client.post(reverse("workouts:start"), {"exercise": "squat"})
        session = WorkoutSession.objects.get(user=self.user)
        self.assertRedirects(res, reverse("workouts:live", args=[session.pk]))

        frame_url = reverse("workouts:api_session_frame", args=[session.pk])
        # frames are rejected until the session is active
        self.assertEqual(self.post_json(frame_url, {"t": 0}).status_code, 409)
        self.post_json(reverse("workouts:api_session_transition", args=[session.pk, "begin"]))

        # TC03/TC04/TC05 recognition, rep counting, posture feedback (simulation mode)
        last = None
        for t in range(0, 20000, 120):
            res = self.post_json(frame_url, {"t": t})
            self.assertEqual(res.status_code, 200, res.content)
            last = res.json()
        self.assertEqual(last["exercise"], "squat")
        self.assertGreaterEqual(last["reps"], 4)
        self.assertTrue(session.feedback.exists())

        # TC06 history
        res = self.post_json(reverse("workouts:api_session_transition", args=[session.pk, "end"]))
        self.assertEqual(res.json()["status"], "completed")
        hist = self.client.get(reverse("workouts:api_history")).json()
        self.assertEqual(hist["by_exercise"][0]["exercise__slug"], "squat")
        self.assertContains(self.client.get(reverse("workouts:history")), "Squat")
        self.assertContains(self.client.get(reverse("workouts:session_detail", args=[session.pk])), "Feedback timeline")

    def test_other_users_session_is_hidden(self):
        other = User.objects.create_user("eve", password="pw-12345-xyz")
        s = WorkoutSession.objects.create(user=other)
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(reverse("workouts:live", args=[s.pk])).status_code, 404)
        self.assertEqual(self.post_json(reverse("workouts:api_session_frame", args=[s.pk]), {"t": 0}).status_code, 404)

    def test_seeded_exercises(self):
        # 6 prototype exercises + bench_press and pull_up (migration 0003)
        self.assertEqual(Exercise.objects.count(), 8)

    def test_reports_staff_only(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(reverse("reports:usage")).status_code, 302)
        self.user.is_staff = True
        self.user.save()
        self.assertEqual(self.client.get(reverse("reports:usage")).status_code, 200)
