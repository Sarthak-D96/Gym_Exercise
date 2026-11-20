from django.urls import path

from . import views

app_name = "workouts"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("workout/start/", views.start_workout, name="start"),
    path("workout/<int:pk>/", views.live_workout, name="live"),
    path("history/", views.history, name="history"),
    path("history/<int:pk>/", views.session_detail, name="session_detail"),
    path("history/<int:pk>/delete/", views.delete_session, name="delete_session"),
    # JSON API
    path("api/exercises/", views.api_exercises, name="api_exercises"),
    path("api/history/", views.api_history, name="api_history"),
    path("api/sessions/<int:pk>/", views.api_session, name="api_session"),
    path("api/sessions/<int:pk>/frames/", views.api_session_frame, name="api_session_frame"),
    path("api/sessions/<int:pk>/<str:event>/", views.api_session_transition, name="api_session_transition"),
]
