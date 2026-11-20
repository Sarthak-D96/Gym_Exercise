from django.urls import path

from . import views

app_name = "reports"

urlpatterns = [
    path("", views.usage_report, name="usage"),
    path("data/", views.usage_data, name="usage_data"),
]
