from django.contrib import admin
from django.urls import include, path

admin.site.site_header = "GymRec Administration"
admin.site.site_title = "GymRec Admin"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("reports/", include("reports.urls")),
    path("", include("workouts.urls")),
]
