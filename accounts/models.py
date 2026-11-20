from django.conf import settings
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class Profile(models.Model):
    """Extra user information (class diagram: User — name, email via auth.User)."""

    class Goal(models.TextChoices):
        STRENGTH = "strength", "Build strength"
        ENDURANCE = "endurance", "Improve endurance"
        WEIGHT_LOSS = "weight_loss", "Lose weight"
        GENERAL = "general", "General fitness"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    height_cm = models.PositiveSmallIntegerField(null=True, blank=True)
    weight_kg = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)
    fitness_goal = models.CharField(max_length=20, choices=Goal.choices, default=Goal.GENERAL)
    weekly_target_sessions = models.PositiveSmallIntegerField(default=3)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Profile<{self.user.username}>"


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.get_or_create(user=instance)
