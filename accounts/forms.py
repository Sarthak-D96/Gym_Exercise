from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Profile


class RegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=150, label="Name")
    email = forms.EmailField()

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "first_name", "email")

    def clean_email(self):
        email = self.cleaned_data["email"].lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email


class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("first_name", "email")
        labels = {"first_name": "Name"}


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ("height_cm", "weight_kg", "fitness_goal", "weekly_target_sessions")
        labels = {
            "height_cm": "Height (cm)",
            "weight_kg": "Weight (kg)",
            "weekly_target_sessions": "Weekly target (sessions)",
        }
