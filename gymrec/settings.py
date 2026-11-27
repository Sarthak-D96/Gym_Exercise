"""
Django settings for the Gym Exercise Recognition System (UCS503 prototype).

Configuration is read from environment variables (optionally from a `.env`
file in the project root) so the same code runs locally and in deployment.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line) so we avoid an extra dependency."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env.local")  # written by `neon deploy` / `neon env pull` when there is no .env
_load_dotenv(BASE_DIR / ".env")


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    return env(key, str(default)).lower() in {"1", "true", "yes", "on"}


SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-insecure-change-me")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [h for h in env("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h]
# Hugging Face sets SPACE_ID inside a Space; used below for proxy / iframe settings.
ON_HF_SPACE = bool(env("SPACE_ID"))

if not DEBUG and SECRET_KEY == "dev-insecure-change-me":
    raise RuntimeError("Set DJANGO_SECRET_KEY when DJANGO_DEBUG=false.")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    # Project apps
    "accounts",
    "recognition",
    "workouts",
    "reports",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # serves /static/ in production (DEBUG=false)
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "gymrec.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "gymrec.wsgi.application"

# --- Database: PostgreSQL ----------------------------------------------------
DIRECT_CONNECTION_COMMANDS = {"migrate", "sqlmigrate", "dbshell", "flush", "loaddata", "dumpdata",
                              "createcachetable", "test"}
# DATABASE_URL (Neon: written by `neon deploy`) wins; otherwise the POSTGRES_* settings
# below (local PostgreSQL / docker-compose). See gymrec/database.py.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB", "gymrec"),
        "USER": env("POSTGRES_USER", ""),
        "PASSWORD": env("POSTGRES_PASSWORD", ""),
        "HOST": env("POSTGRES_HOST", "localhost"),
        "PORT": env("POSTGRES_PORT", "5432"),
        "CONN_MAX_AGE": 60,
    }
}
if env("DATABASE_URL"):
    import sys

    from gymrec.database import database_from_url

    # Neon: the app uses the pooled URL; schema migrations and session-level admin commands
    # use the direct one (PgBouncer in transaction mode can't run them reliably).
    _direct = env("DATABASE_URL_UNPOOLED")
    _cmd = sys.argv[1] if len(sys.argv) > 1 and sys.argv[0].endswith("manage.py") else ""
    _use_direct = _direct and _cmd in DIRECT_CONNECTION_COMMANDS
    DATABASES["default"] = database_from_url(_direct if _use_direct else env("DATABASE_URL"))

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "workouts:dashboard"
LOGOUT_REDIRECT_URL = "accounts:login"

LANGUAGE_CODE = "en-us"
TIME_ZONE = env("DJANGO_TIME_ZONE", "Asia/Kolkata")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    # compressed copies made by `collectstatic`; no manifest, so tests and runserver work without it
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

# --- Production / behind an HTTPS proxy (Hugging Face Spaces, Render, ...) ---------
# e.g. DJANGO_CSRF_TRUSTED_ORIGINS=https://*.hf.space
CSRF_TRUSTED_ORIGINS = [o for o in env("DJANGO_CSRF_TRUSTED_ORIGINS", "https://*.hf.space").split(",") if o]
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
if ON_HF_SPACE:
    # The Space page on huggingface.co shows the app in an iframe: allow that frame and send
    # the login cookies there. (Opening https://<user>-<space>.hf.space directly works too.)
    SESSION_COOKIE_SAMESITE = CSRF_COOKIE_SAMESITE = "None"
    MIDDLEWARE.remove("django.middleware.clickjacking.XFrameOptionsMiddleware")
    MIDDLEWARE.append("gymrec.middleware.FrameAncestorsMiddleware")

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Frames are posted as base64 JPEG data URLs; allow a few hundred KB per request.
DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024

# --- Recognition pipeline ----------------------------------------------------
RECOGNITION = {
    # Directory that holds trained model files (D2 Model Storage).
    "MODEL_DIR": BASE_DIR / "ml_models",
    # Frames are resized to this square size before pose detection (4.1).
    "INPUT_SIZE": int(env("RECOGNITION_INPUT_SIZE", "640")),
    # Sliding window of predictions used by "Validate & Refine" (4.4).
    "SMOOTHING_WINDOW": int(env("RECOGNITION_SMOOTHING_WINDOW", "8")),
    # Below this smoothed confidence, the exercise is reported as "uncertain".
    "MIN_CONFIDENCE": float(env("RECOGNITION_MIN_CONFIDENCE", "0.6")),
    # Allow frames without image data (no camera) — the dummy pose detector
    # synthesises keypoints so the full flow can be demoed without a webcam.
    "ALLOW_SIMULATION": env_bool("RECOGNITION_ALLOW_SIMULATION", True),
    # Count a rep only if its form has no major fault (posture.py). False: count every
    # full rep and only report form problems.
    "COUNT_ONLY_GOOD_FORM": env_bool("RECOGNITION_COUNT_ONLY_GOOD_FORM", True),
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"recognition": {"handlers": ["console"], "level": "INFO"}},
}
