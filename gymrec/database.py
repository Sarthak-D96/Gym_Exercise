"""
DATABASES["default"] from a connection URL (Neon writes DATABASE_URL into .env / .env.local).

Neon best practice for a long-running Django server (gunicorn / runserver):
  * psycopg 3 (already the project's driver) over TCP, with the URL's sslmode /
    channel_binding passed through (Neon requires TLS);
  * the pooled endpoint ("-pooler" in the host) is fine, but PgBouncer in transaction
    mode can't keep server-side cursors open, so DISABLE_SERVER_SIDE_CURSORS is set
    for pooled hosts;
  * short persistent connections (CONN_MAX_AGE) with health checks, because Neon
    closes idle connections when the compute scales to zero.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, unquote, urlparse

PASS_THROUGH = {"sslmode", "channel_binding", "sslrootcert", "connect_timeout", "options", "application_name"}


def database_from_url(url: str, conn_max_age: int = 60) -> dict:
    u = urlparse(url)
    if u.scheme not in ("postgres", "postgresql"):
        raise ValueError(f"Unsupported database URL scheme: {u.scheme!r}")
    query = dict(parse_qsl(u.query))
    pooled = "-pooler" in (u.hostname or "")
    db = {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": unquote(u.path.lstrip("/")) or "postgres",
        "USER": unquote(u.username or ""),
        "PASSWORD": unquote(u.password or ""),
        "HOST": u.hostname or "",
        "PORT": str(u.port or 5432),
        "CONN_MAX_AGE": conn_max_age,
        "CONN_HEALTH_CHECKS": True,
        "OPTIONS": {k: v for k, v in query.items() if k in PASS_THROUGH},
    }
    if pooled:
        db["DISABLE_SERVER_SIDE_CURSORS"] = True
    return db
