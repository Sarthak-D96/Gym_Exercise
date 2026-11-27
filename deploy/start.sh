#!/bin/sh
# Container start: apply any new migrations (direct Neon connection, see settings.py), then serve.
set -e
python manage.py migrate --noinput
# One worker keeps a single copy of PyTorch + YOLO + the classifiers in memory; threads serve
# requests concurrently (the models are loaded once and shared).
exec gunicorn gymrec.wsgi:application \
    --bind "0.0.0.0:${PORT:-7860}" \
    --workers 1 --threads "${GUNICORN_THREADS:-4}" \
    --timeout 120 --access-logfile - --error-logfile -
