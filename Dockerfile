# GymRec website + trained models, for a Hugging Face Docker Space (or any Docker host).
# See DEPLOY.md. Secrets (DATABASE_URL, DJANGO_SECRET_KEY, ...) come from the host's
# environment settings, never from this image.
FROM python:3.12-slim

# OpenCV (used by ultralytics) needs these shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=7860 \
    DJANGO_DEBUG=false \
    RECOGNITION_ALLOW_SIMULATION=false \
    RECOGNITION_INPUT_SIZE=640
WORKDIR /app

# CPU-only PyTorch (the version the model was trained with), then the rest
COPY requirements.txt requirements-deploy.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=app:app . .
# collectstatic doesn't touch the database; the dummy key is only for this build step
RUN DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput \
    && chmod +x deploy/start.sh && chown -R app:app /app

USER app
EXPOSE 7860
CMD ["deploy/start.sh"]
