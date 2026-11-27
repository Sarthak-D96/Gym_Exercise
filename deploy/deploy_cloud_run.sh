#!/bin/sh
# Deploy (or update) the website on Google Cloud Run. See DEPLOY.md, "Google Cloud Run".
#
#   deploy/deploy_cloud_run.sh               # region us-central1, service "gymrec"
#   REGION=us-east4 deploy/deploy_cloud_run.sh
#
# Cloud Build builds the Dockerfile in Google's network (only the source, about 30 MB, is
# uploaded), then Cloud Run runs it. Settings come from deploy/cloudrun.env.yaml.
set -e
SERVICE=${SERVICE:-gymrec}
REGION=${REGION:-us-central1}
ENV_FILE=deploy/cloudrun.env.yaml

command -v gcloud >/dev/null || { echo "Install the Google Cloud CLI: brew install --cask google-cloud-sdk"; exit 1; }
PROJECT=$(gcloud config get-value project 2>/dev/null)
[ -n "$PROJECT" ] || { echo "No project selected: gcloud config set project <project-id>"; exit 1; }
[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE: copy deploy/cloudrun.env.example.yaml and fill it in."; exit 1; }

echo "Deploying '$SERVICE' to project $PROJECT, region $REGION ..."
# 2 vCPU / 2 GiB: the models peak at about 0.6 GB. Scale to zero when unused (free tier),
# at most 2 instances (each loads the models once), CPU boost to start faster after sleeping.
gcloud run deploy "$SERVICE" \
    --source . \
    --region "$REGION" \
    --port 7860 \
    --cpu 2 --memory 2Gi \
    --min-instances 0 --max-instances 2 \
    --concurrency 8 --timeout 120 \
    --cpu-boost \
    --env-vars-file "$ENV_FILE" \
    --allow-unauthenticated

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo "Live at: $URL"
