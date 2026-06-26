#!/usr/bin/env bash
# Cloud Run one-shot deploy for the Controls Agent Streamlit app.
#
# Prerequisites:
#   gcloud auth login && gcloud auth configure-docker
#   export PROJECT_ID=your-gcp-project-id
#
# Usage:
#   bash deploy.sh
#   bash deploy.sh --region us-central1   (optional override)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project}"
REGION="${1:-us-central1}"
SERVICE="controls-agent"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "==> Building image: ${IMAGE}"
docker build --platform linux/amd64 -t "${IMAGE}" .

echo "==> Pushing image to Container Registry"
docker push "${IMAGE}"

echo "==> Deploying to Cloud Run (region: ${REGION})"
gcloud run deploy "${SERVICE}" \
  --image        "${IMAGE}" \
  --platform     managed \
  --region       "${REGION}" \
  --port         8080 \
  --memory       1Gi \
  --cpu          1 \
  --timeout      300 \
  --max-instances 3 \
  --allow-unauthenticated \
  --set-env-vars "DB_PATH=/app/controls_agent.db,DATA_PATH=/app/data/transactions.csv"

URL=$(gcloud run services describe "${SERVICE}" \
  --platform managed --region "${REGION}" \
  --format "value(status.url)")

echo ""
echo "==> Deployed: ${URL}"
