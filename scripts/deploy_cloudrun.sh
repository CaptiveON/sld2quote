#!/usr/bin/env bash
# Build the site image with Cloud Build and deploy it to Cloud Run as the public service "sld2quote"; idempotent, so a
# redeploy is the same command:   scripts/deploy_cloudrun.sh
# Needs gcloud logged in as the owner and huggingface_hub logged in. The Hugging Face token goes into Secret Manager once,
# through stdin, never echoed; to rotate it, add a version to the secret hf-token. Every gcloud call names the project
# (gc below); nothing here changes gcloud's configuration. The upload is what .gcloudignore lets through (about 25 MB).
set -euo pipefail
P=sld2quote-demo R=europe-west1
SA=sld2quote-run@$P.iam.gserviceaccount.com  # runtime identity: no project role, only read access to the one secret
cd "$(dirname "$0")/.."
TAG=$(git rev-parse --short HEAD)$([ -z "$(git status --porcelain)" ] || echo -dirty)
IMAGE=$R-docker.pkg.dev/$P/sld2quote/site:$TAG
gc() { gcloud "$@" --project "$P" --quiet; }
# gen2: the default gen1 sandbox ran the detector and OCR about ten times slower than a laptop core. --no-cpu-throttling:
# the app loads its models in a background thread at start-up, which request-based CPU starves (171 s to warm); this bills
# the instance while it is up (it scales to zero after about 15 idle minutes, at most 1 instance)

gc artifacts repositories describe sld2quote --location "$R" >/dev/null 2>&1 ||
  gc artifacts repositories create sld2quote --repository-format docker --location "$R"
gc iam service-accounts describe "$SA" >/dev/null 2>&1 ||
  gc iam service-accounts create sld2quote-run --display-name "sld2quote Cloud Run runtime"
if ! gc secrets describe hf-token >/dev/null 2>&1; then
  gc secrets create hf-token --replication-policy automatic
  python3 -c 'import sys; from huggingface_hub import get_token; sys.stdout.write(get_token())' | gc secrets versions add hf-token --data-file=-
fi
gc secrets add-iam-policy-binding hf-token --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor >/dev/null

gc builds submit --region "$R" --config site/cloudbuild.yaml --substitutions "_IMAGE=$IMAGE" .
gc run deploy sld2quote --image "$IMAGE" --region "$R" --service-account "$SA" \
  --execution-environment gen2 --no-cpu-throttling \
  --port 7860 --memory 4Gi --cpu 2 --cpu-boost --concurrency 1 --timeout 300 --min-instances 0 --max-instances 1 \
  --allow-unauthenticated --set-secrets HF_TOKEN=hf-token:latest \
  --set-env-vars SLD2QUOTE_MODEL_REPO=itsmoadnan/sld2quote-detector,SLD2QUOTE_OCR=rapidocr,SLD2QUOTE_WEIGHTS=/tmp/weights/rtdetr_r18_full,SLD2QUOTE_RUNS=/tmp/sld2quote-runs,OMP_NUM_THREADS=2
gc run services describe sld2quote --region "$R" --format 'value(status.url)'
