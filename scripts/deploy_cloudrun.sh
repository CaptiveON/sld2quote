#!/usr/bin/env bash
# Build the site image with Cloud Build and deploy it to Cloud Run as the public service "sld2quote", inside the free tier;
# idempotent, so a redeploy is the same command:   scripts/deploy_cloudrun.sh
# Billing is per request (--cpu-throttling): CPU is allocated only while a request is served and while the instance starts.
# The models load during start-up: /api/health answers 503 until they are warm, and the HTTP startup probe waits for its
# 200 (every 2 s, for up to 240 s), so the CPU stays on while they load and the instance takes traffic only when ready; a
# revision whose models fail to load never goes live. At most one instance, scaled to zero when idle; live runs are capped
# per UTC day (SLD2QUOTE_DAILY_RUNS). gen2: the default gen1 sandbox ran the detector and OCR about half as fast.
# Needs gcloud logged in as the owner and huggingface_hub logged in. The Hugging Face token goes into Secret Manager once,
# through stdin, never echoed; to rotate it, add a version to the secret hf-token. Every gcloud call names the project
# (gc below); nothing here changes gcloud's configuration. The upload is what .gcloudignore lets through (about 25 MB).
# After a successful deploy the registry keeps only the deployed image (the free tier stores 0.5 GB; one image is 0.6 GB).
set -euo pipefail
P=sld2quote-demo R=europe-west1
SA=sld2quote-run@$P.iam.gserviceaccount.com  # runtime identity: no project role, only read access to the one secret
cd "$(dirname "$0")/.."
TAG=$(git rev-parse --short HEAD)$([ -z "$(git status --porcelain)" ] || echo -dirty)
REPO=$R-docker.pkg.dev/$P/sld2quote IMAGE=$R-docker.pkg.dev/$P/sld2quote/site:$TAG
gc() { gcloud "$@" --project "$P" --quiet; }

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
  --execution-environment gen2 --cpu-throttling --cpu-boost \
  --startup-probe httpGet.path=/api/health,httpGet.port=7860,periodSeconds=2,timeoutSeconds=1,failureThreshold=120 \
  --port 7860 --memory 4Gi --cpu 2 --concurrency 1 --timeout 300 --min-instances 0 --max-instances 1 \
  --allow-unauthenticated --set-secrets HF_TOKEN=hf-token:latest \
  --set-env-vars SLD2QUOTE_MODEL_REPO=itsmoadnan/sld2quote-detector,SLD2QUOTE_OCR=rapidocr,SLD2QUOTE_WEIGHTS=/tmp/weights/rtdetr_r18_full,SLD2QUOTE_RUNS=/tmp/sld2quote-runs,OMP_NUM_THREADS=2,SLD2QUOTE_DAILY_RUNS=24

KEEP=$(gc artifacts docker images describe "$IMAGE" --format 'value(image_summary.digest)')
gc artifacts docker images list "$REPO" --format 'value(package,version)' | while read -r pkg digest; do
  [ "$digest" = "$KEEP" ] || gc artifacts docker images delete "$pkg@$digest" --delete-tags
done
gc run services describe sld2quote --region "$R" --format 'value(status.url)'
