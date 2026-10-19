#!/usr/bin/env bash
# Build the site image with Cloud Build and deploy it to Cloud Run as the public service "sld2quote", inside the free tier;
# idempotent, so a redeploy is the same command:   scripts/deploy_cloudrun.sh
# The service holds no secret: the detector's weights are baked into the image (the upload carries weights/$MODEL, see
# .gcloudignore), so nothing is downloaded at start-up. A model switch is two edits: MODEL below and the one weights/
# line in .gcloudignore (the build stops early if they disagree).
# Billing is per request (--cpu-throttling): CPU is allocated only while a request is served and while the instance starts.
# The models load during start-up: /api/health answers 503 until they are warm, and the HTTP startup probe waits for its
# 200 (every 2 s, for up to 240 s), so the CPU stays on while they load and the instance takes traffic only when ready; a
# revision whose models fail to load never goes live. At most one instance, scaled to zero when idle; live runs are capped
# per UTC day (SLD2QUOTE_DAILY_RUNS). gen2: the default gen1 sandbox ran the detector and OCR about half as fast.
# Needs gcloud logged in as the owner. Every gcloud call names the project (gc below); nothing here changes gcloud's
# configuration. After a successful deploy the registry keeps only the deployed image (the free tier stores 0.5 GB).
set -euo pipefail
P=sld2quote-demo R=europe-west1 MODEL=rtdetr_r18_full
PROBE=httpGet.path=/api/health,httpGet.port=7860,periodSeconds=2,timeoutSeconds=1,failureThreshold=120 CONC=1
# once site/app.py warms the models inside the first request (no start-up thread), the page's files can be served while a
# run or the warm-up is in flight:   PROBE=tcpSocket.port=7860,periodSeconds=2,timeoutSeconds=1,failureThreshold=120 CONC=8
SA=sld2quote-run@$P.iam.gserviceaccount.com  # runtime identity: no project role and no secret, it only runs the container
cd "$(dirname "$0")/.."
TAG=$(git rev-parse --short HEAD)$([ -z "$(git status --porcelain)" ] || echo -dirty)
REPO=$R-docker.pkg.dev/$P/sld2quote IMAGE=$R-docker.pkg.dev/$P/sld2quote/site:$TAG
gc() { gcloud "$@" --project "$P" --quiet; }

gc meta list-files-for-upload . | grep -qx "weights/$MODEL/model.safetensors" ||
  { echo "weights/$MODEL/model.safetensors is not in the upload: name weights/$MODEL on its line in .gcloudignore" >&2; exit 1; }
gc artifacts repositories describe sld2quote --location "$R" >/dev/null 2>&1 ||
  gc artifacts repositories create sld2quote --repository-format docker --location "$R"
gc iam service-accounts describe "$SA" >/dev/null 2>&1 ||
  gc iam service-accounts create sld2quote-run --display-name "sld2quote Cloud Run runtime"

gc builds submit --region "$R" --config site/cloudbuild.yaml --substitutions "_IMAGE=$IMAGE" .
gc run deploy sld2quote --image "$IMAGE" --region "$R" --service-account "$SA" \
  --execution-environment gen2 --cpu-throttling --cpu-boost --startup-probe "$PROBE" --concurrency "$CONC" \
  --port 7860 --memory 4Gi --cpu 2 --timeout 300 --min-instances 0 --max-instances 1 --allow-unauthenticated --clear-secrets \
  --set-env-vars SLD2QUOTE_WEIGHTS=/home/user/app/weights/$MODEL,SLD2QUOTE_OCR=rapidocr,SLD2QUOTE_RUNS=/tmp/sld2quote-runs,OMP_NUM_THREADS=2,SLD2QUOTE_DAILY_RUNS=24

KEEP=$(gc artifacts docker images describe "$IMAGE" --format 'value(image_summary.digest)')
gc artifacts docker images list "$REPO" --format 'value(package,version)' | while read -r pkg digest; do
  [ "$digest" = "$KEEP" ] || gc artifacts docker images delete "$pkg@$digest" --delete-tags
done
gc run services describe sld2quote --region "$R" --format 'value(status.url)'
