#!/bin/bash
# Night 2 queue (18 Oct 2026) for the sld2quote symbol detector, synthetic tiles only. Two runs on one recipe:
#   run 1, rtdetr_r18_plain: plain r18vd on data/coco with no rare sampling and the whole ten-pass schedule, the control we lacked.
#     rtdetr_r18_rare is measured against it on the same data, and rtdetr_r18_full early-stopped at pass 6, so this is also the
#     proper full-schedule candidate.
#   run 2, rtdetr_r18_fonts: the same recipe on data/coco_fonts, a wider font set built by another engineer (held-out fonts Courier New
#     and Tahoma plus labels below, its own val_unseen and test_unseen). It starts once data/coco_fonts/manifest.json exists, polled
#     every minute and given up with a logged error after 3 hours.
# Recipe: r18vd from the hub, 10 passes at batch 16, lr 1e-4 (backbone 1e-5), warm-up 300, patience 0 so the cosine schedule completes,
# the best pass chosen by the pooled mean per-class F1 at the operating threshold on val and val_unseen with NMS, every pass kept,
# seed 0, backstop 420 min. Afterwards: each run's every split on its own export; each run's last pass on test_unseen (the fully
# annealed candidate beside the F1 pick, for the drawing-level check), rtdetr_r18_rare's last pass too; run 2 also on data/coco's val
# and test_unseen (comparable with the runs trained there, on the same tiles); then weights/night2_summary.md over rtdetr_r18_full,
# rtdetr_r18_rare, rtdetr_r18_plain and rtdetr_r18_fonts.
# Expected with MPS to itself: about 22 min per pass with its val (7.3 images/s on 9,176 tiles), so about 3.8 hours per run on a
# data/coco-sized export, about 8 hours for the queue with its evaluations; the summary gives the measured minutes per pass.
#   scripts/train_night2.sh           the queue, outputs under weights/
#   scripts/train_night2.sh --smoke   the same steps end to end in a few minutes (16 tiles per split, 1 pass per run) under
#                                     weights/night2_smoke_<time>; run 2 uses data/coco when data/coco_fonts is not there yet
# Neither starts over an existing run directory, so the logs of two runs never mix. A failed step is logged and the queue goes on.
set -u
[ -z "${CAFFEINATED:-}" ] && CAFFEINATED=1 exec caffeinate -is "$0" "$@"  # the Mac stays awake until the queue ends
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1  # aten::grid_sampler_2d_backward has no MPS kernel in torch 2.5.1 and runs on the CPU
OUT=weights; CAP=10; LIMIT=""; SMOKE=""; FONTS=data/coco_fonts; WAIT_MIN=180
if [ "${1:-}" = "--smoke" ]; then OUT=weights/night2_smoke_$(date +%Y%m%d-%H%M%S); CAP=1; LIMIT="--limit 16"; SMOKE=1; fi
R1=$OUT/rtdetr_r18_plain; R2=$OUT/rtdetr_r18_fonts
for d in "$R1" "$R2"; do
  if [ -e "$d" ]; then echo "$d exists: move it aside first"; exit 1; fi
done
mkdir -p "$OUT"; exec > >(tee -a "$OUT/night2_queue.log") 2>&1
TD="python3 -m sld2quote.train_detector"
RECIPE="--model PekingU/rtdetr_r18vd --epochs $CAP --batch 16 --lr 1e-4 --backbone-lr 1e-5 --weight-decay 1e-4 --warmup-steps 300 --grad-clip 0.1
  --patience 0 --select f1 --keep-passes --eval-every 1 --nms-iou 0.5 --device mps --seed 0 --time-budget-min 420"
step() { echo "$(date '+%F %T') start: $*"; "$@"; local s=$?; echo "$(date '+%F %T') exit $s: $*"; }
evals() {  # every split of a run on its export, then its last kept pass on test_unseen
  local run=$1 data=$2 s last
  if [ ! -f "$run/metrics.json" ]; then echo "$(date '+%F %T') $run wrote no metrics.json: its evaluations are skipped"; return; fi
  for s in val val_unseen test_clean test_scan test_unseen; do
    [ -d "$data/$s" ] && step $TD eval --weights "$run" --data "$data" --split $s --nms-iou 0.5 --device mps $LIMIT
  done
  last="$run/pass-$(ls -d "$run"/pass-* | sed 's/.*pass-//' | sort -n | tail -1)"  # by pass number, whatever the directory names hold
  step $TD eval --weights "$last" --data "$data" --split test_unseen --nms-iou 0.5 --device mps --out "$run/eval-$(basename "$last")-test_unseen.json" $LIMIT
}

step $TD train --data data/coco --out "$R1" $RECIPE $LIMIT
if [ -n "$SMOKE" ] && [ ! -f "$FONTS/manifest.json" ]; then echo "smoke: $FONTS is not there yet, so run 2 uses data/coco"; FONTS=data/coco; fi
waited=0
until [ -f "$FONTS/manifest.json" ] || [ $waited -ge $WAIT_MIN ]; do sleep 60; waited=$((waited + 1)); done
if [ -f "$FONTS/manifest.json" ]; then
  echo "$(date '+%F %T') $FONTS/manifest.json is there, after $waited min of waiting"
  step $TD train --data "$FONTS" --out "$R2" $RECIPE $LIMIT
else
  echo "$(date '+%F %T') ERROR: $FONTS/manifest.json did not appear within $WAIT_MIN min, so run 2 is not trained"
fi
evals "$R1" data/coco
evals "$R2" "$FONTS"
if [ -f "$R2/metrics.json" ]; then  # run 2 on the old export too, comparable with the runs trained there
  for s in val test_unseen; do step $TD eval --weights "$R2" --data data/coco --split $s --nms-iou 0.5 --device mps --out "$R2/eval-coco-$s.json" $LIMIT; done
fi
if [ -z "$SMOKE" ] && [ ! -f weights/rtdetr_r18_rare/eval-pass-10-test_unseen.json ]; then  # the rare run's annealed candidate, once
  step $TD eval --weights weights/rtdetr_r18_rare/pass-10 --data data/coco --split test_unseen --nms-iou 0.5 --device mps \
    --out weights/rtdetr_r18_rare/eval-pass-10-test_unseen.json
fi
step $TD summary --runs "weights/rtdetr_r18_full,weights/rtdetr_r18_rare,$R1,$R2" --out "$OUT/night2_summary.md"
echo "$(date '+%F %T') queue done: $OUT/night2_summary.md"
