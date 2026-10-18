#!/bin/bash
# Night 2b queue (18 Oct 2026) for the sld2quote symbol detector, synthetic tiles only: night 2's two recipes again with one change,
# a weight average by the RT-DETR authors' rule (lyuwenyu/RT-DETR, src/optim/ema.py: d = decay * (1 - exp(-updates / warm-up)),
# v = d * v + (1 - d) * w after every optimiser step), so that EMA is the only difference from rtdetr_r18_plain and rtdetr_r18_fonts.
# The window is scaled to our run length, not the authors' step count: they use decay 0.9999 with 2000 warm-up updates, tuned to
# 72 epochs of hundreds of thousands of steps, which on our ten passes of about 5,700 steps would end at d of about 0.943, an
# average over about 18 steps. Here decay 0.999 with 200 warm-up updates gives a time constant of about 1,000 steps, roughly 1.7
# passes, which is what the pass-to-pass swings in per-type recall span (decided 18 Oct 2026):
#   run 3, rtdetr_r18_plain_ema: the plain recipe on data/coco with --ema 0.999 --ema-warmup 200;
#   run 4, rtdetr_r18_fonts_ema: the same on data/coco_fonts (the regenerated fonts export).
# It starts once weights/night2_summary.md exists, which night 2 writes last (polled every minute, given up with a logged error
# after 12 hours), so the two queues never share the machine. Recipe as night 2: r18vd from the hub, 10 passes at batch 16, lr 1e-4
# (backbone 1e-5), warm-up 300, patience 0, the best pass by the pooled mean per-class F1 on val and val_unseen with NMS, every
# pass kept, seed 0, backstop 420 min; with the average on, what is evaluated, chosen, kept and finished with is the average.
# Afterwards, as night 2: each run's every split on its own export and its last pass on test_unseen, run 4 also on data/coco's val
# and test_unseen, then weights/night2b_summary.md over rtdetr_r18_full, rtdetr_r18_rare, rtdetr_r18_plain, rtdetr_r18_fonts,
# rtdetr_r18_plain_ema and rtdetr_r18_fonts_ema, with the F1 pick and the last pass as candidates per run.
# Expected with MPS to itself: about 3.8 hours per run, about 8 hours for the queue after night 2 ends.
#   scripts/train_night2b.sh           the queue, outputs under weights/
#   scripts/train_night2b.sh --smoke   the same steps end to end in a few minutes (16 tiles per split, 1 pass per run, no wait for
#                                      night 2) under weights/night2b_smoke_<time>, on the cpu so it never competes with a running queue
#                                      for MPS; run 4 uses data/coco when data/coco_fonts is not there
# Neither starts over an existing run directory, so the logs of two runs never mix. A failed step is logged and the queue goes on.
set -u
[ -z "${CAFFEINATED:-}" ] && CAFFEINATED=1 exec caffeinate -is "$0" "$@"  # the Mac stays awake until the queue ends
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1  # aten::grid_sampler_2d_backward has no MPS kernel in torch 2.5.1 and runs on the CPU
OUT=weights; CAP=10; LIMIT=""; SMOKE=""; DEVICE=mps; FONTS=data/coco_fonts; AFTER=weights/night2_summary.md; WAIT_AFTER_MIN=720; WAIT_FONTS_MIN=180
if [ "${1:-}" = "--smoke" ]; then OUT=weights/night2b_smoke_$(date +%Y%m%d-%H%M%S); CAP=1; LIMIT="--limit 16"; SMOKE=1; DEVICE=cpu; fi
R3=$OUT/rtdetr_r18_plain_ema; R4=$OUT/rtdetr_r18_fonts_ema
for d in "$R3" "$R4"; do
  if [ -e "$d" ]; then echo "$d exists: move it aside first"; exit 1; fi
done
mkdir -p "$OUT"; exec > >(tee -a "$OUT/night2b_queue.log") 2>&1
TD="python3 -m sld2quote.train_detector"
RECIPE="--model PekingU/rtdetr_r18vd --epochs $CAP --batch 16 --lr 1e-4 --backbone-lr 1e-5 --weight-decay 1e-4 --warmup-steps 300 --grad-clip 0.1
  --patience 0 --select f1 --keep-passes --eval-every 1 --nms-iou 0.5 --device $DEVICE --seed 0 --time-budget-min 420 --ema 0.999 --ema-warmup 200"
step() { echo "$(date '+%F %T') start: $*"; "$@"; local s=$?; echo "$(date '+%F %T') exit $s: $*"; }
wait_for() {  # wait_for <file> <minutes>: poll every minute; 0 when it is there
  local waited=0
  until [ -f "$1" ] || [ $waited -ge "$2" ]; do sleep 60; waited=$((waited + 1)); done
  if [ -f "$1" ]; then echo "$(date '+%F %T') $1 is there, after $waited min of waiting"; return 0; fi
  echo "$(date '+%F %T') ERROR: $1 did not appear within $2 min"; return 1
}
evals() {  # every split of a run on its export, then its last kept pass on test_unseen
  local run=$1 data=$2 s last
  if [ ! -f "$run/metrics.json" ]; then echo "$(date '+%F %T') $run wrote no metrics.json: its evaluations are skipped"; return; fi
  for s in val val_unseen test_clean test_scan test_unseen; do
    [ -d "$data/$s" ] && step $TD eval --weights "$run" --data "$data" --split $s --nms-iou 0.5 --device $DEVICE $LIMIT
  done
  last="$run/pass-$(ls -d "$run"/pass-* | sed 's/.*pass-//' | sort -n | tail -1)"  # by pass number, whatever the directory names hold
  step $TD eval --weights "$last" --data "$data" --split test_unseen --nms-iou 0.5 --device $DEVICE --out "$run/eval-$(basename "$last")-test_unseen.json" $LIMIT
}

if [ -n "$SMOKE" ]; then echo "smoke: not waiting for $AFTER"
else echo "$(date '+%F %T') waiting for $AFTER, polled every minute for up to $WAIT_AFTER_MIN min"; fi
if [ -n "$SMOKE" ]; then :
elif ! wait_for "$AFTER" $WAIT_AFTER_MIN; then echo "$(date '+%F %T') night 2 did not finish in time, so night 2b does not start"; exit 1; fi
step $TD train --data data/coco --out "$R3" $RECIPE $LIMIT
if [ -n "$SMOKE" ] && [ ! -f "$FONTS/manifest.json" ]; then echo "smoke: $FONTS is not there, so run 4 uses data/coco"; FONTS=data/coco; fi
if wait_for "$FONTS/manifest.json" $WAIT_FONTS_MIN; then
  step $TD train --data "$FONTS" --out "$R4" $RECIPE $LIMIT
else
  echo "$(date '+%F %T') ERROR: run 4 is not trained"
fi
evals "$R3" data/coco
evals "$R4" "$FONTS"
if [ -f "$R4/metrics.json" ]; then  # run 4 on the old export too, comparable with the runs trained there
  for s in val test_unseen; do step $TD eval --weights "$R4" --data data/coco --split $s --nms-iou 0.5 --device $DEVICE --out "$R4/eval-coco-$s.json" $LIMIT; done
fi
step $TD summary --runs "weights/rtdetr_r18_full,weights/rtdetr_r18_rare,weights/rtdetr_r18_plain,weights/rtdetr_r18_fonts,$R3,$R4" --out "$OUT/night2b_summary.md"
echo "$(date '+%F %T') queue done: $OUT/night2b_summary.md"
