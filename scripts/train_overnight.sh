#!/bin/bash
# Overnight queue for the sld2quote symbol detector, on synthetic tiles only (data/coco, cut from generated sheets): r18vd, then
# r50vd, each fine-tuned from the Hugging Face hub weights into its own weights directory (train.log, history.json, checkpoint-last,
# checkpoint-best, metrics.json and card.md there), then eval of every split for both, then weights/overnight_summary.md.
#
# Protocol (17 Oct 2026, the project owner): each model trains up to a cap of passes with the cosine schedule sized to the cap, val
# after every pass, and early stopping: the run stops after 3 passes in a row that do not raise val mAP@0.5 with NMS by 0.002, and
# the final weights, evaluation, metrics and card are those of the best pass. --time-budget-min is only a backstop; the cap is the
# real limit. Times per pass were measured on this Mac (M2 Max, MPS, torch 2.5.1) as a 50-step rate in the training loop:
#   r18vd, batch 16, cap 10: 6.41 images/s, 573 steps per pass, about 24.7 min per pass with its val (the stopped first attempt ran
#     at 7.2 to 7.8 images/s, about 21 min per pass); at most about 247 min, backstop 330 min.
#   r50vd, batch 8 (batch 16 took 22.3 GiB on the GPU and thrashed at 0.1 images/s; batch 8 takes 12.7 GiB), cap 8: 2.99 images/s,
#     1147 steps per pass, about 52.7 min per pass with its val; at most about 422 min, backstop 480 min.
# With no early stop the queue takes about 11.5 hours including the evaluations; each early stop shortens it.
#
# More passes later, beyond a cap: a new run from the trained weights with its own fresh, shorter and gentler schedule, into a new
# directory, for example
#   PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_full \
#     --out weights/rtdetr_r18_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 \
#     --nms-iou 0.5 --device mps --seed 0
# The trained 14-class head is kept as it is; weights/overnight_summary.md ends with the exact line for each model and the minutes per
# pass measured in its run.
#
#   scripts/train_overnight.sh           the queue, outputs under weights/
#   scripts/train_overnight.sh --smoke   the same steps end to end in a few minutes (16 tiles per split, 1 pass per model) plus a
#                                        one-pass continuation from the smoke r18vd weights, under weights/overnight_smoke_<time>
# Neither starts over an existing run directory, so the logs of two runs never mix. A failed step is logged and the queue goes on.
set -u
[ -z "${CAFFEINATED:-}" ] && CAFFEINATED=1 exec caffeinate -is "$0" "$@"  # the Mac stays awake until the queue ends
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1  # aten::grid_sampler_2d_backward has no MPS kernel in torch 2.5.1 and runs on the CPU
OUT=weights; R18_CAP=10; R50_CAP=8; LIMIT=""; SMOKE=""
if [ "${1:-}" = "--smoke" ]; then OUT=weights/overnight_smoke_$(date +%Y%m%d-%H%M%S); R18_CAP=1; R50_CAP=1; LIMIT="--limit 16"; SMOKE=1; fi
R18=$OUT/rtdetr_r18_full; R50=$OUT/rtdetr_r50_full
for d in "$R18" "$R50"; do
  if [ -e "$d" ]; then echo "$d exists: move it aside first"; exit 1; fi
done
mkdir -p "$OUT"; exec > >(tee -a "$OUT/overnight_queue.log") 2>&1
TD="python3 -m sld2quote.train_detector"
STOP="--patience 3 --min-delta 0.002 --eval-every 1 --nms-iou 0.5"
step() { echo "$(date '+%F %T') start: $*"; "$@"; local s=$?; echo "$(date '+%F %T') exit $s: $*"; }

step $TD train --data data/coco --out "$R18" --model PekingU/rtdetr_r18vd --epochs $R18_CAP --batch 16 --lr 1e-4 --backbone-lr 1e-5 \
  --weight-decay 1e-4 --warmup-steps 300 --grad-clip 0.1 $STOP --device mps --seed 0 --time-budget-min 330 $LIMIT
step $TD train --data data/coco --out "$R50" --model PekingU/rtdetr_r50vd --epochs $R50_CAP --batch 8 --lr 1e-4 --backbone-lr 1e-5 \
  --weight-decay 1e-4 --warmup-steps 300 --grad-clip 0.1 $STOP --device mps --seed 0 --time-budget-min 480 $LIMIT
for m in "$R18" "$R50"; do
  for s in val test_clean test_scan test_unseen; do
    step $TD eval --weights "$m" --data data/coco --split $s --nms-iou 0.5 --device mps $LIMIT
  done
done
step $TD summary --runs "$R18,$R50" --out "$OUT/overnight_summary.md"
if [ -n "$SMOKE" ]; then  # the more-passes path: a new run from trained weights must keep their head
  step $TD train --data data/coco --model "$R18" --out "$OUT/rtdetr_r18_more" --epochs 1 --batch 16 --lr 5e-05 --backbone-lr 5e-06 \
    --warmup-steps 100 $STOP --device mps --seed 0 $LIMIT
  if grep -q "newly initialized" "$OUT/rtdetr_r18_more/train.log"; then echo "more passes: FAILED, weights were newly initialised"
  else echo "more passes: the trained head was kept (no newly initialised weights)"; fi
fi
echo "$(date '+%F %T') queue done: $OUT/overnight_summary.md"
