#!/bin/bash
# Overnight queue for the sld2quote symbol detector, on synthetic tiles only (data/coco, cut from generated sheets): r18vd, then
# r50vd, each fine-tuned from the Hugging Face hub weights into its own weights directory (train.log, history.json, checkpoint-last,
# metrics.json and card.md there), then eval of every split for both, then weights/overnight_summary.md.
#
# Epochs were sized on 17 Oct 2026 on this Mac (M2 Max, MPS, torch 2.5.1) from a 50-step rate measured in the training loop
# (steps 11 to 60, after warm-up), so that each whole cosine schedule, val after every epoch included, ends inside its budget with
# a 10% margin; --time-budget-min stops a run cleanly at an epoch boundary if it would still overrun:
#   r18vd, batch 16: 6.41 images/s, 573 steps per epoch, about 23.8 min of training and 0.8 min of val per epoch, so 90% of
#     270 min holds 9 epochs (about 222 min).
#   r50vd, batch 8, since batch 16 took 22.3 GiB on the GPU and thrashed at 0.1 images/s while batch 8 takes 12.7 GiB: 2.99
#     images/s, 1147 steps per epoch, about 51.1 min of training and 1.5 min of val per epoch, so 90% of 390 min holds 6 epochs
#     (about 316 min).
#
#   scripts/train_overnight.sh           the queue, outputs under weights/
#   scripts/train_overnight.sh --smoke   the same steps end to end in a few minutes (16 tiles per split, 1 epoch per model),
#                                        outputs under weights/overnight_smoke
# Neither starts over an existing run directory, so the logs of two runs never mix. A failed step is logged and the queue goes on.
set -u
[ -z "${CAFFEINATED:-}" ] && CAFFEINATED=1 exec caffeinate -is "$0" "$@"  # the Mac stays awake until the queue ends
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1  # aten::grid_sampler_2d_backward has no MPS kernel in torch 2.5.1 and runs on the CPU
OUT=weights; R18_EPOCHS=9; R50_EPOCHS=6; LIMIT=""
if [ "${1:-}" = "--smoke" ]; then OUT=weights/overnight_smoke; R18_EPOCHS=1; R50_EPOCHS=1; LIMIT="--limit 16"; fi
R18=$OUT/rtdetr_r18_full; R50=$OUT/rtdetr_r50_full
for d in "$R18" "$R50"; do
  if [ -e "$d" ]; then echo "$d exists: move it aside first"; exit 1; fi
done
mkdir -p "$OUT"; exec > >(tee -a "$OUT/overnight_queue.log") 2>&1
TD="python3 -m sld2quote.train_detector"
step() { echo "$(date '+%F %T') start: $*"; "$@"; local s=$?; echo "$(date '+%F %T') exit $s: $*"; }

step $TD train --data data/coco --out "$R18" --model PekingU/rtdetr_r18vd --epochs $R18_EPOCHS --batch 16 --lr 1e-4 --backbone-lr 1e-5 \
  --weight-decay 1e-4 --warmup-steps 300 --grad-clip 0.1 --nms-iou 0.5 --device mps --seed 0 --time-budget-min 270 $LIMIT
step $TD train --data data/coco --out "$R50" --model PekingU/rtdetr_r50vd --epochs $R50_EPOCHS --batch 8 --lr 1e-4 --backbone-lr 1e-5 \
  --weight-decay 1e-4 --warmup-steps 300 --grad-clip 0.1 --nms-iou 0.5 --device mps --seed 0 --time-budget-min 390 $LIMIT
for m in "$R18" "$R50"; do
  for s in val test_clean test_scan test_unseen; do
    step $TD eval --weights "$m" --data data/coco --split $s --nms-iou 0.5 --device mps $LIMIT
  done
done
step $TD summary --runs "$R18,$R50" --out "$OUT/overnight_summary.md"
echo "$(date '+%F %T') queue done: $OUT/overnight_summary.md"
