# Overnight detector runs

Synthetic data only: every tile was cut from a sheet drawn by `sld2quote/generate.py`. Scores are per 640 px tile, so a symbol in two tiles counts twice. "With NMS" is the operating post-processing (class-agnostic duplicate suppression); "raw" is without it; each uses its own threshold, chosen at training time on val (and val_unseen, pooled, for runs that had it), which also picks the best pass; test_unseen is never used for choosing, only for the final numbers and the size rule.

## rtdetr_r18vd (`weights/rtdetr_r18_full`)

- 6 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 3: early stop after epoch 6 of 10: 3 val passes in a row without an improvement of 0.002 in val mAP@0.5 with NMS; the best pass was epoch 3 at 0.9286
- Scored with the best pass's weights, epoch 3 (val mAP@0.5 with NMS 0.9286); 120 min of training and 2.1 min of final evaluation, 7.9784 training images/s
- Thresholds chosen on val at training time: with NMS 0.10, raw 0.15; NMS is class-agnostic at IoU 0.5
- eval of val, test_clean, test_scan after training reproduces the training-time metrics exactly; evaluated after training on a newer export, at the thresholds chosen then, and shown so: val_unseen (396 tiles, export 3f19db3a), test_unseen (1140 tiles, export 3f19db3a)

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 936 | with NMS | 0.9286 | 0.9436 | 0.6842 (relay) | 0.8903 | 0.7653 (earth) |
| val | 936 | raw | 0.9023 | 0.9128 | 0.7188 (relay) | 0.8136 | 0.6 (spd) |
| test_clean | 348 | with NMS | 0.9392 | 0.9557 | 0.7551 (board) | 0.9059 | 0.6731 (earth) |
| test_clean | 348 | raw | 0.9142 | 0.8964 | 0.7333 (relay) | 0.8538 | 0.5577 (earth) |
| test_scan | 348 | with NMS | 0.8402 | 0.9705 | 0.8125 (earth) | 0.7955 | 0.375 (capacitor) |
| test_scan | 348 | raw | 0.816 | 0.9107 | 0.7333 (earth) | 0.6971 | 0.0417 (capacitor) |
| test_unseen | 1140 | with NMS | 0.9219 | 0.9316 | 0.7344 (board) | 0.8801 | 0.6667 (capacitor) |
| test_unseen | 1140 | raw | 0.8881 | 0.8905 | 0.5968 (relay) | 0.809 | 0.5556 (capacitor) |
| val_unseen | 396 | with NMS | 0.9067 | 0.8485 | 0.0 (board) | 0.869 | 0.625 (earth) |
| val_unseen | 396 | raw | 0.8669 | 0.8059 | 0.0 (board) | 0.7809 | 0.5714 (earth) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9897 | 0.844 | 0.9825 | 0.8222 | fail |
| fuse_switch | 0.9963 | 0.9574 | 0.9811 | 0.9594 | pass |
| isolator | 0.9554 | 0.9554 | 0.979 | 0.8537 | fail |
| contactor | 0.97 | 0.9741 | 0.9107 | 0.9518 | fail |
| transformer | 0.9733 | 1.0 | 0.9949 | 1.0 | pass |
| generator | 1.0 | 0.9661 | 0.9881 | 0.9881 | pass |
| motor | 0.9704 | 0.8603 | 0.9931 | 0.8614 | fail |
| board | 0.8516 | 0.8684 | 0.7344 | 0.9038 | fail |
| ct | 0.9858 | 0.8797 | 0.9659 | 0.9 | fail |
| meter | 0.9245 | 0.9608 | 0.9248 | 0.9766 | fail |
| relay | 0.6842 | 0.8667 | 0.7358 | 0.9286 | fail |
| spd | 1.0 | 0.7818 | 0.9801 | 0.8043 | fail |
| capacitor | 0.9718 | 0.7841 | 0.9351 | 0.6667 | fail |
| earth | 0.9375 | 0.7653 | 0.9364 | 0.7055 | fail |

Size rule: 3 of 14 classes pass; fails on breaker, isolator, contactor, motor, board, ct, meter, relay, spd, capacitor, earth.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|
| 1 | 6.5395 | 0.711 | 0.6767 | 7.6389 | 20.9 |
| 2 | 3.0598 | 0.8623 | 0.8297 | 7.9983 | 19.9 |
| 3 (best) | 2.6056 | 0.9286 | 0.9023 | 8.0298 | 19.8 |
| 4 | 2.5886 | 0.905 | 0.8757 | 8.0252 | 19.9 |
| 5 | 2.2667 | 0.9118 | 0.8543 | 8.1326 | 19.6 |
| 6 | 2.1317 | 0.9126 | 0.8735 | 8.0457 | 19.8 |

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 86% of the symbols, and 3% of what it reports is a false alarm (166 on 1140 tiles, 70 of them at a tile edge). It misses too many capacitors (finds 67%), earth symbols (finds 71%), surge arresters (finds 80%), breakers (finds 82%). It reports some symbols that are not there, mostly downstream boards (27% of those reported are wrong), protection relays (26% of those reported are wrong), contactors (9% of those reported are wrong). Scan-like copies are harder: it finds 83% of their symbols against 89% on clean copies. It finds 84% of the breakers drawn as a box with a diagonal and 81% of the IEC breakers. It finds 43% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 314 false alarms on the same tiles instead of 166.

## rtdetr_r18vd (`weights/rtdetr_r18_rare`)

- 10 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 0, rare-class sampling at T 0.15 by class: ended by the cap
- Scored with the best pass's weights, epoch 4 (val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS 0.9201); 294 min of training and 3.1 min of final evaluation, 6.1978 training images/s
- Thresholds chosen on val and val_unseen pooled at training time: with NMS 0.10, raw 0.15; NMS is class-agnostic at IoU 0.5
- eval of val, val_unseen, test_clean, test_scan, test_unseen after training reproduces the training-time metrics exactly

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 936 | with NMS | 0.9281 | 0.9624 | 0.872 (board) | 0.8929 | 0.6091 (spd) |
| val | 936 | raw | 0.8878 | 0.8483 | 0.5833 (generator) | 0.8591 | 0.5091 (spd) |
| val_unseen | 396 | with NMS | 0.9403 | 0.9376 | 0.8 (spd) | 0.8958 | 0.625 (spd) |
| val_unseen | 396 | raw | 0.9023 | 0.7975 | 0.0 (board) | 0.8502 | 0.5 (spd) |
| test_clean | 348 | with NMS | 0.9023 | 0.972 | 0.8182 (board) | 0.8659 | 0.5385 (relay) |
| test_clean | 348 | raw | 0.8615 | 0.8308 | 0.5472 (generator) | 0.8272 | 0.5357 (spd) |
| test_scan | 348 | with NMS | 0.9051 | 0.9748 | 0.8421 (capacitor) | 0.8554 | 0.5 (spd) |
| test_scan | 348 | raw | 0.8772 | 0.8981 | 0.6587 (fuse_switch) | 0.8036 | 0.3214 (spd) |
| test_unseen | 1140 | with NMS | 0.9194 | 0.968 | 0.8361 (board) | 0.8732 | 0.663 (spd) |
| test_unseen | 1140 | raw | 0.8718 | 0.8217 | 0.5795 (board) | 0.8242 | 0.5435 (spd) |
| test_unseen, `pass-10`, threshold 0.08 | 1140 | with NMS | 0.9091 | 0.956 | 0.8217 (earth) | 0.808 | 0.622 (isolator) |
| test_unseen, `pass-10`, threshold 0.08 | 1140 | raw | 0.8634 | 0.8548 | 0.6296 (capacitor) | 0.808 | 0.622 (isolator) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9973 | 0.8809 | 0.9921 | 0.8185 | fail |
| fuse_switch | 0.9755 | 0.9876 | 0.9801 | 0.9729 | pass |
| isolator | 1.0 | 0.9375 | 0.9858 | 0.8476 | fail |
| contactor | 0.876 | 0.944 | 0.975 | 0.9398 | fail |
| transformer | 0.9929 | 0.9589 | 0.9948 | 0.9745 | pass |
| generator | 1.0 | 0.9661 | 1.0 | 0.9881 | pass |
| motor | 0.9648 | 0.8384 | 0.9964 | 0.8373 | fail |
| board | 0.872 | 0.9408 | 0.8361 | 0.9808 | fail |
| ct | 0.9932 | 0.9241 | 0.9905 | 0.95 | fail |
| meter | 1.0 | 0.9608 | 0.9858 | 0.972 | pass |
| relay | 0.9231 | 0.8 | 0.9697 | 0.7619 | fail |
| spd | 1.0 | 0.6091 | 1.0 | 0.663 | fail |
| capacitor | 0.9529 | 0.9205 | 0.9457 | 0.8056 | fail |
| earth | 0.9261 | 0.8316 | 0.9004 | 0.7123 | fail |

Size rule: 4 of 14 classes pass; fails on breaker, isolator, contactor, motor, board, ct, relay, spd, capacitor, earth.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val_unseen mAP@0.5 with NMS | pooled mean F1 | mean mAP@0.5 | val_unseen worst min(P, R) | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.9317 | 0.8078 | 0.768 | 0.7946 | 0.7879 | 0.2286 | 0.7788 | 5.3444 | 33.6 |
| 2 | 2.7728 | 0.8868 | 0.8785 | 0.856 | 0.8826 | 0.4643 | 0.8447 | 6.0457 | 30.0 |
| 3 | 2.4795 | 0.8241 | 0.8255 | 0.7946 | 0.8248 | 0.2857 | 0.7782 | 5.7438 | 31.1 |
| 4 (best) | 2.3617 | 0.9281 | 0.9403 | 0.9201 | 0.9342 | 0.625 | 0.8878 | 5.7228 | 31.5 |
| 5 | 2.1978 | 0.9055 | 0.9242 | 0.8696 | 0.9148 | 0.32 | 0.8505 | 5.8365 | 30.7 |
| 6 | 2.0652 | 0.8978 | 0.8639 | 0.8449 | 0.8809 | 0.4 | 0.8497 | 5.7826 | 31.1 |
| 7 | 1.8983 | 0.9051 | 0.9069 | 0.8616 | 0.906 | 0.4348 | 0.8517 | 5.6478 | 31.6 |
| 8 | 1.8248 | 0.9207 | 0.8925 | 0.8803 | 0.9066 | 0.5517 | 0.8844 | 7.261 | 25.1 |
| 9 | 1.7724 | 0.9162 | 0.914 | 0.8743 | 0.9151 | 0.6071 | 0.8804 | 7.2692 | 24.9 |
| 10 | 1.7564 | 0.919 | 0.9115 | 0.8874 | 0.9153 | 0.6562 | 0.8821 | 7.3242 | 24.7 |

Candidates for the drawing-level check: the pass chosen by val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS, pass 4 (`weights/rtdetr_r18_rare`, the final weights), and the last pass, pass 10 (`weights/rtdetr_r18_rare/pass-10`), fully annealed; every pass's weights are kept.

Passes 1, 2, 3, 4, 5, 6, 7 ran below 85% of the run's best rate (7.3242 images/s), slowed by other work on the machine: those minutes are contended.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 86% of the symbols, and 2% of what it reports is a false alarm (88 on 1140 tiles, 46 of them at a tile edge). It misses too many surge arresters (finds 66%), earth symbols (finds 71%), protection relays (finds 76%), capacitors (finds 81%). It reports some symbols that are not there, mostly downstream boards (16% of those reported are wrong), earth symbols (10% of those reported are wrong), capacitors (5% of those reported are wrong). Scan-like copies are harder: it finds 85% of their symbols against 87% on clean copies. It finds 51% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 842 false alarms on the same tiles instead of 88.

## rtdetr_r18vd (`weights/rtdetr_r18_plain`)

- 10 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 0: ended by the cap
- Scored with the best pass's weights, epoch 10 (val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS 0.8882); 226 min of training and 3.0 min of final evaluation, 7.1887 training images/s
- Thresholds chosen on val and val_unseen pooled at training time: with NMS 0.06, raw 0.08; NMS is class-agnostic at IoU 0.5
- eval of val, val_unseen, test_clean, test_scan, test_unseen after training reproduces the training-time metrics exactly

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 936 | with NMS | 0.9129 | 0.9322 | 0.5564 (capacitor) | 0.8621 | 0.6333 (relay) |
| val | 936 | raw | 0.8731 | 0.9045 | 0.6306 (capacitor) | 0.8179 | 0.6 (relay) |
| val_unseen | 396 | with NMS | 0.9288 | 0.854 | 0.0 (board) | 0.8836 | 0.5893 (earth) |
| val_unseen | 396 | raw | 0.9029 | 0.8406 | 0.0 (board) | 0.8209 | 0.4286 (earth) |
| test_clean | 348 | with NMS | 0.9067 | 0.8956 | 0.3014 (capacitor) | 0.8814 | 0.5577 (earth) |
| test_clean | 348 | raw | 0.879 | 0.8814 | 0.5714 (capacitor) | 0.8483 | 0.5577 (earth) |
| test_scan | 348 | with NMS | 0.9235 | 0.9747 | 0.9091 (earth) | 0.8512 | 0.5769 (earth) |
| test_scan | 348 | raw | 0.8903 | 0.9295 | 0.8667 (transformer) | 0.8 | 0.5 (earth) |
| test_unseen | 1140 | with NMS | 0.913 | 0.9313 | 0.7015 (board) | 0.8439 | 0.5719 (earth) |
| test_unseen | 1140 | raw | 0.8665 | 0.8999 | 0.7 (board) | 0.7888 | 0.4939 (isolator) |
| test_unseen, `pass-10`, threshold 0.06 | 1140 | with NMS | 0.913 | 0.9313 | 0.7015 (board) | 0.8439 | 0.5719 (earth) |
| test_unseen, `pass-10`, threshold 0.08 | 1140 | raw | 0.8665 | 0.8999 | 0.7 (board) | 0.7888 | 0.4939 (isolator) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9919 | 0.9372 | 0.9942 | 0.9017 | fail |
| fuse_switch | 0.9799 | 0.9504 | 0.976 | 0.9507 | pass |
| isolator | 1.0 | 0.8661 | 0.991 | 0.6707 | fail |
| contactor | 0.9545 | 0.9052 | 0.9283 | 0.8193 | fail |
| transformer | 0.9795 | 0.9795 | 0.9898 | 0.9898 | pass |
| generator | 0.9828 | 0.9661 | 0.9882 | 1.0 | pass |
| motor | 0.9301 | 0.7555 | 0.9522 | 0.7801 | fail |
| board | 0.8889 | 0.8947 | 0.7015 | 0.9038 | fail |
| ct | 0.9605 | 0.9241 | 0.986 | 0.9636 | fail |
| meter | 0.9931 | 0.9346 | 0.9808 | 0.9533 | fail |
| relay | 0.95 | 0.6333 | 0.9375 | 0.7143 | fail |
| spd | 0.9878 | 0.7364 | 1.0 | 0.7989 | fail |
| capacitor | 0.5564 | 0.8409 | 0.7049 | 0.7963 | fail |
| earth | 0.8957 | 0.7449 | 0.9076 | 0.5719 | fail |

Size rule: 3 of 14 classes pass; fails on breaker, isolator, contactor, motor, board, ct, meter, relay, spd, capacitor, earth.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val_unseen mAP@0.5 with NMS | pooled mean F1 | mean mAP@0.5 | val_unseen worst min(P, R) | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.376 | 0.7471 | 0.7242 | 0.7482 | 0.7356 | 0.3333 | 0.7311 | 7.0349 | 23.3 |
| 2 | 2.9394 | 0.8641 | 0.8536 | 0.8414 | 0.8589 | 0.3438 | 0.8334 | 6.8571 | 23.6 |
| 3 | 2.6165 | 0.9064 | 0.8699 | 0.8619 | 0.8881 | 0.1 | 0.8593 | 7.3724 | 22.0 |
| 4 | 2.452 | 0.8657 | 0.8766 | 0.8584 | 0.8712 | 0.57 | 0.837 | 7.3059 | 22.2 |
| 5 | 2.3061 | 0.8794 | 0.8858 | 0.8122 | 0.8826 | 0.233 | 0.8455 | 7.2739 | 22.4 |
| 6 | 2.0968 | 0.9188 | 0.9065 | 0.8832 | 0.9126 | 0.4821 | 0.8763 | 7.302 | 22.3 |
| 7 | 2.0182 | 0.9116 | 0.9149 | 0.8862 | 0.9133 | 0.5536 | 0.8689 | 7.2272 | 22.5 |
| 8 | 1.9366 | 0.9161 | 0.9197 | 0.8767 | 0.9179 | 0.386 | 0.8764 | 7.2284 | 22.5 |
| 9 | 1.8801 | 0.9116 | 0.9263 | 0.8878 | 0.9189 | 0.5536 | 0.8714 | 6.989 | 23.1 |
| 10 (best) | 1.8604 | 0.9129 | 0.9288 | 0.8882 | 0.9208 | 0.434 | 0.8731 | 7.2962 | 22.3 |

Candidates for the drawing-level check: the pass chosen by val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS, pass 10 (`weights/rtdetr_r18_plain`, the final weights), and the last pass, pass 10 (`weights/rtdetr_r18_plain/pass-10`), fully annealed; every pass's weights are kept.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 87% of the symbols, and 3% of what it reports is a false alarm (152 on 1140 tiles, 78 of them at a tile edge). It misses too many earth symbols (finds 57%), isolators (finds 67%), protection relays (finds 71%), motors (finds 78%). It reports some symbols that are not there, mostly downstream boards (30% of those reported are wrong), capacitors (30% of those reported are wrong), earth symbols (9% of those reported are wrong). Scan-like copies are harder: it finds 86% of their symbols against 89% on clean copies. It finds 88% of the breakers drawn as a box with a diagonal and 92% of the IEC breakers. It finds 55% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 307 false alarms on the same tiles instead of 152.

## rtdetr_r18vd (`weights/rtdetr_r18_fonts`)

- 10 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 0: ended by the cap
- Scored with the best pass's weights, epoch 4 (val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS 0.8516); 200 min of training and 3.2 min of final evaluation, 7.5881 training images/s
- Thresholds chosen on val and val_unseen pooled at training time: with NMS 0.12, raw 0.20; NMS is class-agnostic at IoU 0.5
- eval of val_unseen, test_clean, test_scan, test_unseen after training reproduces the training-time metrics exactly; eval of val differs from the training-time metrics on the same export

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 824 | with NMS | 0.8514 | 0.9413 | 0.7273 (relay) | 0.8005 | 0.2353 (relay) |
| val | 824 | raw | 0.7189 | 0.7262 | 0.4672 (contactor) | 0.6892 | 0.1471 (relay) |
| val_unseen | 400 | with NMS | 0.8713 | 0.9406 | 0.4643 (board) | 0.8267 | 0.1667 (relay) |
| val_unseen | 400 | raw | 0.7441 | 0.646 | 0.0 (relay) | 0.7355 | 0.0 (relay) |
| test_clean | 312 | with NMS | 0.9018 | 0.9342 | 0.56 (board) | 0.8706 | 0.1875 (relay) |
| test_clean | 312 | raw | 0.7819 | 0.6783 | 0.3819 (contactor) | 0.7702 | 0.1875 (relay) |
| test_scan | 312 | with NMS | 0.8644 | 0.9433 | 0.7 (capacitor) | 0.7946 | 0.3125 (relay) |
| test_scan | 312 | raw | 0.7473 | 0.7264 | 0.5208 (board) | 0.6901 | 0.125 (relay) |
| test_unseen | 1236 | with NMS | 0.8625 | 0.9245 | 0.5556 (relay) | 0.8169 | 0.4167 (relay) |
| test_unseen | 1236 | raw | 0.7266 | 0.7044 | 0.4593 (contactor) | 0.7083 | 0.2917 (relay) |
| val on `data/coco`, threshold 0.12 | 936 | with NMS | 0.869 | 0.9522 | 0.7849 (board) | 0.8201 | 0.1333 (relay) |
| val on `data/coco`, threshold 0.20 | 936 | raw | 0.7301 | 0.7 | 0.5092 (board) | 0.7027 | 0.0667 (relay) |
| test_unseen on `data/coco`, threshold 0.12 | 1140 | with NMS | 0.8652 | 0.9253 | 0.625 (relay) | 0.8116 | 0.2381 (relay) |
| test_unseen on `data/coco`, threshold 0.20 | 1140 | raw | 0.7331 | 0.697 | 0.4375 (board) | 0.6999 | 0.1429 (relay) |
| test_unseen, `pass-10`, threshold 0.09 | 1236 | with NMS | 0.7976 | 0.87 | 0.1818 (relay) | 0.7368 | 0.0833 (relay) |
| test_unseen, `pass-10`, threshold 0.15 | 1236 | raw | 0.6269 | 0.5875 | 0.1 (relay) | 0.6249 | 0.0417 (relay) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.987 | 0.8829 | 0.9877 | 0.8696 | fail |
| fuse_switch | 0.991 | 0.8919 | 0.9976 | 0.9126 | fail |
| isolator | 0.9904 | 0.9537 | 1.0 | 0.9189 | fail |
| contactor | 0.934 | 0.92 | 0.9209 | 0.8442 | fail |
| transformer | 0.9871 | 1.0 | 0.9787 | 0.9892 | pass |
| generator | 1.0 | 0.9839 | 1.0 | 0.9681 | pass |
| motor | 0.9545 | 0.8317 | 0.9536 | 0.8309 | fail |
| board | 0.7544 | 0.9556 | 0.6947 | 0.9785 | fail |
| ct | 0.9429 | 0.9041 | 0.9389 | 0.9037 | fail |
| meter | 0.985 | 0.9357 | 0.9778 | 0.9514 | fail |
| relay | 0.7273 | 0.2353 | 0.5556 | 0.4167 | fail |
| spd | 1.0 | 0.7955 | 0.9837 | 0.8403 | fail |
| capacitor | 0.963 | 0.4062 | 0.9667 | 0.4677 | fail |
| earth | 0.9615 | 0.5102 | 0.9865 | 0.5448 | fail |

Size rule: 2 of 14 classes pass; fails on breaker, fuse_switch, isolator, contactor, motor, board, ct, meter, relay, spd, capacitor, earth.

Per class on `data/coco` with NMS, comparable with the runs trained there, and the size rule on its val and test_unseen:

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9877 | 0.8457 | 0.9876 | 0.8437 | fail |
| fuse_switch | 0.9902 | 0.8936 | 0.9801 | 0.9113 | fail |
| isolator | 0.9709 | 0.8929 | 0.9867 | 0.9024 | fail |
| contactor | 0.9513 | 0.9267 | 0.9283 | 0.8976 | fail |
| transformer | 0.9603 | 0.9932 | 0.98 | 1.0 | pass |
| generator | 0.9825 | 0.9492 | 1.0 | 1.0 | fail |
| motor | 0.9808 | 0.8908 | 0.9627 | 0.7771 | fail |
| board | 0.7849 | 0.9605 | 0.6456 | 0.9808 | fail |
| ct | 0.9867 | 0.9367 | 0.9704 | 0.8955 | fail |
| meter | 0.9737 | 0.9673 | 0.9856 | 0.9579 | pass |
| relay | 0.8 | 0.1333 | 0.625 | 0.2381 | fail |
| spd | 1.0 | 0.7364 | 0.9933 | 0.8043 | fail |
| capacitor | 0.9697 | 0.7273 | 0.9531 | 0.5648 | fail |
| earth | 0.9919 | 0.6276 | 0.9556 | 0.589 | fail |

Size rule on `data/coco`: 2 of 14 classes pass.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val_unseen mAP@0.5 with NMS | pooled mean F1 | mean mAP@0.5 | val_unseen worst min(P, R) | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.3948 | 0.5744 | 0.5608 | 0.5812 | 0.5676 | 0.1111 | 0.5128 | 7.6196 | 19.8 |
| 2 | 3.0845 | 0.7781 | 0.7954 | 0.7722 | 0.7868 | 0.3333 | 0.6373 | 7.8444 | 19.3 |
| 3 | 2.8316 | 0.7914 | 0.8384 | 0.7819 | 0.8149 | 0.3333 | 0.6564 | 7.9542 | 19.0 |
| 4 (best) | 2.538 | 0.8514 | 0.8713 | 0.8516 | 0.8614 | 0.1667 | 0.7188 | 6.6744 | 22.7 |
| 5 | 2.3003 | 0.8158 | 0.8485 | 0.8188 | 0.8321 | 0.3333 | 0.6558 | 7.5463 | 20.1 |
| 6 | 2.1953 | 0.7914 | 0.8377 | 0.7917 | 0.8145 | 0.1667 | 0.5988 | 7.7148 | 19.6 |
| 7 | 2.0432 | 0.8016 | 0.8477 | 0.8065 | 0.8246 | 0.1667 | 0.6157 | 8.029 | 18.8 |
| 8 | 1.9442 | 0.7935 | 0.848 | 0.8017 | 0.8207 | 0.1667 | 0.6144 | 7.8758 | 19.2 |
| 9 | 1.8969 | 0.7769 | 0.8239 | 0.7856 | 0.8004 | 0.1667 | 0.5889 | 7.7787 | 19.6 |
| 10 | 1.8742 | 0.7798 | 0.83 | 0.7865 | 0.8049 | 0.1667 | 0.5982 | 6.8435 | 22.1 |

Candidates for the drawing-level check: the pass chosen by val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS, pass 4 (`weights/rtdetr_r18_fonts`, the final weights), and the last pass, pass 10 (`weights/rtdetr_r18_fonts/pass-10`), fully annealed; every pass's weights are kept.

Pass 4 ran below 85% of the run's best rate (8.029 images/s), slowed by other work on the machine: those minutes are contended.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 86% of the symbols, and 3% of what it reports is a false alarm (130 on 1236 tiles, 58 of them at a tile edge). It misses too many protection relays (finds 42%), capacitors (finds 47%), earth symbols (finds 54%), motors (finds 83%). It reports some symbols that are not there, mostly protection relays (44% of those reported are wrong), downstream boards (31% of those reported are wrong), contactors (8% of those reported are wrong). Scan-like copies are harder: it finds 84% of their symbols against 87% on clean copies. It finds 91% of the breakers drawn as a box with a diagonal and 83% of the IEC breakers. It finds 36% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 1816 false alarms on the same tiles instead of 130.

## rtdetr_r18vd (`weights/rtdetr_r18_plain_ema`)

- 10 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 0, EMA decay 0.999 with 200 warm-up updates (d 0.9990 after 5730 updates, an average over about 1000 steps, 1.7 passes of 573 steps; the authors' 0.9999 with 2000 warm-up updates, tuned to 72 epochs of hundreds of thousands of steps, would average about 18 steps on this run, so the decay is set for its length; the evaluated weights are the average): ended by the cap
- Scored with the best pass's weights, epoch 8 (val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS 0.9186); 255 min of training and 3.2 min of final evaluation, 6.463 training images/s
- Thresholds chosen on val and val_unseen pooled at training time: with NMS 0.07, raw 0.10; NMS is class-agnostic at IoU 0.5
- eval of val, val_unseen, test_clean, test_scan, test_unseen after training reproduces the training-time metrics exactly

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 936 | with NMS | 0.9395 | 0.941 | 0.6538 (relay) | 0.9073 | 0.5667 (relay) |
| val | 936 | raw | 0.8991 | 0.8626 | 0.7692 (ct) | 0.852 | 0.4 (relay) |
| val_unseen | 396 | with NMS | 0.9404 | 0.867 | 0.0 (board) | 0.8918 | 0.5 (relay) |
| val_unseen | 396 | raw | 0.8968 | 0.7746 | 0.0 (board) | 0.8222 | 0.3 (relay) |
| test_clean | 348 | with NMS | 0.9393 | 0.97 | 0.9091 (board) | 0.924 | 0.7692 (relay) |
| test_clean | 348 | raw | 0.9108 | 0.8656 | 0.7724 (motor) | 0.8874 | 0.6154 (relay) |
| test_scan | 348 | with NMS | 0.9006 | 0.9418 | 0.5 (relay) | 0.8641 | 0.3846 (relay) |
| test_scan | 348 | raw | 0.8547 | 0.833 | 0.4 (relay) | 0.8053 | 0.1538 (relay) |
| test_unseen | 1140 | with NMS | 0.9495 | 0.9442 | 0.7838 (relay) | 0.9043 | 0.6905 (relay) |
| test_unseen | 1140 | raw | 0.9099 | 0.8747 | 0.7626 (ct) | 0.8501 | 0.619 (relay) |
| test_unseen, `pass-10`, threshold 0.06 | 1140 | with NMS | 0.9559 | 0.9437 | 0.8108 (relay) | 0.9208 | 0.7143 (relay) |
| test_unseen, `pass-10`, threshold 0.09 | 1140 | raw | 0.9103 | 0.8651 | 0.7405 (ct) | 0.8525 | 0.6429 (relay) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.96 | 0.9443 | 0.9598 | 0.9119 | fail |
| fuse_switch | 0.9928 | 0.984 | 0.985 | 0.9704 | pass |
| isolator | 1.0 | 0.9018 | 0.9927 | 0.8293 | fail |
| contactor | 0.966 | 0.9784 | 0.9763 | 0.991 | pass |
| transformer | 1.0 | 1.0 | 0.9898 | 0.9949 | pass |
| generator | 0.9592 | 0.7966 | 1.0 | 0.7381 | fail |
| motor | 0.9298 | 0.9825 | 0.9384 | 0.9639 | fail |
| board | 0.9236 | 0.9539 | 0.7969 | 0.9808 | fail |
| ct | 0.9 | 0.9684 | 0.9556 | 0.9773 | fail |
| meter | 0.9744 | 0.9935 | 0.9596 | 1.0 | pass |
| relay | 0.6538 | 0.5667 | 0.7838 | 0.6905 | fail |
| spd | 1.0 | 0.8273 | 1.0 | 0.8315 | fail |
| capacitor | 0.9432 | 0.9432 | 0.9533 | 0.9444 | fail |
| earth | 0.9713 | 0.8622 | 0.9278 | 0.8356 | fail |

Size rule: 4 of 14 classes pass; fails on breaker, isolator, generator, motor, board, ct, relay, spd, capacitor, earth.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val_unseen mAP@0.5 with NMS | pooled mean F1 | mean mAP@0.5 | val_unseen worst min(P, R) | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.5953 | 0.7926 | 0.7515 | 0.7743 | 0.772 | 0.069 | 0.7793 | 5.1132 | 31.7 |
| 2 | 2.9541 | 0.8808 | 0.8839 | 0.8674 | 0.8823 | 0.5312 | 0.8538 | 5.2429 | 30.7 |
| 3 | 2.687 | 0.8796 | 0.8822 | 0.8715 | 0.8809 | 0.5417 | 0.8483 | 5.9903 | 27.1 |
| 4 | 2.4508 | 0.8815 | 0.8747 | 0.8711 | 0.8781 | 0.5 | 0.8518 | 5.7904 | 27.9 |
| 5 | 2.2654 | 0.9022 | 0.9032 | 0.891 | 0.9027 | 0.5882 | 0.8675 | 7.1978 | 22.7 |
| 6 | 2.1374 | 0.9245 | 0.9314 | 0.9005 | 0.928 | 0.5294 | 0.8878 | 7.248 | 22.4 |
| 7 | 1.9606 | 0.9279 | 0.9243 | 0.9049 | 0.9261 | 0.5882 | 0.8921 | 7.0642 | 23.0 |
| 8 (best) | 1.8796 | 0.9395 | 0.9404 | 0.9186 | 0.94 | 0.5 | 0.8991 | 7.2089 | 22.6 |
| 9 | 1.8023 | 0.9448 | 0.9422 | 0.9179 | 0.9435 | 0.5938 | 0.8965 | 7.2644 | 22.5 |
| 10 | 1.787 | 0.9492 | 0.9406 | 0.9175 | 0.9449 | 0.6 | 0.8989 | 6.5095 | 24.9 |

Candidates for the drawing-level check: the pass chosen by val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS, pass 8 (`weights/rtdetr_r18_plain_ema`, the final weights), and the last pass, pass 10 (`weights/rtdetr_r18_plain_ema/pass-10`), fully annealed; every pass's weights are kept.

Passes 1, 2, 3, 4 ran below 85% of the run's best rate (7.2644 images/s), slowed by other work on the machine: those minutes are contended.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 93% of the symbols, and 4% of what it reports is a false alarm (202 on 1140 tiles, 73 of them at a tile edge). It misses too many protection relays (finds 69%), generators (finds 74%), isolators (finds 83%), surge arresters (finds 83%). It reports some symbols that are not there, mostly protection relays (22% of those reported are wrong), downstream boards (20% of those reported are wrong), earth symbols (7% of those reported are wrong). It finds 49% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 663 false alarms on the same tiles instead of 202.

## rtdetr_r18vd (`weights/rtdetr_r18_fonts_ema`)

- 10 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 0, EMA decay 0.999 with 200 warm-up updates (d 0.9990 after 5330 updates, an average over about 1000 steps, 1.9 passes of 533 steps; the authors' 0.9999 with 2000 warm-up updates, tuned to 72 epochs of hundreds of thousands of steps, would average about 14 steps on this run, so the decay is set for its length; the evaluated weights are the average): ended by the cap
- Scored with the best pass's weights, epoch 10 (val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS 0.9174); 259 min of training and 2.9 min of final evaluation, 6.1418 training images/s
- Thresholds chosen on val and val_unseen pooled at training time: with NMS 0.07, raw 0.09; NMS is class-agnostic at IoU 0.5
- eval of val, test_clean, test_scan, test_unseen after training reproduces the training-time metrics exactly; eval of val_unseen differs from the training-time metrics on the same export

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 824 | with NMS | 0.9237 | 0.9465 | 0.7 (relay) | 0.8947 | 0.4118 (relay) |
| val | 824 | raw | 0.8841 | 0.8587 | 0.6667 (relay) | 0.8671 | 0.3529 (relay) |
| val_unseen | 400 | with NMS | 0.9209 | 0.9253 | 0.3333 (relay) | 0.8996 | 0.3333 (relay) |
| val_unseen | 400 | raw | 0.8887 | 0.8068 | 0.2857 (relay) | 0.8755 | 0.3333 (relay) |
| test_clean | 312 | with NMS | 0.9417 | 0.9642 | 0.8454 (contactor) | 0.8967 | 0.125 (relay) |
| test_clean | 312 | raw | 0.9035 | 0.8791 | 0.7465 (earth) | 0.8708 | 0.0625 (relay) |
| test_scan | 312 | with NMS | 0.9298 | 0.9265 | 0.5714 (relay) | 0.8853 | 0.25 (relay) |
| test_scan | 312 | raw | 0.8934 | 0.8246 | 0.2222 (relay) | 0.8553 | 0.125 (relay) |
| test_unseen | 1236 | with NMS | 0.9205 | 0.9081 | 0.3333 (relay) | 0.8915 | 0.2083 (relay) |
| test_unseen | 1236 | raw | 0.8811 | 0.8141 | 0.2 (relay) | 0.866 | 0.1667 (relay) |
| val on `data/coco`, threshold 0.07 | 936 | with NMS | 0.95 | 0.9506 | 0.7533 (contactor) | 0.9115 | 0.4 (relay) |
| val on `data/coco`, threshold 0.09 | 936 | raw | 0.9105 | 0.8522 | 0.5882 (relay) | 0.8836 | 0.3333 (relay) |
| test_unseen on `data/coco`, threshold 0.07 | 1140 | with NMS | 0.9397 | 0.9248 | 0.5172 (relay) | 0.9082 | 0.3571 (relay) |
| test_unseen on `data/coco`, threshold 0.09 | 1140 | raw | 0.9036 | 0.8382 | 0.4138 (relay) | 0.8789 | 0.2857 (relay) |
| test_unseen, `pass-10`, threshold 0.07 | 1236 | with NMS | 0.9205 | 0.9081 | 0.3333 (relay) | 0.8915 | 0.2083 (relay) |
| test_unseen, `pass-10`, threshold 0.09 | 1236 | raw | 0.8811 | 0.8141 | 0.2 (relay) | 0.866 | 0.1667 (relay) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9906 | 0.9808 | 0.9899 | 0.9744 | pass |
| fuse_switch | 0.992 | 0.9984 | 0.9989 | 0.9956 | pass |
| isolator | 0.9756 | 0.7407 | 0.9576 | 0.7635 | fail |
| contactor | 0.7586 | 0.99 | 0.7452 | 0.9855 | fail |
| transformer | 0.9935 | 1.0 | 0.9789 | 1.0 | pass |
| generator | 1.0 | 0.9516 | 0.9892 | 0.9787 | pass |
| motor | 0.959 | 0.9257 | 0.9813 | 0.9669 | fail |
| board | 0.9775 | 0.9667 | 0.8491 | 0.9677 | fail |
| ct | 0.9865 | 1.0 | 0.9686 | 0.9893 | pass |
| meter | 0.9683 | 0.8714 | 0.9816 | 0.8649 | fail |
| relay | 0.7 | 0.4118 | 0.3333 | 0.2083 | fail |
| spd | 0.9821 | 0.8333 | 0.9924 | 0.9028 | fail |
| capacitor | 0.9836 | 0.9375 | 0.9669 | 0.9435 | fail |
| earth | 0.9836 | 0.9184 | 0.9805 | 0.9403 | fail |

Size rule: 5 of 14 classes pass; fails on isolator, contactor, motor, board, meter, relay, spd, capacitor, earth.

Per class on `data/coco` with NMS, comparable with the runs trained there, and the size rule on its val and test_unseen:

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9858 | 0.9765 | 0.9871 | 0.9659 | pass |
| fuse_switch | 0.9964 | 0.9911 | 0.989 | 0.9938 | pass |
| isolator | 1.0 | 0.8036 | 1.0 | 0.811 | fail |
| contactor | 0.7533 | 0.9871 | 0.7653 | 0.9819 | fail |
| transformer | 0.9799 | 1.0 | 0.9899 | 1.0 | pass |
| generator | 0.9833 | 1.0 | 1.0 | 0.9524 | pass |
| motor | 0.9738 | 0.9738 | 0.9783 | 0.9518 | pass |
| board | 0.966 | 0.9342 | 0.8254 | 1.0 | fail |
| ct | 0.9812 | 0.9937 | 0.9735 | 1.0 | pass |
| meter | 0.9787 | 0.902 | 0.9617 | 0.9393 | fail |
| relay | 0.8 | 0.4 | 0.5172 | 0.3571 | fail |
| spd | 0.9894 | 0.8455 | 0.994 | 0.8967 | fail |
| capacitor | 0.9565 | 1.0 | 0.9804 | 0.9259 | fail |
| earth | 0.9639 | 0.9541 | 0.9856 | 0.9384 | fail |

Size rule on `data/coco`: 6 of 14 classes pass.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val_unseen mAP@0.5 with NMS | pooled mean F1 | mean mAP@0.5 | val_unseen worst min(P, R) | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.3457 | 0.8753 | 0.8294 | 0.8286 | 0.8523 | 0.1786 | 0.8491 | 3.0092 | 49.9 |
| 2 | 2.2725 | 0.9029 | 0.9007 | 0.8837 | 0.9018 | 0.1667 | 0.8547 | 5.3294 | 27.8 |
| 3 | 1.9621 | 0.9112 | 0.917 | 0.9083 | 0.9141 | 0.1667 | 0.8726 | 6.8561 | 21.9 |
| 4 | 1.8178 | 0.8873 | 0.8985 | 0.8958 | 0.8929 | 0.1667 | 0.8506 | 6.3445 | 23.7 |
| 5 | 1.6392 | 0.8488 | 0.8652 | 0.8627 | 0.857 | 0.1667 | 0.8147 | 6.5895 | 22.8 |
| 6 | 1.5665 | 0.8766 | 0.8937 | 0.8716 | 0.8852 | 0.1667 | 0.8371 | 6.5554 | 22.9 |
| 7 | 1.471 | 0.9113 | 0.9113 | 0.8968 | 0.9113 | 0.1667 | 0.8711 | 6.5622 | 22.9 |
| 8 | 1.4184 | 0.9178 | 0.9209 | 0.9027 | 0.9194 | 0.1667 | 0.8799 | 6.586 | 22.7 |
| 9 | 1.3821 | 0.9199 | 0.9237 | 0.9111 | 0.9218 | 0.3333 | 0.8838 | 6.7605 | 22.2 |
| 10 (best) | 1.363 | 0.9237 | 0.9209 | 0.9174 | 0.9223 | 0.3333 | 0.8841 | 6.8248 | 22.0 |

Candidates for the drawing-level check: the pass chosen by val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS, pass 10 (`weights/rtdetr_r18_fonts_ema`, the final weights), and the last pass, pass 10 (`weights/rtdetr_r18_fonts_ema/pass-10`), fully annealed; every pass's weights are kept.

Passes 1, 2 ran below 85% of the run's best rate (6.8561 images/s), slowed by other work on the machine: those minutes are contended.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 96% of the symbols, and 3% of what it reports is a false alarm (176 on 1236 tiles, 55 of them at a tile edge). It misses too many protection relays (finds 21%), isolators (finds 76%), meters (finds 86%), surge arresters (finds 90%). It reports some symbols that are not there, mostly protection relays (67% of those reported are wrong), contactors (25% of those reported are wrong), downstream boards (15% of those reported are wrong). It finds 73% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 677 false alarms on the same tiles instead of 176.

## rtdetr_r18vd (`weights/rtdetr_r18_plain_ema_fix`)

- 10 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 3, EMA decay 0.999 with 200 warm-up updates (d 0.9990 after 5730 updates, an average over about 1000 steps, 1.7 passes of 573 steps; the authors' 0.9999 with 2000 warm-up updates, tuned to 72 epochs of hundreds of thousands of steps, would average about 18 steps on this run, so the decay is set for its length; the evaluated weights are the average): ended by the cap
- Scored with the best pass's weights, epoch 10 (val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS 0.9169); 274 min of training and 3.1 min of final evaluation, 5.8765 training images/s
- Thresholds chosen on val and val_unseen pooled at training time: with NMS 0.06, raw 0.08; NMS is class-agnostic at IoU 0.5
- eval of val, val_unseen, test_clean, test_scan, test_unseen after training reproduces the training-time metrics exactly

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 936 | with NMS | 0.9246 | 0.958 | 0.8571 (relay) | 0.8955 | 0.4 (relay) |
| val | 936 | raw | 0.872 | 0.8281 | 0.6488 (ct) | 0.8545 | 0.3667 (relay) |
| val_unseen | 396 | with NMS | 0.9299 | 0.9551 | 0.8571 (contactor) | 0.8926 | 0.5 (relay) |
| val_unseen | 396 | raw | 0.8908 | 0.8404 | 0.6412 (ct) | 0.8653 | 0.5 (relay) |
| test_clean | 348 | with NMS | 0.9121 | 0.9615 | 0.8636 (contactor) | 0.8878 | 0.5385 (relay) |
| test_clean | 348 | raw | 0.8616 | 0.7901 | 0.52 (board) | 0.8668 | 0.4615 (relay) |
| test_scan | 348 | with NMS | 0.9074 | 0.9616 | 0.75 (relay) | 0.843 | 0.2308 (relay) |
| test_scan | 348 | raw | 0.8677 | 0.8971 | 0.7183 (ct) | 0.7903 | 0.1538 (relay) |
| test_unseen | 1140 | with NMS | 0.9368 | 0.9547 | 0.8 (relay) | 0.9043 | 0.6667 (relay) |
| test_unseen | 1140 | raw | 0.8896 | 0.8341 | 0.6626 (ct) | 0.874 | 0.6164 (earth) |
| test_unseen, `pass-10`, threshold 0.06 | 1140 | with NMS | 0.9368 | 0.9547 | 0.8 (relay) | 0.9043 | 0.6667 (relay) |
| test_unseen, `pass-10`, threshold 0.08 | 1140 | raw | 0.8896 | 0.8341 | 0.6626 (ct) | 0.874 | 0.6164 (earth) |
| val, pass 4, the site model, threshold 0.09 | 936 | with NMS | 0.9327 | 0.9471 | 0.7273 (relay) | 0.8932 | 0.5333 (relay) |
| val, pass 4, the site model, threshold 0.10 | 936 | raw | 0.911 | 0.8852 | 0.7416 (earth) | 0.8791 | 0.5333 (relay) |
| val_unseen, pass 4, the site model, threshold 0.09 | 396 | with NMS | 0.9328 | 0.8689 | 0.0 (board) | 0.8919 | 0.6 (relay) |
| val_unseen, pass 4, the site model, threshold 0.10 | 396 | raw | 0.9057 | 0.805 | 0.0 (board) | 0.8607 | 0.4 (relay) |
| test_clean, pass 4, the site model, threshold 0.09 | 348 | with NMS | 0.9195 | 0.971 | 0.907 (board) | 0.8776 | 0.5385 (relay) |
| test_clean, pass 4, the site model, threshold 0.10 | 348 | raw | 0.904 | 0.8922 | 0.6724 (earth) | 0.8713 | 0.5385 (relay) |
| test_scan, pass 4, the site model, threshold 0.09 | 348 | with NMS | 0.9013 | 0.9464 | 0.4286 (relay) | 0.8508 | 0.2308 (relay) |
| test_scan, pass 4, the site model, threshold 0.10 | 348 | raw | 0.8707 | 0.8617 | 0.4286 (relay) | 0.8353 | 0.2308 (relay) |
| test_unseen, pass 4, the site model, threshold 0.09 | 1140 | with NMS | 0.9389 | 0.9517 | 0.75 (relay) | 0.9004 | 0.6062 (earth) |
| test_unseen, pass 4, the site model, threshold 0.10 | 1140 | raw | 0.9149 | 0.8765 | 0.6744 (relay) | 0.8867 | 0.5719 (earth) |
| test_unseen, `pass-7`, threshold 0.07 | 1140 | with NMS | 0.923 | 0.952 | 0.8 (relay) | 0.8922 | 0.6267 (earth) |
| test_unseen, `pass-7`, threshold 0.10 | 1140 | raw | 0.8878 | 0.8754 | 0.7391 (capacitor) | 0.8387 | 0.5034 (earth) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9848 | 0.9484 | 0.9889 | 0.9419 | fail |
| fuse_switch | 1.0 | 0.9805 | 0.9937 | 0.9692 | pass |
| isolator | 0.973 | 0.9643 | 0.9686 | 0.939 | fail |
| contactor | 0.8949 | 0.9914 | 0.897 | 0.997 | fail |
| transformer | 0.9799 | 1.0 | 0.9948 | 0.9847 | pass |
| generator | 0.98 | 0.8305 | 1.0 | 0.8214 | fail |
| motor | 0.972 | 0.9083 | 0.9966 | 0.8735 | fail |
| board | 0.9333 | 0.9211 | 0.9259 | 0.9615 | fail |
| ct | 0.9693 | 1.0 | 0.9604 | 0.9909 | pass |
| meter | 0.9554 | 0.9804 | 0.9181 | 0.9953 | fail |
| relay | 0.8571 | 0.4 | 0.8 | 0.6667 | fail |
| spd | 0.9787 | 0.8364 | 0.9645 | 0.8859 | fail |
| capacitor | 0.9462 | 1.0 | 0.9623 | 0.9444 | fail |
| earth | 0.987 | 0.7755 | 0.995 | 0.6884 | fail |

Size rule: 3 of 14 classes pass; fails on breaker, isolator, contactor, generator, motor, board, meter, relay, spd, capacitor, earth.

Per class with NMS for pass 4, the site model, and the size rule on its val and test_unseen:

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9855 | 0.9589 | 0.9868 | 0.9444 | fail |
| fuse_switch | 0.9982 | 0.9823 | 0.99 | 0.9766 | pass |
| isolator | 0.9725 | 0.9464 | 0.9933 | 0.9024 | fail |
| contactor | 0.8842 | 0.9871 | 0.8932 | 0.9819 | fail |
| transformer | 0.9799 | 1.0 | 0.9949 | 1.0 | pass |
| generator | 1.0 | 0.9153 | 1.0 | 0.9881 | fail |
| motor | 0.9596 | 0.8297 | 0.9858 | 0.8373 | fail |
| board | 0.9653 | 0.9145 | 0.8644 | 0.9808 | fail |
| ct | 0.9571 | 0.9873 | 0.9954 | 0.9909 | pass |
| meter | 0.9202 | 0.9804 | 0.9383 | 0.9953 | fail |
| relay | 0.7273 | 0.5333 | 0.75 | 0.7143 | fail |
| spd | 0.9885 | 0.7818 | 1.0 | 0.7609 | fail |
| capacitor | 0.9355 | 0.9886 | 0.9434 | 0.9259 | fail |
| earth | 0.9856 | 0.699 | 0.9888 | 0.6062 | fail |

Size rule for pass 4, the site model: 3 of 14 classes pass.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val_unseen mAP@0.5 with NMS | pooled mean F1 | mean mAP@0.5 | val_unseen worst min(P, R) | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.0747 | 0.8337 | 0.8134 | 0.8218 | 0.8236 | 0.4667 | 0.8075 | 5.7581 | 28.3 |
| 2 | 2.214 | 0.9075 | 0.9043 | 0.8718 | 0.9059 | 0.3478 | 0.8807 | 5.6165 | 28.6 |
| 3 | 1.9249 | 0.9218 | 0.9209 | 0.9002 | 0.9214 | 0.6 | 0.8991 | 5.7958 | 27.7 |
| 4 | 1.8261 | 0.9327 | 0.9328 | 0.9142 | 0.9327 | 0.6 | 0.911 | 5.7834 | 27.8 |
| 5 | 1.6732 | 0.9236 | 0.9288 | 0.9035 | 0.9262 | 0.5536 | 0.8952 | 5.9252 | 27.1 |
| 6 | 1.621 | 0.9174 | 0.9084 | 0.9 | 0.9129 | 0.4706 | 0.8864 | 5.9799 | 26.9 |
| 7 | 1.5238 | 0.9306 | 0.8999 | 0.9107 | 0.9153 | 0.4 | 0.8961 | 6.1944 | 25.9 |
| 8 | 1.4824 | 0.9196 | 0.915 | 0.9088 | 0.9173 | 0.5882 | 0.8745 | 6.2638 | 25.8 |
| 9 | 1.4219 | 0.9207 | 0.9239 | 0.9146 | 0.9223 | 0.5 | 0.8718 | 5.7564 | 28.0 |
| 10 (best) | 1.4131 | 0.9246 | 0.9299 | 0.9169 | 0.9272 | 0.5 | 0.872 | 5.6915 | 28.2 |

Candidates for the drawing-level check: the pass chosen by val and val_unseen pooled: mean per-class F1 at the operating threshold with NMS, pass 10 (`weights/rtdetr_r18_plain_ema_fix`, the final weights), and the last pass, pass 10 (`weights/rtdetr_r18_plain_ema_fix/pass-10`), fully annealed; every pass's weights are kept.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 93% of the symbols, and 2% of what it reports is a false alarm (126 on 1140 tiles, 49 of them at a tile edge). It misses too many protection relays (finds 67%), earth symbols (finds 69%), generators (finds 82%), motors (finds 87%). It reports some symbols that are not there, mostly protection relays (20% of those reported are wrong), contactors (10% of those reported are wrong), meters (8% of those reported are wrong). Scan-like copies are harder: it finds 92% of their symbols against 94% on clean copies. It finds 96% of the breakers drawn as a box with a diagonal and 93% of the IEC breakers. It finds 56% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 983 false alarms on the same tiles instead of 126.

## More passes

To give a model more passes than its cap, start a new run from its trained weights with its own fresh, shorter and gentler schedule (half the learning rates, a 100-step warm-up, the same early stopping; the run's own pass selection, kept passes, weight average and sampling stay as they were). It writes a new directory with its own card and metrics, scored with its best pass, and leaves the overnight run as it is. Replace N with the number of passes:

- `weights/rtdetr_r18_full` (rtdetr_r18vd): one pass with its val took about 20 min tonight, so N passes take about N times 20 min, then about 2 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_full --out weights/rtdetr_r18_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0`

- `weights/rtdetr_r18_rare` (rtdetr_r18vd): one pass with its val took about 25 min tonight at its fastest (other passes were slowed by other work), so N passes take about N times 25 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_rare --out weights/rtdetr_r18_rare_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0 --select f1 --keep-passes --rare-repeat 0.15 --rare-group class`

- `weights/rtdetr_r18_plain` (rtdetr_r18vd): one pass with its val took about 23 min tonight, so N passes take about N times 23 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_plain --out weights/rtdetr_r18_plain_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0 --select f1 --keep-passes`

- `weights/rtdetr_r18_fonts` (rtdetr_r18vd): one pass with its val took about 19 min tonight at its fastest (other passes were slowed by other work), so N passes take about N times 19 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco_fonts --model weights/rtdetr_r18_fonts --out weights/rtdetr_r18_fonts_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0 --select f1 --keep-passes`

- `weights/rtdetr_r18_plain_ema` (rtdetr_r18vd): one pass with its val took about 22 min tonight at its fastest (other passes were slowed by other work), so N passes take about N times 22 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_plain_ema --out weights/rtdetr_r18_plain_ema_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0 --select f1 --keep-passes --ema 0.999 --ema-warmup 200`

- `weights/rtdetr_r18_fonts_ema` (rtdetr_r18vd): one pass with its val took about 22 min tonight at its fastest (other passes were slowed by other work), so N passes take about N times 22 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco_fonts --model weights/rtdetr_r18_fonts_ema --out weights/rtdetr_r18_fonts_ema_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0 --select f1 --keep-passes --ema 0.999 --ema-warmup 200`

- `weights/rtdetr_r18_plain_ema_fix` (rtdetr_r18vd): one pass with its val took about 27 min tonight, so N passes take about N times 27 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_plain_ema_fix --out weights/rtdetr_r18_plain_ema_fix_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0 --select f1 --keep-passes --ema 0.999 --ema-warmup 200`

