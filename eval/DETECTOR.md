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

## rtdetr_r50vd (`weights/rtdetr_r50_full`)

- 6 passes of a cap of 8 at batch 8, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 3: early stop after epoch 6 of 8: 3 val passes in a row without an improvement of 0.002 in val mAP@0.5 with NMS; the best pass was epoch 3 at 0.9048
- Scored with the best pass's weights, epoch 3 (val mAP@0.5 with NMS 0.9048); 278 min of training and 3.4 min of final evaluation, 3.4073 training images/s
- Thresholds chosen on val at training time: with NMS 0.10, raw 0.17; NMS is class-agnostic at IoU 0.5
- eval of val, test_clean, test_scan after training reproduces the training-time metrics exactly; evaluated after training on a newer export, at the thresholds chosen then, and shown so: val_unseen (396 tiles, export 3f19db3a), test_unseen (1140 tiles, export 3f19db3a)

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 936 | with NMS | 0.9048 | 0.9443 | 0.7872 (earth) | 0.8669 | 0.1333 (relay) |
| val | 936 | raw | 0.8909 | 0.8827 | 0.6824 (spd) | 0.8037 | 0.0667 (relay) |
| test_clean | 348 | with NMS | 0.8977 | 0.9706 | 0.8805 (fuse_switch) | 0.8605 | 0.1538 (relay) |
| test_clean | 348 | raw | 0.8744 | 0.8609 | 0.6667 (spd) | 0.8013 | 0.0 (relay) |
| test_scan | 348 | with NMS | 0.9136 | 0.9379 | 0.6957 (earth) | 0.8784 | 0.1538 (relay) |
| test_scan | 348 | raw | 0.8968 | 0.887 | 0.5946 (spd) | 0.8012 | 0.0 (relay) |
| test_unseen | 1140 | with NMS | 0.8872 | 0.9366 | 0.8267 (earth) | 0.8599 | 0.1429 (relay) |
| test_unseen | 1140 | raw | 0.8796 | 0.8842 | 0.668 (spd) | 0.7973 | 0.0476 (relay) |
| val_unseen | 396 | with NMS | 0.895 | 0.9067 | 0.6582 (earth) | 0.8643 | 0.0 (relay) |
| val_unseen | 396 | raw | 0.8771 | 0.883 | 0.6739 (spd) | 0.8144 | 0.0 (relay) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9805 | 0.9724 | 0.987 | 0.9277 | fail |
| fuse_switch | 0.8766 | 0.9823 | 0.8499 | 0.9766 | fail |
| isolator | 0.9569 | 0.9911 | 0.973 | 0.878 | fail |
| contactor | 0.9502 | 0.9871 | 0.9226 | 0.9699 | fail |
| transformer | 1.0 | 0.9726 | 0.9948 | 0.9847 | pass |
| generator | 0.9508 | 0.9831 | 0.9438 | 1.0 | fail |
| motor | 0.9739 | 0.9782 | 0.9877 | 0.9699 | pass |
| board | 1.0 | 0.4934 | 0.9268 | 0.7308 | fail |
| ct | 0.9865 | 0.9241 | 1.0 | 0.8545 | fail |
| meter | 0.9643 | 0.8824 | 0.9462 | 0.8224 | fail |
| relay | 1.0 | 0.1333 | 1.0 | 0.1429 | fail |
| spd | 0.8629 | 0.9727 | 0.8294 | 0.9511 | fail |
| capacitor | 0.931 | 0.9205 | 0.9238 | 0.8981 | fail |
| earth | 0.7872 | 0.9439 | 0.8267 | 0.9315 | fail |

Size rule: 2 of 14 classes pass; fails on breaker, fuse_switch, isolator, contactor, generator, board, ct, meter, relay, spd, capacitor, earth.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|
| 1 | 14.6513 | 0.8258 | 0.7727 | 3.3643 | 46.9 |
| 2 | 7.4291 | 0.8396 | 0.824 | 3.4162 | 46.2 |
| 3 (best) | 7.0493 | 0.9048 | 0.8909 | 3.4248 | 46.1 |
| 4 | 6.4192 | 0.8779 | 0.8717 | 3.4103 | 46.3 |
| 5 | 5.6431 | 0.8804 | 0.8809 | 3.4163 | 46.2 |
| 6 | 5.432 | 0.8436 | 0.8605 | 3.4118 | 46.2 |

What it still gets wrong: On the unseen-style sheets rtdetr_r50vd finds 93% of the symbols, and 6% of what it reports is a false alarm (325 on 1140 tiles, 105 of them at a tile edge). It misses too many protection relays (finds 14%), downstream boards (finds 73%), meters (finds 82%), current transformers (finds 85%). It reports some symbols that are not there, mostly earth symbols (17% of those reported are wrong), surge arresters (17% of those reported are wrong), fuse switches (15% of those reported are wrong). It finds 94% of the breakers drawn as a box with a diagonal and 91% of the IEC breakers. It finds 67% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 987 false alarms on the same tiles instead of 325.

## rtdetr_r18vd continued from rtdetr_r18_full (`weights/rtdetr_r18_more`)

- Continues `weights/rtdetr_r18_full`, whose best pass scored val mAP@0.5 with NMS 0.9286 and test_unseen 0.9262
- 3 passes of a cap of 3 at batch 16, lr 5e-05 (backbone 5e-06), warm-up 100 steps, seed 0, early stopping patience 3: ended by the cap
- Scored with the best pass's weights, epoch 2 (val mAP@0.5 with NMS 0.936); 81 min of training and 2.7 min of final evaluation, 6.0296 training images/s
- Thresholds chosen on val at training time: with NMS 0.10, raw 0.14; NMS is class-agnostic at IoU 0.5
- eval of val, test_clean, test_scan after training reproduces the training-time metrics exactly; evaluated after training on a newer export, at the thresholds chosen then, and shown so: val_unseen (396 tiles, export 3f19db3a), test_unseen (1140 tiles, export 3f19db3a)

| split | tiles | post-processing | mAP@0.5 | mean P | worst P | mean R | worst R |
|---|---|---|---|---|---|---|---|
| val | 936 | with NMS | 0.936 | 0.9485 | 0.85 (board) | 0.8718 | 0.6939 (earth) |
| val | 936 | raw | 0.8967 | 0.907 | 0.8655 (board) | 0.8177 | 0.5663 (earth) |
| test_clean | 348 | with NMS | 0.9377 | 0.9538 | 0.8387 (earth) | 0.8557 | 0.5 (earth) |
| test_clean | 348 | raw | 0.9012 | 0.9247 | 0.7308 (earth) | 0.7852 | 0.3654 (earth) |
| test_scan | 348 | with NMS | 0.8422 | 0.9452 | 0.7 (relay) | 0.7739 | 0.4167 (capacitor) |
| test_scan | 348 | raw | 0.805 | 0.8884 | 0.7037 (earth) | 0.7114 | 0.3654 (earth) |
| test_unseen | 1140 | with NMS | 0.9177 | 0.9339 | 0.7031 (board) | 0.838 | 0.5714 (generator) |
| test_unseen | 1140 | raw | 0.8765 | 0.898 | 0.7692 (board) | 0.7769 | 0.4555 (earth) |
| val_unseen | 396 | with NMS | 0.9295 | 0.86 | 0.0 (board) | 0.8386 | 0.4821 (earth) |
| val_unseen | 396 | raw | 0.8842 | 0.8241 | 0.0 (board) | 0.7842 | 0.4286 (earth) |

Per class with NMS, and the size rule (precision and recall at least 0.95 on val and on test_unseen):

| class | val P | val R | test_unseen P | test_unseen R | rule |
|---|---|---|---|---|---|
| breaker | 0.9868 | 0.8786 | 0.9814 | 0.8559 | fail |
| fuse_switch | 0.9942 | 0.9167 | 0.9836 | 0.8855 | fail |
| isolator | 0.9545 | 0.9375 | 0.9728 | 0.872 | fail |
| contactor | 0.9404 | 0.9526 | 0.955 | 0.9578 | fail |
| transformer | 0.9286 | 0.9795 | 0.9554 | 0.9847 | fail |
| generator | 1.0 | 0.7458 | 1.0 | 0.5714 | fail |
| motor | 0.9583 | 0.9039 | 0.9855 | 0.8193 | fail |
| board | 0.85 | 0.7829 | 0.7031 | 0.8654 | fail |
| ct | 0.9455 | 0.9873 | 0.9909 | 0.9864 | fail |
| meter | 0.9592 | 0.9216 | 0.9565 | 0.9252 | fail |
| relay | 0.8667 | 0.8667 | 0.7447 | 0.8333 | fail |
| spd | 1.0 | 0.7636 | 0.9684 | 0.8315 | fail |
| capacitor | 1.0 | 0.875 | 0.9222 | 0.7685 | fail |
| earth | 0.8947 | 0.6939 | 0.9545 | 0.5753 | fail |

Size rule: 0 of 14 classes pass; fails on breaker, fuse_switch, isolator, contactor, transformer, generator, motor, board, ct, meter, relay, spd, capacitor, earth.

Training curve, val after each epoch:

| epoch | train loss | val mAP@0.5 with NMS | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|
| 1 | 2.4373 | 0.8881 | 0.8345 | 7.253 | 21.9 |
| 2 (best) | 2.2518 | 0.936 | 0.8967 | 6.1939 | 25.6 |
| 3 | 2.1152 | 0.9298 | 0.8929 | 4.6418 | 33.9 |

Pass 3 ran below 85% of the run's best rate (7.253 images/s), slowed by other work on the machine: those minutes are contended.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd continued from rtdetr_r18_full finds 86% of the symbols, and 3% of what it reports is a false alarm (146 on 1140 tiles, 59 of them at a tile edge). It misses too many generators (finds 57%), earth symbols (finds 58%), capacitors (finds 77%), motors (finds 82%). It reports some symbols that are not there, mostly downstream boards (30% of those reported are wrong), protection relays (26% of those reported are wrong), capacitors (8% of those reported are wrong). Scan-like copies are harder: it finds 82% of their symbols against 89% on clean copies. It finds 88% of the breakers drawn as a box with a diagonal and 84% of the IEC breakers. It finds 41% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 367 false alarms on the same tiles instead of 146.

## rtdetr_r18vd (`weights/rtdetr_r18_rare`)

- 10 passes of a cap of 10 at batch 16, lr 1e-04 (backbone 1e-05), warm-up 300 steps, seed 0, early stopping patience 0, rare-class sampling at T 0.15: ended by the cap
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

| epoch | train loss | val mAP@0.5 with NMS | val_unseen mAP@0.5 with NMS | pooled mean F1 | val_unseen worst min(P, R) | val mAP@0.5 raw | images/s | minutes |
|---|---|---|---|---|---|---|---|---|
| 1 | 5.9317 | 0.8078 | 0.768 | 0.7946 | 0.2286 | 0.7788 | 5.3444 | 33.6 |
| 2 | 2.7728 | 0.8868 | 0.8785 | 0.856 | 0.4643 | 0.8447 | 6.0457 | 30.0 |
| 3 | 2.4795 | 0.8241 | 0.8255 | 0.7946 | 0.2857 | 0.7782 | 5.7438 | 31.1 |
| 4 (best) | 2.3617 | 0.9281 | 0.9403 | 0.9201 | 0.625 | 0.8878 | 5.7228 | 31.5 |
| 5 | 2.1978 | 0.9055 | 0.9242 | 0.8696 | 0.32 | 0.8505 | 5.8365 | 30.7 |
| 6 | 2.0652 | 0.8978 | 0.8639 | 0.8449 | 0.4 | 0.8497 | 5.7826 | 31.1 |
| 7 | 1.8983 | 0.9051 | 0.9069 | 0.8616 | 0.4348 | 0.8517 | 5.6478 | 31.6 |
| 8 | 1.8248 | 0.9207 | 0.8925 | 0.8803 | 0.5517 | 0.8844 | 7.261 | 25.1 |
| 9 | 1.7724 | 0.9162 | 0.914 | 0.8743 | 0.6071 | 0.8804 | 7.2692 | 24.9 |
| 10 | 1.7564 | 0.919 | 0.9115 | 0.8874 | 0.6562 | 0.8821 | 7.3242 | 24.7 |

Passes 1, 2, 3, 4, 5, 6, 7 ran below 85% of the run's best rate (7.3242 images/s), slowed by other work on the machine: those minutes are contended.

What it still gets wrong: On the unseen-style sheets rtdetr_r18vd finds 86% of the symbols, and 2% of what it reports is a false alarm (88 on 1140 tiles, 46 of them at a tile edge). It misses too many surge arresters (finds 66%), earth symbols (finds 71%), protection relays (finds 76%), capacitors (finds 81%). It reports some symbols that are not there, mostly downstream boards (16% of those reported are wrong), earth symbols (10% of those reported are wrong), capacitors (5% of those reported are wrong). Scan-like copies are harder: it finds 85% of their symbols against 87% on clean copies. It finds 51% of the symbols cut by a tile edge; each of these is whole in a neighbouring tile, where the reader can find it. Without duplicate suppression, at its own best threshold, it would report 842 false alarms on the same tiles instead of 88.

## More passes

To give a model more passes than its cap, start a new run from its trained weights with its own fresh, shorter and gentler schedule (half the learning rates, a 100-step warm-up, the same early stopping). It writes a new directory with its own card and metrics, scored with its best pass, and leaves the overnight run as it is. Replace N with the number of passes:

- `weights/rtdetr_r18_full` (rtdetr_r18vd): one pass with its val took about 20 min tonight, so N passes take about N times 20 min, then about 2 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_full --out weights/rtdetr_r18_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0`

- `weights/rtdetr_r50_full` (rtdetr_r50vd): one pass with its val took about 46 min tonight, so N passes take about N times 46 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r50_full --out weights/rtdetr_r50_more --epochs N --batch 8 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0`

- `weights/rtdetr_r18_more` (rtdetr_r18vd): one pass with its val took about 22 min tonight at its fastest (other passes were slowed by other work), so N passes take about N times 22 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_more --out weights/rtdetr_r18_more_more --epochs N --batch 16 --lr 3e-05 --backbone-lr 3e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0`

- `weights/rtdetr_r18_rare` (rtdetr_r18vd): one pass with its val took about 25 min tonight at its fastest (other passes were slowed by other work), so N passes take about N times 25 min, then about 3 min of final evaluation:

  `PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is python3 -m sld2quote.train_detector train --data data/coco --model weights/rtdetr_r18_rare --out weights/rtdetr_r18_rare_more --epochs N --batch 16 --lr 5e-05 --backbone-lr 5e-06 --warmup-steps 100 --patience 3 --min-delta 0.002 --nms-iou 0.5 --device mps --seed 0`

