---
title: sld2quote
sdk: docker
app_port: 7860
license: mit
short_description: Single-line diagram to quotation, synthetic drawings only
---

# sld2quote

A miniature single-line-diagram-to-quotation pipeline: a trained symbol detector, a text reader, deterministic rules, a
bill of materials with evidence on every line and an illustrative quotation, scored live against the answer keys of
generated drawings. Synthetic drawings only; prices are illustrative.

Source and how to reproduce: [github.com/CaptiveON/sld2quote](https://github.com/CaptiveON/sld2quote). Detector:
[itsmoadnan/sld2quote-detector](https://huggingface.co/itsmoadnan/sld2quote-detector).
