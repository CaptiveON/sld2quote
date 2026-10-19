# Problems met while building sld2quote, in plain language

A running record, kept as the work goes, of every real problem the project hit, why it happened, what was done, and what it means for
the full system after the KTP. Everything here was measured on synthetic drawings; nothing is an R&B drawing, part or price.
Dates are 2026.

## 1. Scope of symbols: a drawing has far more than breakers and transformers

**What happened (15 Oct).** The first version drew three shapes: a box with a diagonal for every breaker, two circles for a transformer, a
circle with a G for a generator. An estimator's take-off also counts fuse switches, isolators, contactors and motors, power-factor
correction, surge protection, current transformers, meters and protection relays, and each of those changes the price.

**Why it matters.** A detector only finds what it was trained on. Trained on three shapes, it would miss or mislabel every other symbol on
a real drawing, and a high score on the narrow set would prove nothing.

**What was done.** The generator now draws 16 device types with IEC 60617 symbols (seven bodies from an open library, credited under
CC BY 3.0, the rest drawn in the same style), breakers in both the standard style (switch with a cross) and the common box style, and the
pricing rules price all of them. Still missing: a changeover-switch symbol, more layout families, and the American ANSI/IEEE 315 style.

**For the real project.** The symbol vocabulary must be agreed with the estimators first, and every symbol must be priced by a rule,
or the detector's output has nowhere to go.

## 2. There is no public dataset of labelled single-line diagrams

**What happened.** No labelled set exists, and R&B's drawings cannot be used before the project starts. So the training data is synthetic:
a generator that draws switchboards and writes the answer key (every device with its box and ratings, every piece of text with its box).

**Why it matters.** The model is only as realistic as the generator. Scores on synthetic drawings are an upper bound; real drawings will
score lower.

**What was done.** Two levels of board (simple and medium), style variation (fonts, sizes, line widths, label placement), a scan-like copy
of every sheet, and a small real test set of 19 genuine public-domain US federal one-line diagrams, with the provenance of each recorded.
The real set uses the American symbol style, which the model never saw, and only six of its sheets carry ampere ratings, so it is reported
separately and honestly.

**For the real project.** The first weeks are labelling: real sheets with boxes and types, which both measures the gap and closes it.

## 3. The generator is a product too: labels were drawn over symbols

**What happened (15 Oct).** An audit of 200 generated sheets found 16 percent had a label printed across a breaker, some labels overlapping
each other, and some running off the page. A model trained on that learns to accept text over symbols.

**What was done.** The page widens until the widest label clears the next way; an audit script and a test check 300 sheets for any text
over a symbol, text over text, anything off the page or overlapping symbols. The count is now zero.

**For the real project.** Synthetic data needs the same tests as code. "It looks fine" on a few sheets is not a check.

## 4. The answer key for scanned copies was in the wrong frame

**What happened (15 Oct).** Scan-like copies are rotated by up to 1.2 degrees after drawing, but their answer-key boxes were still in the
unrotated frame, up to 35 pixels off on wide pages. The model would have learned wrong positions on every scan copy.

**What was done.** Every box is mapped through the same rotation, and a test draws a box, rotates the image and checks the mapped box
against the ink. A second test checks that every answer-key box hugs the drawn symbol within 3 pixels.

**For the real project.** Every label must be checked against the pixels it claims to describe; a label file can be internally consistent
and still wrong.

## 5. Why the classical reader found 9 percent of symbols

**What happened (17 Oct baseline).** Hand-written shape tests only knew box-style breakers and circles. On the first sheet, the main
breaker was found by the shape test and then thrown away as "text" because the OCR had read its diagonal as a backslash; the transformer
was read as the digit 8.

**What it means.** Rules written for one drawing style break on the next. That is the case for a trained detector, and the baseline
number (9 percent found, 35 percent of reports right) is the "before" every later model is compared with.

## 6. The detector drew two boxes on one symbol

**What happened (17 Oct).** The first trained model found 81 percent of symbols but 92 percent of its false alarms were a second box on a
symbol it had already found.

**Why.** This kind of detector (a DETR) has no built-in rule against duplicates. Each of its 300 "slots" learns, over a long training
schedule, to back off when another slot already holds the symbol. The reference recipe is 72 passes over the data; the first run was 2.

**What was done.** Longer schedules, and a duplicate filter (class-agnostic non-maximum suppression) as part of the detector's output
anyway, because the page is read in overlapping tiles and a quoting system must never count a part twice. Scores are reported with and
without the filter.

**For the real project.** Defence in depth: the model may be imperfect; the pipeline must make a double count impossible (filter within a
tile, merge across tiles, one device per box, a rule that flags near-identical devices, a review page that shows every box).

## 7. Early stopping and a decaying learning-rate schedule do not mix

**What happened (18 Oct).** The protocol was "stop when the validation score has not improved for three passes, with a cap". Both overnight
runs stopped at pass 6 while the learning rate was still high, because the schedule was sized to the cap, so the gentle finishing phase
never ran. A short run that had finished its schedule scored the same or better.

**What was done.** Early stopping was switched off for later runs; the schedule runs to its cap and the best pass is kept; every pass's
weights are kept so another selection rule can be applied without retraining.

**For the real project.** Early stopping belongs with schedules that react to plateaus, not with a schedule that decays on a fixed clock.
This was the orchestrator's own protocol error, caught by the measurements.

## 8. Finishing the schedule helped the seen styles and hurt the unseen one

**What happened (18 Oct).** Three gentle passes from the best weights lifted the validation score (0.929 to 0.936) and tightened the
boxes, but on the unseen style the generator fell from 98 percent found to 50 percent and the earth symbol from 72 to 55.

**Why.** The symbols with a letter inside (G in a generator, M in a motor, the meter units, I> in a relay) depend on the font. The
held-out font draws those letters differently, and the extra passes fitted the four training fonts more closely.

**What was done.** A second validation set of held-out-style drawings, disjoint from the test set, so the best pass is chosen on both
seen and unseen styles while the test set stays untouched. A wider font set in the generator (16 fonts) for a new training run.

**Correction (18 Oct, later the same day).** The engineer who widened the fonts checked the generator and found that the letters inside
symbols were drawn in Arial Bold on every sheet, whatever the sheet's font. So the held-out font never changed a glyph inside a symbol,
and the explanation above cannot be literally right; what the unseen style changes is the text around the symbols and, far more often,
the layout: 49 of the 60 unseen-style test sheets have their labels below the symbols, a layout the model never trained on, against 16
in a held-out font. The lesson is the same as in entries 10 and 10a: an explanation that fits the numbers still has to be checked against
the code. In response, the wide font set now draws the inner glyphs in the sheet's font, the model trains on both label layouts, and the
hold-out is fonts only (three of sixteen), so the "unseen" test measures one thing.

**For the real project.** Font and drafting style vary by consultant; the training set must span them, and the validation set must contain
styles the model was not trained on, or the selection step will quietly pick the model that generalises worst.

## 9. The selection metric picked the wrong model

**What happened (18 Oct).** Choosing the best pass by mean detection score (mAP@0.5) averaged over seen and unseen validation would have
chosen the finishing run, which the test set shows generalises worse. Mean per-class accuracy at the operating threshold (F1) chose the
right one in every comparison.

**Why.** mAP averages over classes and over thresholds, so one class collapsing at the actual operating point barely moves it. The
operating-point measure is closer to the product rule (95 percent precision and recall per class).

**What was done.** Selection on pooled per-class F1 at the operating threshold; every pass's selection scores recorded.

## 10. Rare symbols, and a hypothesis that measurement killed

**What happened (18 Oct).** Relay is 0.7 percent of training boxes, capacitor 1.8, surge arrester 3.2. A run that showed tiles with rare
classes more often (repeat-factor sampling, the LVIS method) halved the false alarms and raised precision across classes, but recall was
mixed, and at drawing level it had stopped finding the box-style bus-section, which made every quote with one worse.

The first explanation, that class-level sampling had taken weight from the bus-section (a rare appearance of the commonest class), was
tested and found wrong: the bus-section is in 16.7 percent of training tiles, so its sampling weight was unchanged. The loss is in the
detector or in the pass selection, and a run-to-run variance baseline (the same recipe without sampling, full schedule) is the control
that was missing.

**For the real project.** Per-tile scores are not the product score; a model that looks better on tiles can price worse. And every
hypothesis about a model gets measured before it is acted on, including the orchestrator's.

## 10a. The result depends on which pass you pick, far more than it should

**What happened (18 Oct).** Across the ten kept passes of one run, recall for a device type swung by up to 0.3 between adjacent passes,
even late in the schedule (motor 0.84 then 0.60; bus-section 0.43 then 0.77). The pass picked by the selection rule happened to be the
worst pass for bus-sections. So "which pass" was dominating the per-type results, and comparisons between runs were partly comparisons
between lucky and unlucky passes.

**What the authors do.** RT-DETR's own training recipe (checked at its source, the lyuwenyu/RT-DETR repository: optimizer.yml and
ema.py) keeps an exponential moving average of the weights (decay 0.9999 with a 2,000-step warm-up, updated after every optimiser step)
and evaluates the averaged weights, not the live ones; it even trains at a constant learning rate for 72 passes and lets the average do
the smoothing. The Hugging Face port used here provides the model but not that part of the recipe, so the trainer had no averaging.

**What was done.** Weight averaging added to the trainer as a single, separately measured change: two runs with it (plain data, wider
fonts) queued behind the two without it, so its effect is attributed cleanly. Also a control run of the plain recipe with the full schedule,
which the earlier comparisons lacked.

**For the real project.** Read the authors' recipe, not only the paper; a port of a model is not a port of its training. And measure
run-to-run and pass-to-pass variance before claiming that one change beat another.

## 10b. A threshold chosen on tiles hallucinates parts on whole drawings

**What happened (18 Oct).** The full-schedule control model's operating threshold, chosen per tile to maximise accuracy on validation
tiles, came out at 0.06. On whole drawings that let through 35 to 45 "capacitors" per test set, scored 0.06 to 0.08, sitting on blank paper
with no symbol or text under them, some over 100 pixels wide. The price barely moved only because capacitor ratings (kVAr) are not read yet;
once they are, every ghost becomes a priced power-factor-correction line.

**Why.** Tiles with symbols dominate the tile score, and a low threshold costs little there. A whole drawing is mostly blank paper, where a
low threshold has far more room to invent things. The two measurements disagree, and the product lives at the drawing level.

**What was done.** The operating threshold is now chosen on whole validation drawings (seen and unseen styles pooled), by a sweep over the
detector and the tile merge, and stored with the weights as a separate value from the per-tile one. Test sets are never used for choosing.

**For the real project.** Every threshold and every selection is made on the unit the product is judged by, a whole drawing, on
validation drawings that are not the test drawings; and "false positives on blank paper" is reported as its own number, because each one
is a line on a quote.

## 10c. More variety made the rarest symbol unlearnable with the same number of examples

**What happened (18 Oct, evening).** The run on the wider font set never learned protection relays at all: at its last pass it puts no
box of any class on 38 of 40 of its own training relays, even with the score floor removed, and it lost capacitors and earths in its late
passes. Its validation score fell from pass 4 onwards, which a healthy run never does. A full audit of the data found nothing wrong: relay
counts, box sizes and the drawn "I>" glyph are correct in every font; every box sits on its ink across all 900 drawings; the trainer's own
targets keep 99 percent of the relay boxes with the right class.

**Why, as far as it can be inferred.** Relay is the rarest class (270 training boxes). In the classic set all 270 carry one glyph in one
font; in the wide set they are spread over 13 faces, about 21 each. The same number of examples now has to cover far more variety, so
the class starts weak, and this kind of detector only refines the regions its first stage proposes: a class that is weak early can drop out
for good. Meters vary their glyph too, but there are five times as many of them, and they were learned. Capacitors and earths are a
milder version: their scores drifted below the fixed threshold as training went on, at precision near 0.96, so a threshold chosen later
recovers most of them.

**What was done, in order.** Confirmed first that it was not the data. Then: let the queued run with weight averaging show whether it
keeps relays; if not, show tiles with rare classes more often on this set; choose per-class or drawing-level thresholds; and only as a last
resort draw more relays in the generator.

**For the real project.** Variety and sample size trade off per class, not per dataset: when the drawing styles widen, the rare symbols
need more examples in proportion, and a per-class recall curve across passes is the early warning. One run per condition is not enough
to be sure of a cause; the mechanism here is inferred from the outputs and will be confirmed only by the next runs.

## 11. Tile edges: the per-tile score is a floor, and a tempting fix was wrong

**What happened (18 Oct).** Pages are cut into 640-pixel tiles with a quarter overlap. A symbol cut by a tile edge is found only 40 to 50
percent of the time in that tile, but every symbol is whole in some tile, so drawing-level recall after merging is higher than the tile
score. Two merge filters (drop boxes cut by an inner tile edge; drop a box that mostly contains a better-scoring one) raised drawing-level
precision 2 to 6 points at no recall cost. A refinement that kept a cut box when no whole counterpart existed was measured and rejected:
it regained about 3 true symbols per set and added 6 to 15 false ones.

**For the real project.** The metric that matters is at the drawing level after merging; and a plausible refinement is tested, not
assumed.

## 12. A sign error doubled the tilt of every scan

**What happened (18 Oct).** The "straighten the scan" step rotated the page by minus the measured angle instead of plus, so a 0.77 degree
tilt became 1.43. It was found by the OCR engineer measuring text positions against the generator's ground truth. Fixing it roughly
doubled every scan-copy number for both readers.

**For the real project.** Every stage needs its own ground truth and its own measurement; a bug in a shared step hides inside every
downstream score.

## 13. Text reading: the Mac engine is not the best engine, and the second pass can hurt

**What happened (18 Oct).** Apple Vision, the original engine, reads the labels and notes at 87 percent character accuracy on our sheets.
RapidOCR (an open engine that also runs on Linux) reads them at 96 percent, reads the held-out font at 98 against Vision's 72, and reads
the single letters inside symbols that Vision skips. Its bundled Chinese recogniser drops the spaces in English labels ("ACB-13200A4P"),
so the English one is used. With RapidOCR the close-up second pass around each symbol is worse than the page pass (strokes inside the
window are read as text) and costs 4.75 seconds a sheet. "mm²" comes out as "mm2"; scan noise can put a dot between a number and its unit.

**What else.** On a probe, 26 of 42 wrong ratings had the label read exactly but attached to the wrong device: the link between symbol
and label, not the OCR, is the larger error source.

**For the real project.** Choose the engine by measuring on your own drawings against ground truth, not by reputation; make the label
rules tolerant of the engine's quirks; and treat the symbol-to-label link as its own problem with its own score.

## 14. Rules written for one symbol style broke on the other

**What happened (18 Oct).** The bus-section rule allowed 2 pixels around the busbar line; the standard IEC bus-section symbol sits 4.5 to
6 pixels off it, so 14 of 38 were typed as incoming breakers. On blurred scans the 3-pixel borders of the notes box and title block thicken
to 4 and pass the busbar finder, the "main bus" lands near the bottom of the page, and feeders are typed as incomers.

**For the real project.** Role rules need tolerances derived from the symbol geometry and a main-bus choice by length and thickness; and
they need the same per-type scorecard the detector has.

## 15. Small counts make pass or fail a coin toss

**What happened (18 Oct).** With 24 relay boxes in the unseen test set, a 95 percent recall rule allows one miss. The unseen set was
widened from 20 to 60 drawings (5,495 boxes) before any model was judged on it.

**For the real project.** Set the size of each test set from the rarest class and the rule's margin, not from convenience.

## 16. One machine, several jobs: contention changed the timings

**What happened (17 and 18 Oct).** Evaluations and OCR benchmarks run by other engineers slowed a training run from 20 to 31 minutes per
pass and nearly cut it short at its time budget. The graphics chip was given to training alone; everything else was capped to a few
processor threads; a run was restarted with a longer time backstop so that it would complete its whole schedule.

**For the real project.** Scheduling compute is part of the engineering, and timings in a report must say what else was running.

## 17. Platform quirks on the Mac

**What happened.** One operator of the detector (the backward pass of deformable attention) has no graphics-chip kernel and falls back to
the processor. The large model at batch 16 thrashed the chip's memory (22 GB) and ran at 0.1 images a second; batch 8 was fine. The default
way of starting data-loading workers left 5-second waits per worker after every pass, 20 seconds a pass, until the workers were started
from a fork server. Training on the chip is seeded but not bit-for-bit repeatable. Leftover worker processes from stopped runs had to be
cleaned up by hand.

**For the real project.** These are the kind of details that decide whether a training run fits in a night; they are recorded with the
run so they are not rediscovered.

## 17a. The graphics chip was computing some gradients wrongly

**What happened (18 Oct, late).** A training run crashed within a minute with "matrix contains invalid numeric entries" inside the
matching step: the model's outputs had become NaN. The losses had all been finite, so the damage had come in through the gradients. On a
fixed batch, the trainer's engineer compared every parameter's gradient on the Mac's graphics chip (MPS, PyTorch 2.5.1) with a float64
recomputation on the CPU. Every gradient agreed except one kind: the weight gradient of the three large linear layers in the decoder's
deformable attention (256 by 256 over 134,400 rows), which was wrong by anywhere from 0.08 to 10^36 in six of six repeats, and in one
repeat two different layers returned the identical wrong value, as if stale memory were being read. A standalone product of two plain
matrices was wrong above 65,536 rows and right at or below it.

**Why it mattered.** When the wrong value is finite, gradient clipping divides the whole update by it, so those steps do almost nothing
(norms of 30,000 against true norms of 30 to 700). When it is infinite, the clip produces NaN weights and the next forward pass fails.
Every graphics-chip run of this project trained through this, which likely explains part of the pass-to-pass swings seen earlier.

**What was done.** A workaround in the trainer: on the graphics chip, every linear layer runs in pieces of at most 65,536 rows during
training (gradients add up across pieces), which brings the error to 10^-7 at under 1 percent cost; inference is untouched. A finite loss
with a non-finite gradient now skips that one update and names the tiles, as mixed-precision training does. The crashed run was relaunched
once with the fix, and a clean rerun of the best recipe is queued so that the recipe is measured with correct gradients.

**Public record.** PyTorch's own tracker holds related silent-correctness reports for MPS matrix products: a batched product returning
wrong entries once a dimension reaches 32,769 ([issue 111634](https://github.com/pytorch/pytorch/issues/111634), marked high priority), a
follow-up saying earlier fixes were partial ([issue 122123](https://github.com/pytorch/pytorch/issues/122123)), and wrong results from a
linear layer with a non-contiguous weight ([issue 162730](https://github.com/pytorch/pytorch/issues/162730)). None names this exact case;
our evidence is the CPU comparison above.

**For the real project.** Train on hardware whose numerics are verified, and verify them: a one-batch gradient check against the CPU
before a long run is cheap, and the only reason this was found at all was that a crash forced the look.

## 18. Licences decide what can ship

**What happened.** The most popular detector family (Ultralytics YOLO) is AGPL, which would oblige a company building on it to publish its
own code or buy a licence; RT-DETR through Hugging Face is Apache-2.0, so it was chosen. The symbol artwork is CC BY 3.0 and is credited.
The OCR engine needs the non-headless OpenCV wheel, which bundles LGPL components. For the real test set, figures credited to engineering
firms or manufacturers inside federal documents were excluded even where the document said "non-copyrighted", because the figure's
author was not the government.

**For the real project.** Licence review is part of model selection and of data collection, from the first day.

## 19. Speed: the second OCR pass dominates the time per drawing

**What happened (18 Oct).** The trained reader takes about 10 seconds a drawing on this Mac's processor: 6 to 7 seconds of OCR, of which
most is the close-up pass around each of about 25 symbols, and 3 to 4 seconds of detection. The live demo runs on a free server with two
slower cores.

**What is planned.** Drop the close-up pass when the engine is RapidOCR (it hurts accuracy anyway) and keep the small detector; the
benchmark is rerun on the server itself.

## 20. The symbol-to-label link is the next error source

**What happened.** Today's rule takes the nearest text to the right or below a symbol. On the new drawings a way's label block has up to
four lines, sits below the bottom of a chain of symbols in one layout, and holds ratings for devices (contactor kW, capacitor kVAr) that
have no label of their own.

**What is planned (agreed 17 Oct).** The circuit owns the labels: symbols on one vertical line form a way, text lines stacked beside it
form the way's label block, the block goes to the way with the shortest edge-to-edge gap, and each line is routed to a device by what it
says. The generator records which device each text was drawn for, so the link gets its own precision and recall. The real project replaces
the rule with a learned link model trained on estimators' corrections.

## 21. Putting the demo on a server that sleeps

**What happened.** The free Hugging Face plan refused a Docker Space (payment required), so the site went to Google Cloud Run, which
charges nothing while nobody visits. Three things bit on the first deployment. The default sandbox ran the detector and the text reader
about ten times slower than a laptop core (95 seconds for one generated drawing); the second-generation runtime halves that. The server
loads its models in a background thread when it starts, and Cloud Run gives a container processor time only while it is answering a
request, so that thread was starved and took 171 seconds to be ready; with the processor kept on it is ready in 20 to 24 seconds, at the
price of being billed for the whole time the instance is awake (about 16 minutes per visit, which the free allowance covers for roughly
110 visits a month). The token the server uses to fetch the private model was the owner's write token for every repository; it should be
a read-only token for that one model. The service now runs under its own account that can reach nothing but that token.

**What a real deployment does differently.** A readiness probe (the health check answering "not yet" until the models are loaded) lets
the platform keep the processor on only during start-up and go back to per-request billing; a copy of the model baked into the image
removes the token altogether; a few requests at a time keep the page responsive while a run is in progress; and a budget alert on the
billing account catches a demo that something keeps awake around the clock.

**What happened next (19 Oct, 01:00 to 03:00).** The processor was first kept on so the models could load in a background thread,
which would have cost about $110 a month if something kept the service awake around the clock; that was reversed the same night: the
processor runs only during start-up and requests, the models load inside the first request instead of a thread, and live runs are
capped per day so the free allowance holds even in the worst case. The token the server used to fetch the model was the owner's write
token, kept in the cloud's secret store and reachable only by the service's own account, but far too powerful for a server that only
needed to read one file; the fix was to need no token at all: the model is copied into the container image and the secret was destroyed.

## 22. The confidence shown was the detector's raw score

**What happened.** The page painted almost every device red, "low confidence", on drawings where the detector had found every symbol.
The number shown was the detector's raw score, and the colour bands (high from 0.8, medium from 0.6) had been set for the old
shape-matching reader, whose scores behave like percentages. The trained detector's scores do not: it is trained to rank boxes, its
working threshold is 0.11, and on the validation drawings a box scoring 0.3 was right 99.9% of the time.

**What we did.** The number is now calibrated. Each model's weights carry a table, built from the same drawing-level sweep that chooses
its threshold, giving for each band of score how often such a box was right on the validation drawings (for the current model a score
from 0.11 to 0.15 reads 0.75, from 0.26 upward 0.996). The raw score is kept in the result for the record, the page says what the number
means and that it was calibrated on synthetic drawings, and a device's colour is now mostly decided by whether its rating label was
read, which is the reading an estimator needs. On real drawings the calibration overstates, because they are not the drawings it was
measured on; those are the stretch test and say so.
