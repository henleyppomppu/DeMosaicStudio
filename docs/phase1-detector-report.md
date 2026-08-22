# Experiment A — Can the detector learn the mosaic signature?

**Run 2026-08-22. Answer: yes, and the corpus is not the current limit. Negatives are.**

Raw data: `docs/phase1-detector.json`. Script: `training/train_detector.py`.

---

## 1. Why this experiment exists

The dataset plan called for tens of hours of clean video before Phase 1 could start. **That figure
was never measured** — it was written into `prd.md` v2.0 as a plausible-sounding number, which is
exactly what §12.7 forbids elsewhere in the same document. It is withdrawn in v3.2.

The replacement is to measure. This experiment asks the cheapest useful question first:

> Does the D-03 architecture learn to segment a synthetic mosaic *from the corpus already in hand*?

Both answers are actionable. A failure means the architecture is wrong and no quantity of video
fixes it. A success means the corpus is not what binds, and the next thing to buy is something else.

## 2. Setup

| | |
| --- | --- |
| Corpus | `clean-tos`, 24 clips (96 s) from Tears of Steel, CC BY 3.0 |
| Split | **by clip**, stratified by motion band — 18 train / 6 validation (§11.6) |
| Validation clips | `tos_000`, `tos_001`, `tos_002`, `tos_005`, `tos_009`, `tos_010` — one per band or better, never seen in training |
| Model | `MosaicUNet`, width 32, full-resolution decoder, 7.76 M parameters |
| Positives | synthetic: block 4–24 px, non-square allowed, random phase, ellipse or rectangle regions, pixelation or box blur, opacity 0.85–1.0 |
| Hard negatives | **manufactured, not imitated** — real JPEG blocking at quality 3–12, real downscale/upscale resampling, heavy grain (§11.4) |
| Recompression | every sample JPEG round-trips at quality 55–95 (§11.3) |
| Training | 8000 steps, batch 12, 256×256 crops, AdamW + OneCycle, Dice+BCE |
| Hardware | RTX 3080 Ti, 16 min |
| Runs | **two, identical configuration** — see §3.1 |

Evaluation runs **two separate passes** rather than one mixed pass: IoU on mosaicked crops, and
false-positive area on clean ones. An aggregate over both would let a good IoU hide a bad
false-positive rate, and §5.2.5 says the second is the one that damages a user's footage.

## 3. Result

| Metric | Train | **Validation (held-out clips)** |
| --- | ---: | ---: |
| Mask IoU, mean | 0.869 | **0.820** |
| Mask IoU, worst tenth (p10) | — | 0.495 |
| False-positive area on clean crops, mean | — | 0.0043 |
| False-positive area, p95 | — | 0.0304 |
| **Clean crops marking >0.5% of area** | — | **10.6%** |

Numbers above are run 2 (`docs/phase1-detector.json`); run 1 is preserved at
`docs/phase1-detector-run1.json`. Learning curve (validation IoU): 0.003 → 0.415 → 0.693 → 0.767 →
0.810 → **0.834** at 8000 steps, still creeping up.

### 3.1 Training noise floor — measured, not assumed

§13.5 requires knowing how much a metric moves when *nothing* changes, before any change is credited
with moving it. Two runs at identical configuration and identical seed:

| Metric | Run 1 | Run 2 | Δ |
| --- | ---: | ---: | ---: |
| val IoU mean | 0.8150 | 0.8201 | **0.0052** |
| val IoU p10 | 0.5161 | 0.4954 | **0.0206** |
| val FP area mean | 0.0046 | 0.0043 | 0.0003 |
| val clean crops >0.5% | 11.72% | 10.55% | **1.17 pp** |
| train IoU mean | 0.8666 | 0.8688 | 0.0022 |

The seed is fixed, so this is GPU nondeterminism — cuDNN and cuBLAS reductions are not bit-exact by
default. **Any future claim that a change improved validation IoU must exceed ~0.005, and any claim
about the worst-tenth or the false-positive rate must exceed ~0.02 and ~1.2 pp respectively.** Those
are large tolerances relative to the improvements a tuning change typically produces, which is
precisely why they are worth knowing before the tuning starts.

### 3.2 IoU by block size

| Block | Validation IoU | n |
| --- | ---: | ---: |
| 4–6 px | **0.748** | 52 |
| 7–12 px | 0.834 | 121 |
| 13–18 px | 0.846 | 103 |
| 19–24 px | **0.878** | 108 |

**Detection gets monotonically easier as blocks get larger** — and that is the exact inverse of
restoration. §1.4.2 and the Phase 0 gate say large blocks destroy the information that makes
restoration possible; this says large blocks are the ones a detector finds most obvious.

So the two halves of the product are hardest in opposite places:

| Block | Detect | Restore |
| --- | --- | --- |
| small (4–6) | **hard** — subtle, looks like ordinary softness | easy — little was destroyed |
| large (19–24) | easy — unmistakable | **hard** — nothing left to recover |

The small-block band is also where false positives are most likely, for the same reason: a 4 px
mosaic and mild compression softness are nearly the same signal. That makes 4–6 px the band where
detector work and negatives work are the *same* work, and it is where both should be aimed.

## 4. What it means

### 4.1 The architecture works. D-03 stands.

0.820 IoU on clips the model never saw, from a 7.8 M-parameter network with no pretrained encoder,
trained for sixteen minutes on 96 seconds of video. The train/validation gap is 0.05 — it is
learning the *signature*, not memorising the shots.

That was the point of running this before buying data. **The corpus is not what limits the detector
today.**

### 4.2 False positives are the binding constraint, as predicted

**10.6% of clean crops mark more than 0.5% of their area** (11.7% in run 1 — the spread is the
noise floor of §3.1, not a difference). §5.2.5a asks for at most 0.5% of negative *frames* to
produce any region at all.

The two are not the same measure — crops are not frames, and these hard negatives are manufactured
rather than collected — so the gap is not directly 23×. But the direction is unambiguous: the IoU
side is already respectable and the false-positive side is nowhere near requirement. **More clean
video does not improve this number.** More and better *negatives* do.

This is the answer to "is a corpus necessary?": not a large clean one, not yet. A negatives corpus,
yes — and §11.4 in v3.2 now records that roughly half of it can be manufactured authentically (real
encoder blocking, real resampling softness, real grain) while defocus, bokeh and genuinely-on-a-grid
content (pixel art, LED walls, mesh fabric) must be collected because a synthetic approximation
would teach the detector precisely the wrong boundary.

### 4.3 The worst tenth is still poor

p10 IoU of ~0.50 means one crop in ten is substantially wrong, and §3.2 says where: the 4–6 px band
scores 0.748 against 0.878 at 19–24 px. The failures are **small blocks**, not large ones.

That is the useful shape of the answer. It is not a capacity problem — the model handles the harder-
to-segment large regions fine. It is a *discriminability* problem at the boundary between a faint
mosaic and ordinary compression softness, and the fix is better negatives in that band, not a bigger
network.

## 5. Limitations — read before quoting any number above

- **One film.** Every clip is from Tears of Steel. "Held-out clip" means a shot the model has not
  seen; it does **not** mean different grading, different grain, different camera, or different
  subject matter. Cross-content generalisation is untested and is the first thing a second CC source
  would measure.
- **JPEG stands in for H.264.** Training recompresses per crop with JPEG because it is fast and
  per-crop. It is a real DCT-quantisation artifact, not an imitation — but it is not the artifact the
  product will face. The Phase 0 gate showed H.264 CRF matters (23% of the recoverable gain between
  CRF 18 and 26), so the real training set needs a pre-encoded H.264 ladder.
- **All positives are synthetic.** That is what makes the mask exact and the experiment possible, and
  it is also the domain gap (§18 R-03). No real mosaicked footage has been tested against this model.
- **Hard negatives are manufactured.** The categories that must be collected — defocus, bokeh, real
  grid content — are absent, and they are the ones most likely to fool a mosaic detector.
- **`fp_crops_over_half_percent` is not §5.2.5a.** It is a crop-level proxy chosen because it is
  measurable today. The real requirement is frame-level and needs the collected corpus.
- **Detection threshold is fixed at 0.5.** No sweep, no operating-point selection. §5.2.3's default
  of 0.45 has not been calibrated against anything.
- **Two runs is a thin noise floor.** §3.1's tolerances come from a sample of two. They are enough to
  reject a 0.002 "improvement" and not enough to characterise the distribution.
- **The checkpoint (`docs/phase1-detector.pt`) is an experiment artifact, not a model release.** It
  has no `metadata.json`, no version, and no entry in the model store (§14.1). It exists so §3.2 can
  be re-analysed without retraining.

## 6. What to do next

1. **Build the negatives corpus, aimed at the 4–6 px band.** Manufacture what can be manufactured
   (§11.4 v3.2); collect defocus/bokeh and genuine grid content. This is the binding constraint, and
   §3.2 says exactly where to point it: the band where a faint mosaic and mild compression softness
   are nearly the same signal.
2. **Add a second CC source** to measure cross-content generalisation. Cheap, and the current number
   cannot see this axis at all.
3. **Replace JPEG with a pre-encoded H.264 CRF ladder** in the training data.
4. **Sweep the detection threshold** and pick an operating point from the precision/recall curve
   rather than inheriting 0.45 from the PRD.
5. Only then consider more clean hours, and only if measurement says so.
