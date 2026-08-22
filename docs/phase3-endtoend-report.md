# First end-to-end run

**2026-08-22. The pipeline runs. The output is worse than the input.**

Both halves of that sentence matter, and the second is not a surprise — it is what every component
measurement predicted.

Raw material: `artifacts/mosaicked_tos_002.mp4` (input), `artifacts/restored_tos_002.mp4` (output).

---

## 1. What ran

A worker process, driven over the real protocol — `hello` → `probe` → `process` → `shutdown` on
stdin, JSON Lines on stdout. No test harness shortcuts, no in-process calls.

```
ready            worker 0.1.0, protocol 1.0, cuda=True (RTX 3080 Ti), models=[det-unet 0.1.0]
probeResult      1920x800 h264 vfr=False 4.00s
result           status=completed
```

The input was built by applying a screen-anchored 10 px mosaic to a drifting ellipse over a clean
corpus clip, so there is a known ground truth to score against.

### The timeline invariant holds

```
framesSeen 96 · framesRestored 95 · framesPassedThrough 1
frames 96/96, max PTS error 0 ticks, monotonic=True
```

§5.1.7's rule — output frame count equals input frame count, every PTS preserved — survived a full
pass with restoration active. That was the point of building the media layer first.

### The router explained itself

```
SufficientTemporalEvidence      743
MotionOutsideOperatingWindow    347
ObjectAnchoredGrid              100
SingleValidFrame                  9
NoRegion                          1
```

Multi-frame restoration ran 743 times. Every decision carries a reason from the closed enum, which
is what made the diagnosis below possible at all.

## 2. What it produced

Scored against the **clean** original, inside the mosaicked region only:

| | mosaicked input | restored output | delta |
| --- | ---: | ---: | ---: |
| PSNR | 37.243 dB | 34.880 dB | **−2.364 dB** |
| SSIM | 0.9673 | 0.9586 | −0.0087 |

**Frames improved: 0 of 96.**

Outside the region — the part the pipeline is supposed to leave alone:

| | PSNR against clean |
| --- | ---: |
| mosaicked input | 49.55 dB |
| restored output | **39.49 dB** |

**A 10 dB loss on picture that was never mosaicked.**

## 3. Where the damage comes from

The null run (§5.1.8, `T-QUALITY-NULLRUN-01`) separates the encoder from the restoration by
re-encoding the input with restoration disabled:

| Encode | PSNR vs input | SSIM | Transparent? (≥42 dB / ≥0.99) |
| --- | ---: | ---: | --- |
| x265 fast, CRF 20 | 45.381 | 0.9864 | **no** — SSIM fails |
| x265 fast, CRF 16 | 47.088 | 0.9891 | **no** |
| x265 fast, CRF 12 | 48.842 | 0.9917 | **yes** |

So of the 10 dB lost outside the region, roughly half is the encoder at CRF 20 and the rest is the
pipeline touching pixels it should not have. Two causes, both already measured elsewhere:

**The detector is over-firing.** 843 regions across 96 frames — about nine per frame — where the
input contains exactly one. `docs/phase1-detector-report.md` measured 10.6% of clean crops marking
more than 0.5% of their area, and this is that number arriving in a video. Every false positive is a
patch of clean picture that gets "restored", and restoring clean content damages it.

**Blending bleeds, by design, into content that should not have been touched.** §5.11 dilates the
mask by `2 + ceil(block/4)` and feathers 3 px beyond that — around 8 px here. That allowance is
correct when the mask is correct. Around a false positive it is 8 px of extra damage.

**And the restoration itself is not helping inside the region either.** `tos_002` sits at the top of
the slow band (median 0.97 px/frame), which `docs/phase2-alignment-report.md` §3 identified as the
only regime where multi-frame gains at all — and even there it measured +0.36 dB with a *classical*
solver on *centred* windows. This run uses a causal window (§5, below) and a real detector's masks
rather than exact ones. Zero frames improved is consistent with that, not a contradiction of it.

## 4. Bugs the run found

Three, none of which any unit test would have caught, because each was a mismatch *between* stages.

**Full-frame restoration.** The first attempt ran dense flow and back-projection at 1920×800 for a
region covering a fraction of the frame. It used 10.9 GB of VRAM and had produced almost nothing
after eight minutes. §5.5's ROI stabilisation is not an optimisation — the pipeline is not viable
without it. `roi.py` and 16 tests now exist; the crop for this content is under a fifth of the frame.

**RAFT fails below 128 px.** "Feature maps are too small to be down-sampled by the correlation
pyramid." Measured: 96×96 fails, 128×128 works. Small ROIs are this pipeline's *common* case, so
every alignment on a small region failed and the router fell back to single-frame everywhere — while
the logs said the window was 3 and nothing looked broken. The aligner now pads up internally, which
is where the constraint belongs.

**`jobId` in the envelope, read from the payload.** `parse_request` lifts `jobId` out of the message
body, and `_start` then looked for it in the body. Every `process` was refused with E7006.

## 5. A deliberate deviation from §5.6

**The window is causal, not centred.** §5.6 centres the window on the target, which needs a
look-ahead of `K//2` frames — the writer trails the reader. This pipeline hands each frame to the
encoder as it is produced, so only past frames exist when a frame is restored.

Taking `K//2` *past* neighbours leaves **one** neighbour at the measured K=3, and the router's
two-neighbour minimum then makes multi-frame unreachable while every log line reports a window of 3.
That is a worse failure than the deviation: it is silent. So the window keeps its size and spends it
backwards — target plus two past frames.

The cost is a longer maximum baseline (two frames back rather than one each way), and
`docs/phase2-alignment-report.md` §3 measured that shorter baselines align better. Implementing the
look-ahead in the media layer is the correct fix.

## 6. What this changes

1. **The detector's false-positive rate is now the top blocker, not a Phase 1 nicety.** It was
   already the identified bottleneck; this run shows what it does to a real output. Nothing
   downstream can compensate for restoring content that was never damaged.
2. **The default encode is not transparent.** CRF 20 fails §5.1.8's SSIM bar; CRF 12 passes. The
   default has to come from this measurement rather than from a guess, and the pass-through path of
   R-1.8a (stream-copy when nothing is detected) matters more than it looked.
3. **Look-ahead belongs in the media layer**, so §5.6's centred window becomes available.
4. **An end-to-end quality check belongs in the test suite**, not in a scratch script. This one was
   run by hand.

## 7. Limitations

- **One clip, one synthetic mosaic, one setting.** Nothing here is a benchmark.
- **The mosaic is synthetic and screen-anchored**, with a shape the generator chose. Real mosaics
  have not been processed by this pipeline or any other part of this project.
- **PSNR and SSIM only.** No LPIPS, no temporal metric, and **nobody has looked at the output**.
  A −2.4 dB result can look better or worse than its number; that has not been checked.
- **The scoring script is not a test.** It lives in a scratch file and its region mask is derived by
  differencing, which is available here only because the degradation was synthetic.


---

## 8. The ladder — one variable per rung

`scripts/eval_endtoend.py` reruns the pipeline over a fixed input changing one thing at a time, so
each of §3's three causes gets a number instead of a share of the blame. All figures are against the
**clean** original; the input's own scores are the bar the pipeline has to beat.

### 8.1 Before the YUV fix

| rung | changed | inside | vs input | outside | vs input | regions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| (input) | the mosaicked video itself | 37.24 | — | 49.55 | — | — |
| baseline | detector v0.1.0, mask 0.5, CRF 20 | 34.88 | −2.36 | 39.49 | −10.06 | 843 |
| model | + detector v0.2.0 | 34.90 | −2.34 | 39.62 | −9.93 | 529 |
| threshold | + mask 0.9 (calibrated) | 35.04 | −2.21 | 40.01 | −9.55 | 194 |
| encode | + CRF 12 (measured transparent) | 35.09 | −2.15 | 40.44 | −9.12 | 194 |

**Regions fell 4.3x and the quality barely moved** — 0.21 dB inside, 0.94 dB outside. That is the
result that mattered: the detector work was real and it bought almost nothing, so the damage was
mostly coming from somewhere else entirely.

### 8.2 Finding it

Scoring the output against the **input** rather than against the clean original, restricted to
pixels the pipeline never restored, separates what the pipeline did from what it was given:

| | PSNR vs input, on never-restored pixels |
| --- | ---: |
| encode only, CRF 20 | 46.46 |
| encode only, CRF 12 | 49.72 |
| **pipeline, CRF 20** | **41.29** |
| **pipeline, CRF 12** | **42.64** |

Five to seven dB, on pixels nothing had touched, independent of both the detector and the encoder.

The cause, measured directly with no processing at all in between:

| one frame round trip | luma PSNR |
| --- | ---: |
| via `rgb24` — what the pipeline did | **45.33 dB** |
| via `yuv420p` planes | **inf** (lossless) |

Decoding each frame to RGB and re-encoding from RGB destroys and re-creates 4:2:0 chroma and rounds
twice through the colour matrix. It cost more than the encoder did, and it applied to **the whole
frame**, including everything the pipeline never went near.

### 8.3 After the fix

The pipeline now works on the **luma plane** and leaves chroma exactly as decoded. Chroma is not
rewritten to chase the restored luma: a mosaic destroys luma detail, and the pipeline has no
evidence about colour that would justify altering it. `astype(uint8)` was also replaced with
rounding — truncation biases every pixel in the frame down by half a level.

| rung | changed | inside | vs input | outside | vs input | regions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| (input) | the mosaicked video itself | 37.24 | — | 49.55 | — | — |
| baseline | detector v0.1.0, mask 0.5, CRF 20 | 36.30 | **−0.94** | 43.51 | **−6.05** | 823 |
| model | + detector v0.2.0 | 36.30 | −0.94 | 43.50 | −6.05 | 521 |
| threshold | + mask 0.9 (calibrated) | 36.45 | −0.80 | 43.94 | −5.62 | 177 |
| encode | + CRF 12 (measured transparent) | 36.55 | **−0.69** | 44.98 | **−4.57** | 177 |

Inside the region: −2.15 dB → **−0.69 dB**. Outside: −9.12 dB → **−4.57 dB**.

**The output is still worse than the input.** It is now worse by an amount that could plausibly be
closed rather than by an amount that says the approach is wrong.

### 8.4 What the ladder says about each cause

| Cause | Contribution |
| --- | --- |
| **RGB round trip** | **The largest single item.** 1.5 dB inside, 4.5 dB outside, and it was not on the list of three suspects at all |
| Encode not transparent | ~1.0 dB outside, ~0.1 inside. Real, and the cheapest thing on the list |
| Detector over-firing | ~0.15 dB inside, ~0.45 dB outside — from a **4.6x** reduction in regions |
| Restoration itself | Whatever remains: −0.69 dB inside with 177 regions and a transparent encode |

The detector line is the surprise. It was the identified top blocker after the first run, the
retraining was justified by measurement, the retraining worked — and it moved the output by a
fraction of a dB. **The blocker was correctly identified as a defect and incorrectly identified as
the cause of the damage.** Separating those two is what the ladder is for.

## 9. Limitations, updated

Everything in §7 still holds, and:

- **Still one clip.** The ladder is four runs over one 96-frame input.
- **The detector change is now confounded.** Switching to the luma plane changed what the detector
  sees — the Y plane rather than a matrix-weighted RGB approximation — so the 823 vs 843 region
  counts are not a controlled comparison across §8.1 and §8.2. Within each table the comparison is
  controlled.
- **Nobody has looked at any output.** −0.69 dB may look better or worse than the input; that has
  not been checked, and PSNR on a restoration is a weak proxy for it.
