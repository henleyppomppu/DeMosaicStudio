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
