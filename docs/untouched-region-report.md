# Where the untouched-region loss comes from

**2026-08-22.** The ladder left the output **4.57 dB below its input on picture that was never
mosaicked** (`docs/phase3-endtoend-report.md` §8.3). The obvious reading was that this is the cost
of re-encoding every frame, and that R-1.8c's smart-cut is therefore the fix.

That reading is 64% right and its recommended fix has a ceiling of **zero** on the content we have.

Raw material: `scripts/eval_untouched.py`, `docs/untouched-decomposition.json`.

---

## 1. The split

A null run separates the two spenders: same media layer, same encoder, same settings, restoration
disabled. Whatever the null run loses is the encoder; whatever the pipeline loses beyond it is the
pipeline. All figures are luma PSNR on pixels the mosaic never touched.

| arm | what | vs clean | vs input |
| --- | --- | ---: | ---: |
| input | the mosaicked video itself | 49.55 | — |
| null run | re-encode only, restoration disabled | 46.62 | 49.72 |
| pipeline | re-encode + restoration | 44.98 | 46.94 |

| | dB | share |
| --- | ---: | ---: |
| **encoder** (input → null run) | 2.94 | 64% |
| **pipeline** (null run → pipeline) | 1.63 | 36% |
| total | 4.57 | |

**The encoder is already transparent by this product's own definition.** R-1.8b sets the bar at
≥42 dB against the source, and the null run scores 49.72 dB. The 2.94 dB is what "transparent"
costs when a third reference is introduced; buying it back means going beyond the threshold the
product itself chose, with the bitrate that implies.

The 1.63 dB is not covered by any threshold. It is waste.

## 2. Smart-cut cannot help here

Smart-cut re-encodes any GOP containing a restored frame and stream-copies the rest. So its ceiling
is set by where the detections are, not by how well it is implemented.

| | |
| --- | ---: |
| frames firing | **96 / 96** |
| frames copyable | 0 |
| GOPs firing | **2 / 2** (keyframes at 0, 72) |
| frames in copyable GOPs | 0 |

Every GOP contains a detection, so **smart-cut saves nothing on this file**. It remains the right
answer for content where the mosaic covers part of the runtime — but this clip cannot measure that,
and neither can this ladder. What decides its value in practice is the detector's false-positive
rate, which is the same thing that decides everything else here.

## 3. What the pipeline's 1.63 dB actually is

| | |
| --- | ---: |
| untouched picture | 61.7% of the frame |
| of which the pipeline altered | **9.47%** (5.84% of the whole frame) |
| on those pixels | 42.24 dB → **38.03 dB** (−4.21 dB) |

The dilation-and-feather halo cannot account for that area. §5.11 dilates by `2 + ceil(block/4)`
and feathers 3 px — about 8 px here — and 8 px around one drifting ellipse is roughly **0.43%** of
the frame. The measurement is **13× larger**.

So it is not bleed. It is **false positives**: regions the detector invented in clean picture, each
one restored, each one damaged by roughly the same amount the real region is. 177 regions were
found across 96 frames where the truth is one per frame.

**The restoration currently makes things worse everywhere it touches** — −0.69 dB inside the
mosaicked region, −4.21 dB on the clean picture it should not have touched at all.

## 4. What was fixed off the back of this

Three defects, all in the same place, all of the family this project keeps finding: **the summary
and the bytes disagreed, and no test compared them.** The job runner had no tests at all.

1. **R-1.8a was reported, not implemented.** `passthrough: regionsDetected == 0` was a claim about
   the decision; the video was fully re-encoded either way. It is now a stream copy, and
   `T-IO-PASSTHROUGH-COPY-01` asserts a byte-identical video stream. Verified end to end: a clean
   clip through the CLI came back at 8,240,353 bytes with an identical packet hash.
2. **`analyze` ran the whole restoration** and discarded the pixels — 162 s against 153 s for the
   real job — and wrote a video next to the source when given no output path. It is now detection
   and tracking only, writes nothing, and honours `sampleEvery`: **162 s → 35 s**, and 11 s at
   `--sample-every 4`, with identical detections.
3. **PyAV cannot remux video at all.** `add_stream_from_template` builds an encoder-backed stream
   and muxing demuxed packets through it writes one byte per packet — 23 KB of video comes out as
   21 bytes and does not decode. Audio survives the same call, which is why the audio pass-through
   has always been correct. D-22.

## 5. What this changes

1. **Smart-cut is not the next thing to build.** It is deferred by D-07 and this measurement gives
   no reason to promote it: on content the detector fires on continuously, its ceiling is zero.
2. **The remaining outside loss is the detector, again** — but now with a number attached to the
   right cause. 5.84% of the frame altered, at −4.21 dB.
3. **The pipeline should not write pixels it cannot improve.** `minRestorationConfidence` exists
   and defaults to 0.0, which is off. Nothing currently stops a restoration that makes a region
   worse from being blended in. That is a gate, not a model, and it is cheap to measure.

## 6. Limitations

- **One clip, one synthetic mosaic.** Same limitation as every other number in this project.
- **The "untouched" mask is derived by differencing** clean against degraded, which is only
  available because the degradation was synthetic.
- **The 2.94 dB encoder cost is specific to x265 `fast` at CRF 12.** A slower preset would narrow
  it; that has not been measured.
- **Nobody has looked at any of these outputs.** Still true, and still the cheapest item on the
  list.
