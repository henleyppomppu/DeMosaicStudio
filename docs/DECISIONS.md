# Architecture Decision Records

Seeded from `prd.md` §0.3. **Check here before reverting anything.**

Format: what was decided, what was rejected, why, and what it costs to undo. A decision that is
cheap to undo should not be re-litigated; one that is expensive should be argued about now.

---

## D-01 — Engine runs out-of-process in Python, native C++ deferred

**Decided.** Host ↔ engine over stdio JSON Lines (`prd.md` §8). A native C++ engine is deferred and
would speak the same protocol.

**Rejected:** C++20 core + C ABI + P/Invoke from day one (PRD v1.0).

**Why:** Phases 0–2 are research, and that research is in PyTorch whichever engine ships. The three
hardest requirements — the feasibility gate (§1.4), the detector domain gap, temporal consistency —
are all model problems that a native engine makes slower to iterate on, not easier. The dev machine
has no native toolchain at all (§4.5).

**Reversal cost: Low.** The boundary is a process boundary plus a wire protocol, not a language
binding. Guarded by `IEngineLauncher` (`T-HOST-LAUNCHER-SWAP-01`).

---

## D-02 — .NET 10 for the WPF host

**Decided.** `net10.0` / `net10.0-windows`, SDK pinned in `global.json`.

**Why:** SDK 10.0.302 is the only SDK on the dev machine; PRD v1.0's .NET 8 could not have built here.

**Reversal cost: Trivial.**

---

## D-03 — Detector is an in-house U-Net over a `timm` encoder

**Decided.** Binary semantic segmentation, U-Net decoder, Apache-2.0 encoder, trained in-house.

**Rejected:** Ultralytics YOLOv8/v11-seg. Legally available under D-11 and still rejected on
engineering grounds: no pretrained mosaic weights exist either way, so the gain is a mature training
loop rather than accuracy — and prototype-coefficient masks are *coarser* at the boundary than a
full-resolution decoder, which is the opposite of what mask-aware blending (§5.11) needs.

**Reversal cost: Medium.** An encoder/decoder swap is a training run.

---

## D-04 — In-house restoration model, optionally initialized from third-party VSR weights

**Decided.** Train on the degradation this product actually faces. Third-party pretrained weights may
initialize the model **only if Phase 2 measures a benefit** beyond the noise floor (§13.5).

**Rejected:** (a) using third-party weights as-is — trained to invert bicubic downsampling, not
pixelation, so they produce confidently wrong texture; (b) mandating from-scratch training when a
measured head start is available.

**Reversal cost: High.** This is most of Phase 2. Initialization itself is cheap to abandon; the
inherited license is not (§18 R-05). Contained behind `IRestorationBackend` (R-4.2a).

---

## D-05 — Training data is synthetic degradation over licensed clean video

**Decided.** The generator (`training/degradation/`) makes paired data with perfect masks.

**Reversal cost: Low.** The generator is needed regardless.

---

## D-06 — Checkpoint state is JSON with per-artifact fingerprints

**Decided.** One `job.json` per job directory, written atomically, with separate
`detection`/`restoration`/`encode` fingerprints (§9.3).

**Rejected:** one monolithic fingerprint (forces a full restart on any change); no fingerprint (lets a
resume mix artifacts produced under different settings — data corruption, not a UX bug).

**Reversal cost: Low.**

---

## D-07 — Full re-encode by default, stream-copy when nothing is detected

**Decided.** Smart-cut segment splicing deferred.

**Reversal cost: Low.**

---

## D-08 — FFmpeg via PyAV for MVP decode/encode

**Decided.** Hardware decode where available; full frames on CPU.

**Reversal cost: Medium.** It is Phase 5's main work item.

---

## D-09 — PyTorch + CUDA permanently; ONNX/TensorRT export dropped

**Decided.** No export toolchain, no per-backend numerical-parity gate, no deformable-conv TensorRT
plugin.

**Why:** Under D-11 there is no second machine to deploy to, so all of that is work with no
beneficiary. It also removes the project's single most likely integration failure.

**Reversal cost: Low.** Re-adding export is a Phase 5 item; §13.6's tolerances are retained, marked
inactive.

---

## D-10 — Protocol, error codes and job schema are versioned and change-controlled

**Decided.** Bump → change both sides → update docs → add a round-trip test. `PROTOCOL_VERSION` has
exactly one definition (`worker/demosaic_worker/protocol.py`) and one mirror, locked by
`ProtocolVersionTests`.

---

## D-11 — Personal use only; not distributed

**Decided.** No binaries, source, models or derived weights to anyone. AGPL and non-commercial
obligations therefore do not trigger; installer/packaging/portability work leaves scope.

**Reversal cost: High and asymmetric,** and it grows with every phase. See `prd.md` §2.4 for the
reversal trigger. **If distribution is ever discussed, stop and re-plan before writing more code.**

---

## D-12 — GPL FFmpeg with x264/x265; x265 is the default quality encoder

**Decided.** Two encoder profiles: Quality (x265 slow) and Speed (NVENC).

**Why:** NVENC HEVC is noticeably worse than x265 at equal bitrate, and §5.1.8's problem — a full
re-encode degrading the 85%+ of the picture that was never mosaicked — *is* an encoder-quality
problem.

**Reversal cost: Trivial** (a build flag), but see D-11: a GPL FFmpeg build makes the application GPL
if it is ever distributed.

---

## D-13 — Quality preset caps the temporal window; motion chooses within it

**Decided.** `PresetMaxWindow`: Fast 3, Balanced 7, Quality 9. The motion policy (§5.6) then selects
within that ceiling.

**Why:** `prd.md` §5.6's motion table and §15's per-preset table appear to conflict — §15 says
"Balanced K=5" while §5.6 says "low motion 7–9". Reading the preset as a *ceiling* satisfies both:
Balanced lands on 5 at medium motion and 7 at low motion. Recorded here because it is an
interpretation, not something either table states outright.

**Reversal cost: Trivial.** One function, covered by `TemporalWindowPolicyTests`.

**Date:** 2026-08-22. Implemented in `TemporalWindowPolicy.PresetMaxWindow`.

---

## D-14 — The Phase 0 gate measures two multi-frame arms, not one

**Decided.** `scripts/eval_multiframe_gate.py` reconstructs three ways per configuration:

| Arm | Neighbours | Answers |
| --- | --- | --- |
| `single` | none (K=1) | the floor |
| `multi_oracle` | synthesised from the target at known shifts | **is the information recoverable?** |
| `multi_estimated` | the actual neighbouring frames, aligned by global translation | can our alignment reach it? |

**Rejected:** a single multi-frame arm, which is what `prd.md` §1.4.3 originally implied.

**Why.** A single arm cannot distinguish *"the information is not in the neighbouring frames"* from
*"we could not align well enough to use it"*, and those two lead to opposite decisions — the first
kills the product, the second makes alignment the top priority. Measured on real footage the two
diverge enormously:

```
tos_009, block 8, SCREEN, CRF 18
  single           30.91 dB
  multi, oracle    34.82 dB    +3.91   <- the information is there
  multi, estimated 18.35 dB   -12.56   <- global translation cannot reach it
```

A one-arm gate would have reported KILL and ended the project on an alignment problem.

**Consequence.** The verdict vocabulary gains `PASS_ALIGNMENT_BLOCKED`: information recoverable,
current alignment insufficient. That is a *pass* for §1.4's premise and a re-prioritisation for
Phase 2, not a stop.

**Reversal cost: Low.** One extra arm in one script.

**Date:** 2026-08-22.

---

## D-15 — The corpus is committed as provenance, not as bytes

**Decided.** `training/datasets/clean/*.mp4` is git-ignored. What is committed is
`clean-tos.manifest.json` (per-clip SHA-256, motion band, source timestamp) and `SOURCES.md` (URL,
archive SHA-256, exact build command).

**Why.** 165 MB of clips would bloat the repository for no benefit: the corpus is reproducible from
a CC-BY download plus one deterministic command, and the manifest is what actually makes a result
traceable. A hash in a manifest answers "was this the same data?"; a checked-in video does not
answer it any better and costs every clone.

**Reversal cost: Trivial.**

**Date:** 2026-08-22.

---

## D-16 — Multi-frame runs only in a measured operating window, with K=3

**Decided.** `prd.md` §5.6's window table is replaced by measurement
(`docs/phase2-alignment-report.md`):

| Motion | K |
| --- | --- |
| static (< 0.25 px/frame) | **1 — multi-frame off** |
| slow (0.25–1) | **3** |
| medium (1–6) | **3**, only with high alignment confidence |
| fast (> 6) | **1 — multi-frame off** |

And §5.8's router gates path A on motion band rather than merely feeding it to the window policy.

**Rejected:** the previous table — low motion → K of 7–9 — which was written before any experiment.

**Why.** With dense flow and per-pixel confidence, only the slow band gains (+0.39 dB); static loses
0.29 dB *despite the best alignment of any band* because there is no phase diversity to exploit; fast
loses 1.48 dB because correspondence is gone. K=3 matched or beat K=5 everywhere, so the extra frames
were contributing content that no longer corresponds to the target.

Two independent measurements agree on the slow band: the Phase 0 gate's `estimated` arm (+0.32 dB)
and this experiment (+0.36 dB).

**Scope.** Measured with a **classical** solver. A learned restorer may tolerate imperfect
correspondence — that is Phase 2's open question, and if the answer is yes, this table is re-measured
rather than assumed to hold.

**Reversal cost: Low.** One table and one router condition, both covered by tests.

**Date:** 2026-08-22.

---

## D-17 — Dense flow is adopted, and flow work stops here

**Decided.** RAFT-small with forward-backward per-pixel confidence replaces global translation as the
alignment model (`training/restore/flow.py`). **No further investment in the flow estimator** until
something else changes.

**Why.** Dense flow cut the post-alignment residual by 69% (12.54 → 3.91 grey levels) and moved the
reconstruction gain by 1.3 dB out of a 5.6 dB gap. The remaining 4.3 dB is not misalignment: it is
that a frame 2/24 s away genuinely contains different content. A larger flow network buys a smaller
residual against a term that is no longer the binding one.

Per-pixel confidence is worth **+2.3 dB** over unweighted dense flow on the diagnostic clip, which is
why §5.9.4 now requires it per pixel rather than per frame.

**Reversal cost: Low.** `DenseAligner` is one class behind one interface.

**Date:** 2026-08-22.

---

## D-18 — Scene-cut thresholds are calibrated, and the first structure signal was wrong

**Decided.** Cut detection uses two signals with thresholds measured on the corpus by
`scripts/calibrate_scene_cuts.py`:

| Signal | Threshold | Separation measured |
| --- | ---: | --- |
| histogram distance | **0.09** | continuations p95 0.047 · cuts p05 0.135 |
| structure = `1 - |NCC|` | **0.515** | flashes p95 0.337 · cuts p05 0.693 |

**Rejected:** the original structure measure — mean absolute difference of peak-normalised gradient
maps — and the original guessed thresholds (0.45 / 0.30).

**Why.** The calibration showed the gradient measure did not separate the populations *at all*:
within-shot p95 was 0.058 against across-shot p05 of 0.042, fully overlapping. It would have made
the second signal pure noise, and since the second signal is the only thing distinguishing a flash
from a cut, every flash would have reset temporal context.

Normalised cross-correlation is invariant to the affine luminance change a flash applies, which is
precisely the discrimination required. After the swap both signals separate cleanly.

**Method note.** The two thresholds are calibrated against *different* populations, because they are
confusable with different things: the histogram threshold must sit above ordinary continuations, and
the structure threshold must sit above flashes. Using one population for both is what produced the
first, unusable number.

**Reversal cost: Low.** Two constants and one function, with the calibration script checked in so
the numbers can be re-derived rather than re-guessed.

**Date:** 2026-08-22.

---

## D-19 — Restoration is ROI-scoped, and the window is causal until look-ahead exists

Two decisions from the first end-to-end run (`docs/phase3-endtoend-report.md`).

### ROI scoping is mandatory, not an optimisation

**Decided.** Every restoration stage — flow, back-projection, blending — works inside a padded ROI
(`roi.py`, §5.5), never on the full frame.

**Why.** The first attempt ran dense flow and IBP at 1920x800 for a region covering a fraction of
the frame. It consumed 10.9 GB of VRAM and produced almost nothing in eight minutes. §5.5 reads like
a quality requirement about padding and reflection; it is also the difference between a pipeline
that finishes and one that does not.

**Reversal cost: n/a.** There is no version of this worth reversing.

### The window is causal, not centred

**Decided.** The temporal window is the target plus `K-1` **past** frames, not `K//2` on each side.

**Rejected:** taking `K//2` past neighbours, which leaves *one* neighbour at the measured K=3. The
router's two-neighbour minimum would then make multi-frame unreachable while every log line reported
a window of 3 — a silent failure, and worse than a documented deviation.

**Why not centred.** §5.6 centres the window, which requires the writer to trail the reader by
`K//2` frames. This pipeline encodes each frame as it is produced, so future frames do not exist yet.

**Cost, measured elsewhere:** a longer maximum baseline (two frames back rather than one each way),
and `docs/phase2-alignment-report.md` §3 found shorter baselines align better. So this deviation
costs quality in the one band where multi-frame works at all.

**Reversal cost: Medium.** Implementing look-ahead means buffering frames in the media layer between
the transform and the muxer. It is the correct fix and is recorded as a next step rather than done.

**Date:** 2026-08-22.

---

## D-20 — The detector's operating point is measured on video, and the requirement is frame-level

**Decided.** Two things, both prompted by the first end-to-end run finding 843 regions in a clip
containing one.

**The threshold is calibrated, not chosen.** `scripts/calibrate_detector.py` sweeps it against video
with a known ground-truth mask and reports precision, recall, regions per frame, and the rate at
which clean frames fire. §5.2.3's default of 0.45 had never been calibrated against anything.

**The training metric is `clean_frames_firing`, not a crop-area proxy.** §5.2.5a is a frame-level
requirement — at most 0.5% of negative frames may produce any region — and the crop-level proxy used
for v0.1.0 read a comfortable 10.6% while the real quantity was 37x over the bar. A proxy that
comfortable is worse than no proxy, because it stops the search.

**Consequence.** The sweep showed **no threshold** could satisfy §5.2.5a on v0.1.0, which is what
turned "tune the operating point" into "retrain the model" (§8 of the detector report). Widening the
negatives then halved the video-level rate, 18.8% to 9.4%, at a cost of 0.032 val IoU — outside the
0.005 noise floor, so a real cost knowingly accepted.

**Note.** `clean_frames_firing` is trivially satisfied by a model that predicts nothing; a 300-step
smoke run scored 0.39% against the 0.5% bar while its IoU was zero. It is only meaningful read
alongside IoU, and the evaluation reports both for that reason.

**Reversal cost: Low.** A metric and a script.

**Date:** 2026-08-22.

---

## D-21 — The pipeline works on luma planes, not RGB

**Decided.** Frames are handled as `yuv420p` planes. Restoration writes the luma plane; chroma is
passed through exactly as decoded.

**Rejected:** converting each frame to `rgb24`, restoring, and converting back — which is what the
first implementation did because RGB is the convenient array shape.

**Why.** Measured with no processing in between: one `yuv420p -> rgb24 -> yuv420p` round trip costs
**45.33 dB of luma**, while the plane round trip is lossless. That is worse than the encoder
(46.46 dB at CRF 20) and it applies to the **whole frame**, including every pixel the pipeline never
touched — the exact thing §5.1.8 exists to prevent.

On the end-to-end ladder it was worth **1.5 dB inside the region and 4.5 dB outside**, which made it
the largest single source of damage. It was not among the three causes named after the first run.

**Chroma is not rewritten to follow the restored luma.** The first implementation scaled RGB by the
luma ratio, which is a plausible-looking way to invent colour detail the pipeline has no evidence
for. A mosaic destroys luma detail; leaving chroma alone is the honest response.

**Also fixed here:** `astype(np.uint8)` truncates. Every pixel in every restored frame was biased
down by half a level. Now `np.rint` then cast.

**Reversal cost: Low.** One function's input and output handling.

**Date:** 2026-08-22.

---

## D-22 — A stream copy shells out to ffmpeg, because PyAV cannot remux video

**Status:** accepted (2026-08-22) · **Closes:** R-1.8a, `T-IO-PASSTHROUGH-COPY-01`

### Context

R-1.8a says a job that restores nothing must stream-copy rather than re-encode. Until now the
summary reported `passthrough: regions_detected == 0` and the video was fully re-encoded either
way — a claim about the *decision*, not about the bytes. There were no tests on the job runner at
all, so nothing noticed.

The cost of getting this wrong is measured, not assumed: re-encoding at the operating point the
project itself calls transparent still costs **2.94 dB across the whole frame**
(`docs/untouched-decomposition.json`). A file the pipeline had no reason to touch came back
measurably softer for nothing.

### The obstacle

The natural implementation — PyAV's `add_stream_from_template` plus muxing the demuxed packets —
**silently produces a broken file**. Measured on PyAV 14.0.1:

| | video bytes in | video bytes out |
| --- | ---: | ---: |
| MP4 → MP4 | 23,062 | **21** |
| MP4 → Matroska | 23,062 | **0** |
| audio (Matroska → Matroska) | 1,324 | 1,324 |

One byte per packet, and the result does not decode (`InvalidDataError`). The output stream that
`add_stream_from_template` builds for video is **encoder**-backed (`libx264`, `is_encoder=1`), and
muxing raw packets through it does not write them. Copying the extradata across and constructing the
stream by codec name both fail identically.

Audio survives the same call, which is why the audio pass-through in `run_passthrough` has always
been correct and why this could not simply be written the same way.

### Decision

`run_stream_copy` invokes `ffmpeg -c copy` and then reads the timeline back off **both** files
rather than assuming a copy preserved it.

`tools/ffmpeg` is gitignored — it is part of the machine, not the repository — so a fresh checkout
may have no ffmpeg. In that case the copy raises `StreamCopyUnavailable`, the runner keeps the
re-encoded output, and it emits **W5102**. It never reports a pass-through it did not perform;
reporting one is the defect this replaced.

### Consequences

- The pass-through path depends on an external binary and is skipped in CI. The skip says so.
- `test_pyav_still_cannot_remux_video_on_its_own` pins the reason for the shell-out. If a future
  PyAV fixes remuxing, that test fails and the decision can be revisited — which is its purpose.
- **How often this fires is set by the detector, not by this decision.** On a clean corpus clip the
  detector found 48 regions across 96 frames and the file was re-encoded; on another it found none
  and came back bit-identical. Pass-through is worth exactly as much as the false-positive rate
  allows.

---

## D-23 — `analyze` is detection and tracking only, and writes nothing

**Status:** accepted (2026-08-22) · **Closes:** §8.3, §5.2.5c

### Context

The protocol has always defined `analyze` as "detection and tracking only". The implementation ran
the **entire restoration** and discarded the pixels, then encoded the untouched frames to a
throwaway file — at `output_path or <source>.analysis.mp4`, so a preview with no output path wrote
a video next to the user's source.

Measured on a 96-frame clip: **162 s to analyze, 153 s to process.** A preview that costs more than
the job it previews is not a preview, and §5.2.5c wants it precisely so the user can see what would
be altered *before* committing. R-04 lists that preview as the mitigation for false positives
altering clean footage — the risk this project has measured as its largest.

### Decision

The transform returns immediately after tracking when the job is an analysis. A new media-layer
entry point, `run_analysis`, decodes and hands frames to the transform without ever opening an
encoder. `sampleEvery` — in the protocol table since v1.0 and accepted by nobody — is honoured.

### Consequences

- **162 s → 35 s** on the same clip, with identical detections (177 regions); `--sample-every 4`
  brings it to 11 s.
- `framesSeen` comes from the decoder and `framesExamined` from the sampler. Under sampling the
  transform's own counter reported 93 frames for a 96-frame file.
- The analysis summary has no `frameCountPreserved`: no file was written, and claiming a preserved
  timeline would be asserting something about bytes that do not exist.

---

## D-24 — The observation is given, not computed

**Status:** accepted (2026-08-22) · **Closes:** a defect in §5.7's restoration path

### Context

Both restoration paths built their observations by applying the forward operator to the crop:

```python
reconstruct([Observation(block_average(crop_target, profile, phase), 0.0, 0.0)], ...)
```

The frame already contains the block averages where the mosaic is — `block_average` is what
produced them. Applying it again re-quantised the mosaic onto the **estimated** grid, shifting every
block whenever the phase estimate was off, and destroyed the clean picture inside the ROI rectangle
but outside the mask, which the dilation-and-feather blend then composited back over the original.

It was invisible because single-frame back-projection is a **no-op by construction** — the residual
has zero mean inside every block, so its block average is identically zero. The solver could not
undo what was done to its input, and it could not be blamed for it either.

### Decision

Observations are passed as observed. The forward operator is applied only where the model calls for
it, inside the solver.

### Consequences

Measured on one clip, one variable changed:

| | before | after |
| --- | ---: | ---: |
| pipeline's share of the untouched-region loss | 1.63 dB | **0.26 dB** |
| damage on altered pixels outside the region | −4.21 dB | **−0.52 dB** |
| restorations that helped | 20 of 214 | **68 of 208** |
| inside the region, ungated | −0.759 dB | −0.385 dB |

- Guards: `test_a_single_observation_is_an_exact_no_op` pins the invariant that made this
  undetectable, and `test_re_averaging_an_observation_destroys_picture_the_solver_never_touches`
  pins the damage itself.
- **The model is still not mask-aware.** The forward model assumes the whole crop was block-averaged
  when only the masked region was. Neighbour crops carry clean observations of content that is
  mosaicked in the target — which is the entire reason multi-frame could work — and nothing yet
  distinguishes the two.

---

## D-25 — A confidence gate that means what it says

**Status:** accepted (2026-08-22) · **Closes:** §5.8.1 R-8.1c

### Context

`minRestorationConfidence` had existed since v3.1 and had never been exercised. Measuring it turned
up three defects at once, each of which made the gate quietly do something other than what it said.

1. **The threshold was unreachable.** Release required `confidence > threshold + margin`, and the
   confidence formula tops out at `0.25 + 0.35 + 0.4 · blockPenalty` — 0.90 for a 10 px mosaic. No
   threshold above 0.85 could ever open the gate; above 0.90 it silently meant "never restore".
2. **A track started open**, so every new track was restored for `hysteresis − 1` frames regardless
   of confidence. A gate set above every reachable confidence still let two frames per track
   through, which is how this was found.
3. **The gate was fed a raw per-frame signal** while its parameter is named `smoothedConfidence`.
   Being per track and sticky in both directions, one long track opened on a run of good frames and
   coasted through the bad ones.

### Decision

- The margin guards the **closing** side: open at `confidence >= threshold`, close below
  `threshold − margin`. A threshold now means "restore where confidence is at least this".
- **A track starts gated.** Restoration is an intervention: it takes evidence to begin, not evidence
  to stop.
- `ConfidenceSmoother` damps confidence per track before the gate sees it, with a time constant of
  `1 / hysteresisFrames` — the gate's own window, so the two cannot drift apart.

### Consequences

| fed to the gate | best threshold | weighted dB | kept |
| --- | ---: | ---: | ---: |
| raw per-frame confidence | 0.91 | 0.0000 (withholds everything) | 0 |
| **smoothed, α = 1/3** | **0.88** | **+0.0511** | 18 |
| per-region ideal, no hysteresis — flickers | 0.88 | +0.0751 | 47 |

- Gate and smoother are both locked by `fixtures/parity/policies.json`; the gate had been mirrored
  in two languages and locked by nothing, and had the same hole in both.
- **The default stays 0.0 (off).** 0.87 scores −0.60 dB and 0.88 scores +0.05 dB on one synthetic
  clip. That knife-edge is a reason to record the number, not to ship it.
- `eval_gate.py` now carries a *shipped gate* arm alongside the idealised per-region sweep. The
  idealised sweep picked an operating point the real gate could not reach — calibrating against a
  model of the thing rather than the thing is how that gets shipped as "calibrated".

---

## D-26 — The forward model describes the mask, and the solver keeps its best iterate

**Status:** accepted (2026-08-23) · **Supersedes the calibration of:** D-16, §5.6

### Context

The forward operator declared every pixel of every frame block-averaged. A mosaic covers part of a
frame: where it does not, the frame observes the scene **directly**, at full resolution. A neighbour
that saw content the target has lost is not weak evidence about it — it is the answer, and the model
was discarding it.

Separately, back-projection with a dense flow **diverges**. The to-target and to-neighbour fields
are estimated independently, so the forward warp and the back warp are not exact inverses and the
iteration is not a descent on a consistent objective. Measured with the real aligner: +0.58 dB at 5
iterations, −0.18 at 20, −2.79 at 40. **The pipeline ran 20.** The stopping rule watched the
*change* in the residual, which divergence never produces.

### Decision

- `forward_and_adjoint` applies block averaging where a frame is mosaicked and the identity where it
  is not, with the adjoint to match. The block mean is taken over masked pixels only, so a block
  straddling the boundary is not diluted.
- Both solvers keep the iterate with the **lowest data residual** and stop when it stops being the
  latest one. The residual is the only signal available at runtime; returning the last iterate
  computed was returning whatever divergence had reached.

### Consequences

- The convergence fix is an unambiguous win: 40 iterations went from −2.79 dB to +0.57, and the
  pipeline improved from −0.385 dB to **−0.311 dB** inside the region.
- The mask model is **implemented and switched off** at `MASK_MODEL_MIN_NEIGHBOURS = 16`. Below that
  it costs about 0.12 dB; above it, it is worth 4.4 dB against the old model. The window policy caps
  K at 9, so it is currently never taken — deliberately, and it engages by itself when the policy is
  re-measured.
- **D-16 and §5.6's window table are calibrated against a model the code no longer uses.** Under the
  old model more frames genuinely did hurt; under the corrected one the direction reverses. Nothing
  in `WINDOW_BY_MOTION`, `PRESET_MAX_WINDOW` or the cap of 9 should be trusted until it is
  re-measured.
- Guards: `test_the_mask_model_turns_a_neighbour_into_a_direct_observation` and
  `test_dense_flow_back_projection_does_not_walk_away_from_the_answer`.

---

## D-27 — The evaluation clip was in the regime the design calls hopeless

**Status:** accepted (2026-08-23) · **Affects:** every end-to-end number in this repository

### Context

The ladder input applies a mosaic to a drifting ellipse. Measured properly — with the exact ellipse
and motion taken from aligning the *clean* frames — the mask moves 3 px per frame over content that
moves 0.1 to 0.6 px per frame. So **1.6%** of what the target lost was ever seen clean by its
neighbour at t−1, and 3.5% by t−2.

The mosaic slides over near-static content: the same picture is covered in every frame. That is the
object-anchored regime §1.4.1 predicts to be unrecoverable and the Phase 0 gate measured at −0.79 to
−1.50 dB.

An earlier reading of 40% / 55% came from deriving the mask as `|clean − degraded| > 12`. That marks
only pixels the block average moved far, so it is a holey mask rather than the region, and it
overstated the diversity by more than an order of magnitude. **A threshold on a difference is not
ground truth**, even when the degradation is synthetic and the truth is available exactly.

### Decision

`artifacts/screen_clean.mp4` and `artifacts/screen_input.mp4`: content panning 4 px/frame past a
mosaic that does not move. Coverage there follows `2·d / (π·rx)` — 1.8% per neighbour, matching the
arithmetic to a tenth of a percent.

### Consequences

- Every end-to-end comparison in this repository was measured in the unrecoverable regime. They are
  still valid as measurements of *damage* — which is what they were used for — and worth nothing as
  measurements of restoration.
- The generator belongs in `scripts/`, alongside the ladder's, rather than in a scratch file. It is
  not there yet.
- The screen-anchored clip is synthetic in a second way: the pan is a crop of a real frame, so the
  motion is a pure translation with no parallax, occlusion or object motion. It exercises the
  mechanism; it does not represent real footage.
