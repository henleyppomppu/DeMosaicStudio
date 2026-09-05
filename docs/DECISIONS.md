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

---

## D-28 — Evidence is carried forward, not re-gathered every frame

**Status:** accepted (2026-08-23) · **Supersedes:** the batch window of §5.6 · **Closes:** the first
positive end-to-end result this project has produced

### Context

The corrected forward model (D-26) needs about 17 neighbours before it pays, because a neighbour one
frame away exposes only `2·d / (π·rx)` of the region — 1.7% for a 300 px mosaic at 4 px/frame. The
batch form gathers that window afresh for every frame, so it costs K alignments and K times the
solver work per frame. Measured:

| neighbours | solve | align | per frame | fps |
| ---: | ---: | ---: | ---: | ---: |
| 2 (today) | 294 ms | 219 ms | 634 ms | 1.58 |
| 16 | 2217 ms | 1748 ms | 4086 ms | **0.24** |
| 24 | 3444 ms | 2622 ms | 6188 ms | **0.16** |

Against §6.1's ≥4 fps target that is 17× to 25× too slow, and the solver costs *more* than the
alignment — so "8× the alignment work" understated it.

### Decision

One estimate per track, carried forward. Each frame it is warped by a **single** frame-to-frame
flow and the new observation is folded in. The cost is one alignment and one warp regardless of how
far back the evidence goes, and the history is unbounded rather than K.

The router's "how many neighbours are aligned right now" becomes "how deep is the chain", which is
the same question asked of a structure that has an answer.

### Consequences

**It is faster.** 231 ms per frame against 634 ms — **4.33 fps** on the measured ROI, the first time
this pipeline has met §6.1's MVP target.

**It is also better**, for a reason `docs/phase2-alignment-report.md` §3 already measured: shorter
baselines align better. The batch form reaches across the whole window; this chains one-frame
alignments.

| | inside the region | vs input | SSIM | frames improved |
| --- | ---: | ---: | ---: | ---: |
| **screen-anchored clip** | | | | |
| mosaicked input | 25.306 | — | 0.7756 | — |
| batch, K=3 | 25.126 | −0.180 | 0.7675 | 51% |
| **accumulator** | **27.110** | **+1.804** | **0.8178** | **76%** |
| **object-anchored ladder clip** | | | | |
| mosaicked input | 32.854 | — | 0.9144 | — |
| batch, K=3 (shipped yesterday) | 32.543 | −0.311 | 0.9117 | 42% |
| **accumulator** | **34.358** | **+1.504** | **0.9248** | **97%** |

It wins on the object-anchored clip too, which the batch form could not: 1.6% of the region per
frame is nothing to a window of 3 and a third of the region to a chain of 24.

**Looking at it**, the blocks are gone and the structure is back — and the replacement carries
**directional streaking**, which is what repeated warping along the motion leaves behind. That is
the next thing to attack, and it is a far better problem than the one before it.

**The resets are the correctness.** A scene cut, a dropped frame, a track that jumped or a lost
alignment all mean the accumulated pixels are about different content, and compositing them would
put one shot's picture into another. Each is a guard test in `test_accumulator.py`.

**Two defects the wiring found**, both invisible from the outside:

- `target_index` indexes the *rolling history buffer*, which stops advancing once the buffer is
  full. Using it as a frame number restarted the chain on every frame while the logs looked normal.
- The ROI applies reflect padding, so `Roi.bounds` is not the rectangle `Roi.crop` returns.
  Comparing the wrong rectangles mismatched shapes by the padding. `Roi.crop_bounds` now names the
  one that means "what the crop covers".

**§5.6's window is now mostly vestigial.** It still gates *whether* multi-frame runs — through the
motion band and the object-anchored rule — but its size no longer sets how much evidence is used.
`WINDOW_BY_MOTION` and the cap of 9 should be re-read in that light rather than trusted.

---

## D-29 — Evidence is forgotten on a horizon, and the streaking was mostly not there

**Status:** accepted (2026-08-23) · **Corrects:** D-28's description of the artefact

### What the artefact turned out to be

D-28 reported "directional streaking" in the restored region. Measured, that was an over-reading.
The error's high-frequency horizontal energy is **lower** in the output (0.352) than in the
mosaicked input (0.391), and the offline prototype — no detector, no ROI, no blending, no encoder —
produces the same texture. Most of what looked like streaking is residual block structure and
partially recovered detail: an incomplete restoration, not an added artefact.

Two ablations pinned the loop down:

| | PSNR | ripple |
| --- | ---: | ---: |
| the mosaicked input | 24.12 | 0.391 |
| **warping alone, from a clean frame** | **13.19** | 0.146 |
| folding alone, nothing moving | 24.11 | 0.394 |
| both, as shipped | 28.96 | 0.352 |

Chaining 24 warps of a *clean* frame costs 11 dB. Folding, with nothing moving, costs nothing. So
the warp is where the damage is and the fold is what keeps the chain honest — it re-anchors the
estimate to a real observation every frame.

### The real defect

Quality does not rise forever with the chain. It peaks and falls, because the oldest evidence in a
chain of N has been warped N times and carries N frames of the flow's error:

| depth | 4 | 8 | 12 | 16 | 24 | 32 | 48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| screen-anchored | +3.28 | +4.07 | +4.07 | **+4.10** | +3.78 | +3.59 | +3.27 |
| object-anchored | +1.71 | +2.49 | +2.95 | +3.23 | +3.31 | **+3.35** | +2.72 |

### Decision

An exponential forgetting horizon: each frame the carried estimate is decayed towards what that
frame observed, by `1 / EVIDENCE_HORIZON_FRAMES`. A depth **cap** would be a reset — it discards
the chain and restarts at zero, which oscillates. Decaying bounds the horizon and keeps the chain.

Measured at depth 48:

| horizon | off | 64 | **32** | 16 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen | +3.27 | +4.49 | **+5.28** | +5.97 | +6.01 |
| ladder | +2.72 | +2.85 | **+2.81** | +2.54 | +1.98 |

32 improves both. **The optimum itself differs** — 8–16 for the fast pan, 32–64 for the slow drift
— so a frame count is not the governing quantity. Two clips cannot say what is; accumulated warping
error, which grows with motion per frame, is the candidate.

### Consequences

| | before | after |
| --- | ---: | ---: |
| screen-anchored, vs input | +1.804 dB, 76% of frames | **+2.296 dB, 84%** |
| object-anchored ladder | +2.087 dB, 99% | **+2.217 dB, 100%** |

Through the pipeline, the gain rises with the chain's depth exactly as the design says it should:
**+0.84 dB** at depth 0–4, **+1.56** at 5–19, **+2.15** at 20 or more.

### Four fixes that were measured and rejected

Recorded so nobody spends the afternoon again:

- **Anchoring the estimate and composing flows** — hold the estimate in the chain's first frame and
  warp only corrections, so the estimate is never cumulatively resampled. **Much worse**: 11.51 dB
  at depth 24 against 28.96. Composed flows drift, and a drifting flow misplaces everything at once.
- **Skipping the warp when nothing moves** — 1.64 dB in the static tail became −1.11. Folding keeps
  paying even without motion, because it re-anchors the estimate; freezing it does not.
- **Decaying by forward–backward flow confidence** rather than uniformly — costs 2.24 dB.
- **Damping the fold** (step 0.5, 0.25) — costs 0.9 and 3.2 dB.

Only the uniform horizon helped, and it is the one that touches how long evidence lives rather than
how strongly each frame speaks.


---

## D-30 — The calibrated detection threshold was calibrated in the wrong regime

**Status:** accepted (2026-08-23) · **Reverses:** the operating point of `docs/detector-calibration.json`

### Context

The mask threshold was swept and set to 0.9, and the ablation ladder showed it helping: it cut the
region count 4.6× and the output improved. That was measured while **restoration was net-harmful**.
When every restoration damages the picture, fewer regions is strictly better and a tight threshold
looks like a calibration.

Now that the pipeline restores, the same ladder run on both clips says the opposite:

| rung | screen-anchored | object-anchored ladder |
| --- | ---: | ---: |
| detector v0.1.0, mask 0.5 | +2.89 dB | +2.92 dB |
| + detector v0.2.0 | **+3.21** | +2.50 |
| + mask 0.9 ("calibrated") | +2.51 | +2.33 |
| + CRF 12 | +2.30 | +2.22 |

A tight threshold now withholds restorations that would have helped. Both clips agree on the
direction, which the earlier measurement could not have shown.

### Decision

The CLI default returns to **0.5**, which is the worker's own default and what the detector was
trained at. `docs/detector-calibration.json` stands as a record of the sweep, not as an operating
point.

### Consequences

- **Any threshold calibrated against a harmful restoration is measuring harm avoidance**, not
  detection quality. That applies to the confidence gate too: its oracle is now worth only +0.18 dB
  over ungated on the screen clip, against +1.5 dB when restoration was damaging. Its default stays
  0.0 — for a good reason now rather than a cautious one.
- **v0.2.0 is no longer clearly better than v0.1.0**: +0.33 dB on the screen clip, −0.42 on the
  ladder. The retraining was justified by a false-positive measurement that is still valid; what
  changed is that false positives cost less when the thing they trigger is useful.

---

## D-31 — The motion bands were re-measured, and the fast band is the best one

**Status:** accepted (2026-08-23) · **Supersedes:** D-16's motion table

### Context

`WINDOW_BY_MOTION` switched multi-frame **off** for static and fast content: static had no phase
diversity to exploit, fast had no content correspondence left across a long baseline. Both were
measured — with the batch solver, which reaches across the whole window.

The accumulator chains one-frame baselines. Re-measured against it, on clips built at a range of
pan rates:

| pan px/frame | band | gain |
| ---: | --- | ---: |
| 1 | slow | +2.83 |
| 2 | medium | +2.43 |
| 4 | medium | +2.87 |
| 8 | medium | +2.63 |
| 16 | **fast** | **+5.03** |
| 24 | **fast** | **+6.45** |

The fast band is the *best* one, because fast motion is exactly what exposes the region soonest.
The old rule was not miscalibrated; it was **inverted** for this solver.

### The window is inert

Measured on the same clip: `temporalWindow` of 1, 3 and 9, and the Fast and Quality presets, all
produce **+2.83 dB**. The accumulator ignores K. Only `> SINGLE_FRAME` is load-bearing, and only
because the router reads it.

### Decision

Every band permits multi-frame, and `WINDOW_BY_MOTION` is 3 throughout. The number is kept rather
than removed so routing reasons stay comparable across versions and a future table that sets a band
back to 1 is a deliberate act.

**The medium-band alignment rule stays.** It has not been re-measured against the accumulator, and
removing rules because a neighbouring one was disproved is how a measured policy turns back into a
guessed one. The same applies to the object-anchored rule: the accumulator gains on the ladder clip,
but that clip's *grid* is screen-anchored — only its mask moves — so a genuinely object-anchored
grid remains untested.

### Consequences

- **Quality is unchanged**: +2.296 dB and 84% of frames, before and after. The window was already
  inert, so this changes nothing about what the pipeline does.
- **It changes what the pipeline says it does.** On the screen clip, `SufficientTemporalEvidence`
  went from 145 to **191** and `MotionOutsideOperatingWindow` from 61 to **2**. The router was
  reporting a decision it did not make — the same family of defect as `passthrough` and `analyze`.
- `make_policy_fixture.py` now generates the whole parity fixture. The confidence gate and smoother
  sections had been added by hand, and the first regeneration silently deleted them: **a generator
  that writes part of a file deletes the rest.**

---

## D-32 — The flow is estimated below full resolution, and only in one direction

**Status:** accepted (2026-08-23) · **Closes:** §6.1 throughput, and 0.8 dB nobody was looking for

### Context

Alignment is half the per-frame cost — 107 ms of the 231 ms — and it runs RAFT **twice**: forward
for the flow, backward for the forward-backward consistency that becomes the per-pixel confidence.

The accumulator uses the flow every frame. It uses the confidence only as a **scalar** — "is this
alignment usable at all" — because decaying by it per pixel was measured to cost 2.24 dB (D-29).

### Two findings, one of them not what was being looked for

**The backward pass is free to drop.** Estimating the confidence from a photometric residual instead
— warp the neighbour onto the target and see where it lands — gives **the same result** (+2.81 dB
either way) for **half the time** (104 ms against 49).

**Full resolution is the worst place to estimate the flow.** This was a search for speed and turned
out to be a quality knob:

| clip | 1.00 | 0.75 | 0.50 | 0.35 |
| --- | ---: | ---: | ---: | ---: |
| screen | +2.81 | **+5.39** | +5.07 | +4.84 |
| pan1 | +3.93 | +4.71 | **+4.76** | +4.74 |
| fast16 | +5.35 | **+5.91** | +5.65 | +5.83 |

Full resolution is worst on all three, by 0.8 to 2.6 dB. The timing barely moves — at these crop
sizes RAFT is dominated by fixed overhead — so this is not a speed/quality trade at all.

Two mechanisms fit. The mosaic is a screen-fixed high-frequency texture competing with the content
for the flow estimator's attention, and downscaling suppresses it. Or downscaling shrinks the
displacement into the range a small model handles best. The `pan1` row — 0.79 px of motion, and
still better downscaled — favours the first. Three clips cannot separate them.

### Decision

`DEFAULT_FLOW_SCALE = 0.75`, and the backward pass is optional with the pipeline not asking for it.

**The scale never takes a side below `MIN_FLOW_SIZE`.** A crop already smaller than that is padded
up to it, so shrinking it first replaces content with padding: a 24 px region at 0.75 drops from a
usable fraction of 0.9 to **0.31**, and small ROIs are this pipeline's common case.

**`neighbour_to_target` is absent rather than approximated** when the backward pass is skipped. A
plausible stand-in would be used by `reconstruct_flow` without anyone noticing it was not the real
thing.

### Consequences

| | before | after |
| --- | ---: | ---: |
| screen-anchored | +2.296 dB, 84% of frames | **+3.072 dB, 97%** |
| object-anchored ladder | +2.217 dB, 100% | **+3.013 dB, 100%** |
| 320×240, 40 frames | +2.83 dB, 4.49 fps | **+4.72 dB, 6.42 fps** |
| 1680×800, 96 frames | 1.12 fps | **1.37 fps** |

Quality and speed together, which is rare enough to be suspicious — the reason it happens is that
the two changes are independent: one removes work, the other removes a distraction.

---

## D-33 — The host is four layers, and only one of them cannot be tested

**Status:** accepted (2026-08-23) · **Closes:** Phase 4's absence

### Context

The Domain had policies and nothing that used them. The only way to run the pipeline was
`scripts/run_job.py`, which is a Python client for a product whose host is meant to be WPF.

### Decision

`Application` says what a job is and what may be done to one. `Infrastructure` knows the engine is a
process speaking JSON Lines. `App` is a window. AGENTS.md's layer rules are what shape this, and the
line they draw is the useful one: **`IRestorationEngine` says nothing about processes**, so the
queue's rules are tested with neither a worker nor a GPU.

Almost nothing lives in the view model. Which jobs may be cancelled, what a late progress message
does, whether a status may move — all of that is `JobList`, and all of it has tests. What is left in
`App` is the dispatcher marshalling and the display strings, which is the part that genuinely needs
a window.

### The first round-trip test found a defect immediately

`configure_stdio_utf8` configured stdout and stderr and **never touched stdin** — the channel
requests arrive on, carrying file paths. On a Korean-locale machine stdin is cp949, so a host that
correctly wrote UTF-8 had its path mis-decoded and the worker died mid-handshake.

`run_job.py` never showed it because every path it was ever given was ASCII. A C# test against a
directory named in Hangul found it in one run. On this machine that is not an edge case; it is the
normal one.

### Consequences

- 51 host tests: 25 on the queue's rules, 21 on the codec, 5 round-trip against the real worker.
- The round-trip tests **skip and say why** where the interpreter or the model weights are absent —
  both are part of the machine, not the repository. `Xunit.SkippableFact` is there because xunit 2
  cannot skip at run time and a test that quietly passes when its environment is missing is the
  defect this repository keeps finding.
- `JobQueue` is `JobList`: it has no FIFO semantics and the suffix would promise them (CA1711).
- Verified: the application starts, spawns the worker, closes cleanly, and leaves no orphan.
- **Not verified:** nobody has processed a video through the window. The engine underneath it has
  round-trip tests; the buttons have been clicked by no one.

---

## D-34 — Looking forward as well as backward is not worth the architecture

**Status:** accepted (2026-08-23) · **Closes:** the look-ahead item that has been open since D-19

### Context

The accumulator carries one estimate forward, and coverage governs the gain (+0.84 dB at depth 0–4,
+2.15 at 20 or more). A frame in the middle of a shot has as much future as past, so a second chain
running backwards looked like twice the evidence at the same O(1) cost.

The price is architectural: the media layer hands each frame to the encoder as it is produced, and a
backward chain needs the whole shot in hand before the first frame can be written. D-19 recorded the
causal window as a deviation from §5.6 and named look-ahead as the correct fix.

### Measured, on three target frames of each clip, at depth 24

| clip | backward alone | both, averaged | difference |
| --- | ---: | ---: | ---: |
| screen-anchored | +6.57 dB | +6.62 dB | **+0.05** |
| object-anchored | +4.31 dB | +4.90 dB | **+0.59** |

**The evidence is not doubled, it is duplicated.** Content that passes through a screen-fixed mosaic
is seen outside it on both sides — the same pixel, twice, not two different pixels.

And **backward beat forward at every one of the six targets**, so no selection rule can do better
than the backward chain alone. What the averaging buys on the ladder clip is error cancellation, not
new information. At one target it is actively worse: 36.24 dB backward against 34.02 averaged, where
the pan had run out and the forward chain had nothing to work with.

### Decision

The window stays causal. D-19's deviation from §5.6 is no longer a deviation to be corrected — it is
what the measurement supports.

### Consequences

- The media layer keeps streaming, which is what keeps memory flat and lets the first frame be
  written immediately.
- If a future solver makes the two directions carry different information — a learned restorer that
  weights evidence rather than averaging it — this is worth re-measuring. Nothing here says
  look-ahead is useless; it says it is worth 0.05 dB to *this* solver.

---

## D-35 — The Speed encoder profile is not worth wiring, and the reason is that encoding is 1.4%

**Status:** accepted (2026-08-23) · **Defers:** §5.1.4's NVENC path indefinitely

### Context

PyAV bundles its own FFmpeg without NVENC, so the Speed profile has to shell out to
`tools/ffmpeg` — which does have `h264_nvenc`, `hevc_nvenc` and `av1_nvenc`. The profile currently
refuses rather than silently giving the user x265 while the settings say NVENC.

Wiring it means piping restored frames to a subprocess. Doing that **without losing timestamps**
means a lossless intermediate stream carrying PTS — raw video over a pipe has none, and ffmpeg would
synthesise them from a frame rate. §5.1.7's guarantee, that every output frame carries its source
PTS, is the thing the media layer was built around.

### Measured

| stage | ms/frame |
| --- | ---: |
| detection | 119 |
| one alignment | 107 |
| accumulator | 6 |
| **decode + encode + mux, x265 `fast` CRF 12** | **9.9** |

On a real job — the screen clip at 1680×800, 96 frames in 70 s — that is **1.4%** of the wall clock,
and the 9.9 ms includes decoding, so the encode alone is less.

### Decision

Deferred. NVENC could save at most 1.4%, and the price is a lossless intermediate pipe plus the
central timestamp invariant.

The profile keeps refusing rather than substituting. A user who asks for NVENC and silently gets
x265 has been told something untrue about their output.

### Consequences

- §5.1.4's premise — that the encoder is where the time goes — is false for this pipeline. It was
  written before there was a pipeline to measure.
- If detection and alignment ever drop by an order of magnitude, this is worth re-reading. Until
  then the two of them are 97% of the cost and the encoder is a rounding error.

---

## D-36 — A retraining measured on one film is not evidence that it generalised

**Status:** accepted (2026-08-23) · **Qualifies:** the v0.2.0 detector decision

### Context

The corpus was 24 clips of Tears of Steel. v0.2.0 was retrained on widened negatives, and the
retraining was justified by measurement: video-level firing fell 18.8% → 9.4% for 0.032 of IoU. Two
more films — Sintel and Big Buck Bunny — now say what that measurement could not.

### Measured, on clean frames the detector has never seen

Fraction producing at least one region. §5.2.5a asks for **≤ 0.5%**.

| threshold | 0.5 | 0.9 | 0.99 | 0.999 |
| --- | ---: | ---: | ---: | ---: |
| **v0.1.0** tos / sintel / bbb | 80.6 / 77.8 / 89.6% | 48.6 / 41.7 / 73.6% | 17.4 / 11.1 / 41.7% | 4.2 / **1.4** / 10.4% |
| **v0.2.0** tos / sintel / bbb | 56.2 / 61.8 / 84.0% | 30.6 / 26.4 / 47.2% | 7.6 / 13.2 / 26.4% | 3.5 / **6.2** / 16.7% |

Two things, and the second is the one that matters.

**No threshold on either model meets §5.2.5a on any source.** The best figure anywhere is 1.4%, at
an operating point that would detect nothing.

**v0.2.0 is better than v0.1.0 on Tears of Steel at every threshold, and worse on both unseen films
at the high ones** — 1.4% → 6.2% on Sintel, 10.4% → 16.7% on Big Buck Bunny. That is what
overfitting to a negatives corpus looks like from outside: the model learned which textures *in this
film* are not mosaics.

### Decision

v0.2.0 stays, because the pipeline ships threshold 0.5 and it is better there on all three sources.
What does not stand is the claim that the retraining generalised, and `docs/phase1-detector-report.md`
should be read with this next to it.

**§5.2.5a cannot be met by tuning.** No threshold reaches it. That is a training-data problem, which
is what `scripts/fetch_negatives.py` and the collected negatives are for.

### Consequences

- **Any measurement on one source is a measurement of that source** until a second one agrees. This
  is the fifth time this repository has drawn a conclusion that a second measurement overturned;
  it is the first time the second measurement was different *content* rather than a different
  method.
- **Flat cartoon shading is the worst case by a wide margin** — 84% at the shipped threshold, 16.7%
  even at 0.999. Large regions of near-constant colour with hard edges and no grain is what a mosaic
  looks like when nothing is wrong.
- The restorer is content-dependent too: **+7.05 dB** on the screen-anchored clip, **+0.39** on
  Sintel, **−0.07** on Big Buck Bunny. Every headline this project has quoted came from the first.
  Future figures belong per content class or as a range, never as one number.

---

## D-37 — The worker reads stdin on its own thread, because Stop could not otherwise work

**Status:** accepted (2026-08-23) · **Fixes:** the claim §8.5.3 and `main_loop.py` both made

### Context

`Worker.run` was `for line in sys.stdin:` on one thread, and `handle()` for a `process` message ran
the entire job inline. Between those two facts, **the loop is not reading stdin for the whole
duration of a job**. A `cancel` sits in the pipe until the work it was meant to interrupt has
finished; `context.cancelled` is then set on a job that has already ended.

Every downstream cancellation check — the one at the top of `transform`, the `status = "cancelled"
if context.cancelled` in `_start` — was therefore unreachable in practice. The module docstring said
"the host asks; the worker drains, checkpoints, and emits a terminal `result`", and nothing
implemented it. The Stop button did nothing, and that is what the user reported.

### Decision

Reading and dispatching are separate threads. The reader parses stdin and answers the four
**interrupts** — `cancel`, `pause`, `resume`, `shutdown` — itself, because they only set a flag the
running job polls. Everything else goes on a queue the dispatch thread drains one at a time, so the
protocol's guarantee about *work* ordering (§8.5.2, one job at a time) is untouched.

Three details are load-bearing:

- **`shutdown` is both.** It is applied as an interrupt *and* queued, because stopping the loop is
  ordered work: handling it only on the reader would drop requests already waiting in the inbox.
- **A cancel that beats the job to the starting line is remembered.** The reader can answer a
  `cancel` in the window between the host sending `process` and `_start` assigning `current` — so a
  cancel with no job to attach to is recorded by id and applied when that id starts.
- **stdout now has two writers**, so `Emitter.send` takes a lock. One JSON Lines message is one
  line, and two unlocked writes can interleave into a line that parses as neither.

### Alternatives rejected

- **Polling stdin between frames.** Non-blocking stdin on Windows means `msvcrt` or an overlapped
  handle, and the job loop would have to call it — spreading protocol concerns into the pipeline.
- **Killing the process on cancel.** That is what the cooperative design exists to avoid: a worker
  killed mid-write loses the checkpoint it was about to write, so the next attempt starts at zero.

### Measured

Real worker, real 1080p clip, cancel sent 45 s into the job: acknowledged in **under 0.1 s**,
terminal `result` with `status="cancelled"` **0.8 s** later, after the encoder closed its file.

### Guards

`worker/tests/test_main_loop.py`. The stdin fixture releases the cancel only once the job has
started — a `StringIO` holding both lines up front cannot tell the two designs apart, because by
the time the job ended the flag would be set either way.

---

## D-38 — Progress is reported from the decode, and offered every frame

**Status:** accepted (2026-08-23) · **Fixes:** two defects with the same symptom

### Context

Progress was emitted at one point in the frame transform: after a frame had been restored. Four
early returns sat above it — cancelled, detector failure, **no restorable region**, analysis — and
the third is the ordinary case, not the rare one. A video with no mosaic in it, or a stretch of one
between regions, reported 0% for its whole run while the decoder worked through it.

The emit was also gated on `index % 8 == 0`.

### Decision

The report is offered at the top of the transform, before anything can return, and on **every**
frame.

The stage is chosen by the job's mode rather than by what the frame turned out to contain: §8.4
forbids a stage moving backwards, and picking it per-frame would do exactly that on the first frame
with no region in it.

Dropping the stride is the part worth explaining. `Emitter.progress` already rate-limits to four a
second, and that limit does its job whatever it is offered. A fixed stride is not a second limiter —
it is a **floor on the interval**, and the two compose badly: on hardware managing about a frame a
second, every eighth frame is one update every eight seconds.

### Measured

Same clip, same machine, 1080p detection on the CPU. First figures are the stride, second are
without it:

| | updates in 43 s | worst gap |
| --- | ---: | ---: |
| every 8th frame | 4 | 20 s |
| every frame | 22 | 2.7 s |

Neither run approached the four-a-second ceiling, which is the point: the limiter was never what
was throttling this.

### Guards

`test_a_job_with_nothing_to_restore_still_reports_progress`, which lifts the rate limit — at four a
second a short fixture can legitimately emit only the two forced endpoints, and the defect would
survive the test.

---

## D-39 — The window's rows are updated in place, not rebuilt

**Status:** accepted (2026-08-23) · **Follows from:** D-38

### Context

`JobList.Changed` hands out the whole list, and the view model rebuilt `Rows` from it: `Clear()`,
then add. That is a fine shape for a list that changes when the user does something. It stopped
being fine the moment progress started arriving several times a second (D-38), because rebuilding
drops the `ListView`'s selection — so a user selecting a job in order to Remove or Retry it would
have the selection taken out from under their hand before they could press the button.

This is a defect D-38 *created*. Before it, progress moved rarely enough that nobody noticed.

### Decision

Rows are matched by id and mutated. `JobRow` became a mutable class implementing
`INotifyPropertyChanged` rather than a record, which is the cost: an immutable row cannot be updated
in place, and replacing it is what breaks selection.

The reconciliation removes rows whose jobs are gone, updates the ones that remain, inserts new ones
at the job's index, and moves a row whose position changed — a retry appends, so positions do move.

### Alternatives rejected

- **Rebinding selection by id after each rebuild.** Restores the selection but not the focus, the
  scroll position or an in-progress rubber-band drag.
- **Rate-limiting the host's redraw instead.** Treats the symptom, and the correct interval is a
  guess that would be wrong on some machine.

---

## D-40 — WPF opts out of invariant globalization, and the window says when it breaks

**Status:** accepted (2026-08-23) · **Fixes:** the settings dialog ending the process

### Context

`Directory.Build.props` sets `InvariantGlobalization=true` for every project. For Domain,
Application and the tests that is right — it is what keeps parsing and formatting identical
whatever locale the machine has, and this machine is Korean.

**WPF does not run under it.** `FrameworkElement.Language` defaults to the XML language `en-US`, and
`BindingExpression.TransferValue` resolves it through `XmlLanguage.GetSpecificCulture()` whenever a
binding converts a value — a `StringFormat`, a converter, or an int reaching a `Text` property. With
the switch on there is no non-neutral culture to find:

```
System.InvalidOperationException: Cannot find non-neutral culture related to 'en-us'.
   at System.Windows.Markup.XmlLanguage.GetSpecificCulture()
   at System.Windows.Data.BindingExpressionBase.GetCulture()
   ...
   at System.Windows.Window.ShowDialog()
   at DeMosaicStudio.App.Views.MainWindow.OnSettings(...)
```

It builds clean and starts clean. It waits for the first such binding. The main window has none —
its rows are formatted into strings in the view model, which was a decision about testability
(D-33) and turns out to have been what kept the application alive. The settings dialog's sliders
show numbers, so it was the first, and opening Settings killed the application.

### Decision

`InvariantGlobalization=false` in `DeMosaicStudio.App.csproj` only. It is a host-level switch read
from the entry assembly's runtime configuration, so it governs the application's own process and
leaves every test process invariant.

The exemption stays one project rather than becoming a habit. Turning it off also re-enabled the
Globalization analyzers in that project, which immediately found a culture-sensitive `ToString()`
on a protocol token — so the narrow scope pays for itself.

### The second defect, which is why this was hard

**The application had no `DispatcherUnhandledException` handler.** An exception reaching the message
loop ended the process with no dialog, no message and nothing on screen. Diagnosing it took Windows
Error Reporting for the exception *type*, and then a UI-automation script driving the button with
stderr redirected, because the process was gone before anything could be read.

A handler now shows the exception and its stack, and keeps the application running: a failure in one
button is rarely a reason to discard the queue as well. It is not a way of ignoring bugs — it is the
difference between a bug report and a disappearance.

`WorkerProcessEngine.StartAsync` was the same shape: a missing interpreter arrives as
`Win32Exception`, which the view model's catch did not cover, and start-up is an `async void` — so a
bad interpreter path ended the process before the window appeared. It is now wrapped as the
`InvalidOperationException` the caller already handles, with the path in the message.

### Guards

`ProjectConfigurationTests`. It reads the project files as text, which is unusual and is the point:
this rule has no compile-time expression at all.

---

## D-41 — The window is Korean, translated in place

**Status:** accepted (2026-08-23) · **Revises:** prd.md §20 Q6

### Context

Q6 answered "English only, strings externalized — low value under D-11, kept because it costs
nothing now and costs a refactor later." The one user this application is for (D-11) is Korean and
asked for a Korean window, so "English only" was an assumption rather than a requirement.

The second half of that answer had not been done either. The UI text was XAML literals and string
switches in the view model, so the externalization that was supposed to make this cheap did not
exist when the moment arrived.

### Decision

Korean, written directly where it is shown: the XAML for the two windows, the display strings in
`MainViewModel` and `JobRow`.

**A resource file was considered and rejected — for now.** The honest accounting:

- The dialog carries about twenty explanatory sentences, several of them long. Replacing them with
  resource keys makes the layout unreadable at exactly the place where the wording matters most.
- A `.resx` reachable from XAML needs a *public* generated accessor, which the SDK's generator does
  not produce; the workaround is a hand-written façade, which is the duplicated edit surface this
  was meant to avoid.
- The application has one user and one language.

So the refactor Q6 hoped to pre-pay for is still ahead, and it is now the price of a second
language. That is a known, bounded cost rather than a surprise, which is the only claim being made.

### Two things that are deliberately *not* translated

**Error-code meanings.** `ErrorCodes` is locked to `worker/demosaic_worker/errors.py` by
`fixtures/parity/error_codes.json` (§13.4) and printed in `docs/ERROR_CODES.md`. Translating in
place would break the fixture and change what two implementations agree on. `ErrorText` in
Application holds a Korean line *beside* each code, the number is always shown with it, and
`ErrorTextTests` fails if a code has no line — or if a line has no code, which is how a typo in a
key would otherwise stay invisible. An unknown code falls back to the English meaning rather than
throwing: a newer worker may report one, and losing the failure over its wording would be worse.

**Enumeration values** — `Balanced`, `H265`, `QualityX265`, `auto`. They are the tokens written to
`settings.json` and sent on the wire; a translated label would stop matching the file the user can
open. The hint under each combo box explains them in Korean instead.

### Culture

`UseKorean()` sets three things, because they are three different things: `CurrentUICulture` picks
resources, `CurrentCulture` picks number and date formatting, and `FrameworkElement.Language` is
what a **binding** consults — WPF defaults that to the XML language `en-US` and leaves it there
however the thread is configured. It is set explicitly rather than read from the machine so the
window reads the same anywhere.

That third one is the property whose resolution ended the process under invariant globalization
(D-40). Setting it here is not what fixed that; the project-file opt-out is.

### Verified

Driven through UI automation against the real executable: menus, buttons, column headers, the empty
-state hint and the §1.3 disclaimer in the main window; the title, three tab headers, buttons and
labels in the settings dialog. Numbers render `0.45`, `0.50` — Korean uses the same decimal point,
so the values are unchanged.

---

## D-42 — A percentage alone cannot answer "is it stuck?"

**Status:** accepted (2026-08-23) · **Prompted by:** a job reported as hung that was not

### What happened

A user watched a job sit at **0%** while the CPU and GPU were plainly busy, and reported it as hung.
It was not. Measured on the live process: 35.7 minutes elapsed, **8.9 of 32 cores** occupied
continuously by the worker, GPU at 25%. It was working the whole time.

The arithmetic explains the rest. Measured throughput is **0.44 frames a second at 1080p**, so ten
minutes is about 270 frames:

| source | frames | after ten minutes | shown as `P0` |
| --- | ---: | ---: | ---: |
| 10 min @ 30 fps | 18,000 | 1.50% | 2% |
| 30 min @ 30 fps | 54,000 | 0.50% | **0%** |
| 60 min @ 30 fps | 108,000 | 0.25% | **0%** |

**For any source over about twenty minutes, 0% is the correct display and it is also useless.**

### Decision

The window shows the rate and the estimate, not only a percentage.

- `eta` has been in the protocol since 1.0 and **nothing had ever filled it in**. The worker now
  does, from the frame rate it already computes.
- `EngineProgress.Fps` was parsed by the codec and then **thrown away by the queue**. It now
  reaches the job.
- The percentage uses one decimal below 10%, so 0.25% reads as `0.3%` rather than `0%`.
- A rate with **no** estimate is how the host learns the container never reported a duration —
  which is also the case where `fraction` is pinned at zero forever. The detail line says so
  instead of showing a zero that will never move.

### The deeper finding: false positives are the throughput problem

One alignment per **track** per frame (D-28), so every spurious region costs a dense optical flow
and a solve. The detector fires on 54–84% of *clean* frames (D-36), and that is not only a quality
defect — it is where the time goes. Measured today on one 1080p clip:

| mask threshold | min region area | fps | speed-up |
| ---: | ---: | ---: | ---: |
| 0.5 (shipped) | 1024 | 0.44 | — |
| 0.5 | 4096 | 0.54 | 1.2× |
| 0.9 | 1024 | 0.74 | 1.7× |
| 0.9 | 4096 | 0.91 | 2.1× |
| 0.99 | 1024 | **1.03** | **2.3×** |

D-30 priced the quality side of that same lever: mask 0.9 costs about 0.7 dB inside the region
(+3.21 → +2.51 on the screen clip). So the trade is roughly **2× throughput for 0.7 dB**, and it is
a trade, not a free win.

### The host was throwing every report away

The above was written after fixing the worker, and the window still did not move. The reason was on
the other side of the seam:

```csharp
return Rank(to) > Rank(from);   // JobStatusTransition.IsAllowed
```

**Strictly greater.** `JobList.Report(EngineProgress)` maps `restoring` to `Processing` and consults
that rule. The first such report moved the job Probing → Processing and was applied. **Every report
after it was refused**, because a job already in Processing cannot "move to" Processing — so the
fraction froze at the 0.0 that arrived with the first, and the rate never reached the window at all.

The queue read `복원 중 · 0.0% · 시작하는 중` for as long as the job ran, which is exactly what a
hang looks like.

Staying in a stage is not leaving it backwards. The comparison is now `>=`, and the terminal rule is
untouched — `IsAllowed(Completed, Completed)` is still false, so a second `result` cannot re-open a
finished job.

**What let this through:** the worker had a test that it emits progress, and the codec had a test
that it parses progress. Neither ran a *sequence* through the queue. `A_run_of_reports_in_the_same_stage_all_land`
does, and it fails without the fix. This is the fourth defect in this repository living exactly on a
seam that both sides tested from their own end.

### Consequences

- **§6.1's ≥4 fps target is missed by an order of magnitude at 1080p.** The 4.33 fps in D-28 and
  the 6.42 fps quoted for the accumulator were measured on far smaller frames.
- A one-hour source is a **67-hour job** at the shipped settings. Nothing in the product says so
  before the user starts it. Estimating up front from a probe is the obvious follow-on and is not
  done.

---

## D-43 — Speed first: single-frame restoration, and the four things that were actually slow

**Status:** accepted (2026-08-23) · **Revises:** D-04 (third-party weights as-is) · **Gives meaning to:** `QualityPreset` (D-31 found it inert)

### Context

The evidence pipeline ran at **0.44 fps at 1080p** (D-42): a one-hour source was a 67-hour job. The
user looked at the output and called the restoration a mess, and asked whether a
decimate → single-frame super-resolution → temporal-blend → feather pipeline would do better,
speed first.

### What was actually slow

Everyone's model of the cost — per-region optical flow and the solver — was wrong. Removing both
moved the frame rate from 0.74 to 1.0. `scripts/profile_job.py` then said where a 1272 ms frame
went:

| function | ms/frame | what it was |
| --- | ---: | --- |
| `extract_regions → _label` | **397** | connected components as a per-pixel Python breadth-first search |
| `detect_cuts` | **268** | all seven frame pairs re-classified every frame, full-resolution histograms |
| detector | 209 | fifteen 512-pixel tiles of a 1080p frame — the network itself was **19.6 ms** |
| `estimate_geometry → _best_period` | 137 | 333,000 `.mean()` calls in nested Python loops |
| `blend_region` | 109 | dilation, box blur, two gradient calls and three `np.power` in numpy per track |

None of it was the restoration. The fixes, each held to its predecessor by a test:

- detector: one pass over the whole frame (it is fully convolutional; the tiles were seams plus
  fourteen extra launches), fp16 on CUDA, uint8 uploaded and scaled on the device → **67 ms**;
- labelling: runs, not pixels — rows cut into runs vectorised, union-find over runs → ~14 ms;
- scene cuts: one new pair per frame, the rest carried with the buffer; histogram by `bincount`
  on the uint8 plane → ~5 ms;
- period search: one reshape per period, same tie-breaking, oracle test against the loop → ~2 ms;
- blend and close: `max_pool2d` / `avg_pool2d` / element-wise on the GPU, numpy forms kept as the
  specification → ~5 ms;
- region extraction: areas by `bincount`, boxes by `minimum.at`, masks built inside the box.

Profiled result: **1272 → 245 ms a frame** in-process before any restorer change.

### The restorer

Three presets, and for the first time they differ (D-31 measured Fast, Balanced and Quality
producing identical output):

| preset | restorer | what it promises |
| --- | --- | --- |
| Fast | decimate to block resolution, bicubic back | removes the grid, leaves blur. No model. |
| Balanced | decimate, compact SR network on every region in one batch, temporal blend | invents plausible detail. Every pixel is a guess. Falls back to Fast with W6101 when no weights are installed. |
| Quality | the evidence accumulator (D-28) | the only path that consults neighbouring frames. ~20× slower. |

**Decimation is what makes third-party weights usable.** D-04 rejected them because they invert
bicubic downsampling, not pixelation, and produce confidently wrong texture when handed a mosaic.
A mosaic of block B *is* a B-times downsample; decimating first hands the network exactly the
input it was trained on. D-04's objection stands for the mosaicked frame and does not apply to
the decimated one. The network is `realesr-general-x4v3` (SRVGGNetCompact, 1.2 M parameters,
BSD-3), reimplemented in thirty lines rather than imported, weights installed by
`scripts/fetch_restorer.py` into the model store with a hash like the detector's.

**CodeFormer and GFPGAN are refused**, and will stay refused however much better a face would
look: they are face-identity priors, which §2.3 C-2 and C-4 rule out permanently, and D-11 does
not relax C-1..C-6.

### The temporal blend nearly shipped as a −5.5 dB defect

On the quality fixture — a screen-fixed mosaic over a panning picture — the proposed
unconditional 7:3 blend scored **−4.52 dB** inside the region against **+1.00 dB** for the same
restorer with no blend. Every frame carried a ghost of the previous one. The blend is now applied
only where the *observation* — the mosaicked crop, deterministic per frame — did not change
between frames:

| motion tolerance (luma levels) | gain | frames improved |
| ---: | ---: | ---: |
| unconditional | −4.52 | 5% |
| 6.0 | +0.20 | 55% |
| 2.0 | +0.97 | 70% |
| **1.0** | **+1.06** | 70% |
| 0.5 | +1.02 | 68% |
| no blend | +1.00 | 68% |

At 1.0 the blend is a small net gain; above 2 the flat parts of a moving picture blend with their
own ghost. A face under a mosaic that follows it is unchanged block to block and blends; a pan is
changed everywhere and does not.

### scipy went in and came out again

`scipy.ndimage.label` was the first fix for the labeller and is faster than the run labeller. It
also **deadlocked every worker on its first frame**: main thread inside `create_module` loading a
scipy extension DLL, the D-37 stdin reader thread blocked on stdin, forever. No in-process test
showed it; every subprocess did — the bench with its minimal PATH and the C# tests with the full
one. `faulthandler.dump_traceback_later` found it. A dependency that hangs the shipped engine
while passing every in-process test is not one to keep for a few milliseconds; the original
author avoided it and was right.

The investigation also found that `WorkerProcessEngine` redirected stderr and never read it.
That is a deadlock of its own class — x265 alone prints twenty lines per encode into a 4 KB pipe —
and is fixed regardless: stderr is drained and forwarded as log lines.

### Quality, measured

Same clip through both restorers after the changes (`test_endtoend_quality.py`, now parametrised
over both):

| restorer | gain inside the region | frames improved | outside |
| --- | ---: | ---: | ---: |
| Quality (evidence) | **+4.72 dB** | 100% | 48.2 dB |
| Fast (single-frame floor, blend at 1.0) | **+1.06 dB** | 70% | 41.6 dB |

The evidence path went from the +2.8 dB this file used to quote to +4.72: the single-pass detector
has no seams, and seams were in the masks. Balanced is not in this table — its weights are not in
the repository and PSNR is the wrong instrument for an inventing restorer (D-36's LPIPS is the
right one, and the measurement is still to be made by eye).

### Throughput, measured

`scripts/bench_throughput.py`, the real worker over stdio, 1080p, nothing else on the GPU:

| preset | detectEvery | fps | vs. D-42's 0.44 |
| --- | ---: | ---: | ---: |
| Fast | 1 | **4.25** | 9.7× |
| Fast | 2 | **5.70** | 13× |
| Balanced (no weights installed → Fast) | 1 | 4.28 | 9.7× |
| Quality (evidence) | 1 | 1.64 | 3.7× |

A one-hour source: 67 hours this morning, **4.4 hours** on Fast at every frame, 3.3 with the
detector on every second frame. Quality's 3.7× came entirely from the shared fixes — it does
exactly what it did before, on the same masks minus their seams.

### Then someone looked — and the default changed

With the weights installed, `scripts/compare_presets.py` put the same frame through every preset
and scored it against the clean picture (the quality fixture: a screen-fixed mosaic over a
panning crop of Tears of Steel):

| | PSNR dB | LPIPS (lower is better) |
| --- | ---: | ---: |
| input (mosaic) | 25.97 | 0.544 |
| **Fast** | 28.17 | **0.115** |
| Balanced (SR) | **20.95** | 0.167 |
| Quality (evidence) | **30.11** | 0.211 |

Balanced scored below the *mosaic* on PSNR and behind Fast on LPIPS, and the picture agreed: a
hard bright blob with a dark halo where the clean frame is a soft one. Quality had the highest
PSNR and the vertical streaking D-29 documented, which LPIPS charged it for.

That could have been the content — the clean frame is itself out of focus — so the network was
isolated on a *sharp* crop, box-decimated 4× (its native scale, a 64×64 input, the ideal case):

| sharp crop, block 4 | luma PSNR | LPIPS |
| --- | ---: | ---: |
| bicubic | **45.2** | **0.070** |
| SR, grey replicated to RGB | 32.7 | 0.188 |
| SR, real RGB | 34.9 | — |

A constant image passes through the network almost unchanged (0.1 → 0.101, 0.9 → 0.920) and
fp16 equals fp32, so the weights and the in-house architecture are right; colour buys 2 dB and
no more. **On clean box-decimated input this network is worse than bicubic on both metrics, on
both blurry and sharp content.** The likeliest reason is what it was trained on: heavily
degraded photographs, whose blur and noise it removes and whose textures it supplies — neither
of which a clean decimation of CG footage has or wants. That is D-04's "confidently wrong
texture" in a new form, wrong sharpness rather than wrong texture, and decimation did not
remove it after all.

**The default preset is Fast.** Balanced stays one click away, and a photographic source may yet
be where the network earns its place — every clip this project owns is a Blender open movie
(D-36's limitation applies here with full force). But a default is what was measured best, and
that is Fast on the footage in hand.

### Settings and protocol

Protocol 1.2. `detection.detectEvery` runs the detector on every Nth frame with the tracker
coasting between (`Tracker.coast`, which does not count a skipped frame as a miss);
`restoration.temporalAlpha` is the blend weight. Both change the output and are in the settings
fingerprint, which invalidated every cached artifact once — the price of a fingerprinted key, paid
knowingly. The parity fixture was regenerated from the Python side and the C# side holds to it.

### Consequences

- §6.1's ≥4 fps target at 1080p is met by the Fast path on the real stdio channel and by Balanced
  once weights are installed; Quality remains below it.
- The default preset is Balanced, which without weights *is* Fast. Nothing in the product invents
  detail until the user runs `fetch_restorer.py`.
- Every profile number above is one clip on one machine. `scripts/profile_job.py` and
  `scripts/bench_throughput.py` are how the next claim gets checked.

---

## D-44 — A diffusion refiner at low strength is the first restorer to beat bicubic perceptually

**Status:** measured, not yet shipped (2026-08-23) · **Phase 0 of** a user-chosen diffusion model behind the single-frame path

### Context

The user asked whether a generative model could paint the region ("LLM으로 그려야 할 듯"). An
LLM cannot; a diffusion model can, and the question is whether it *should* — every pixel it
produces is invented, and D-43 had just measured a learned prior (a compact SR network) making
things worse than bicubic on this footage. The mosaicked regions are, per the user, almost never
faces; §2.3 C-3 and C-4 still apply and shaped the experiment: **no prompt**, no reference, a
user-chosen model fetched from Hugging Face by the user, nothing bundled.

### Setup

`scripts/fetch_diffusion.py` (repo id → `models/diffusion/<name>/`, fp16 files only, file list
discovered from the hub, hashes recorded; the same code the application will use) and
`scripts/diffusion_probe.py`: the quality fixture's region, bicubic result as the init image,
SD1.5 + LCM-LoRA img2img, fixed seed, empty prompt, per-strength PSNR/LPIPS against the clean
frame, flicker as mean LPIPS between consecutive outputs, seconds per region.

### The first run was my probe's fault, not the model's

A 209×169 crop fed straight in came back as coloured blobs: PSNR 16.4, worse than the mosaic.
SD1.5 was trained at 512 and does not work far below it. And with four LCM steps, strengths 0.5
and 0.7 both rounded to two steps and produced *identical* output — a tell that the strength
axis was not being sampled at all. Both fixed: the crop is upscaled by an integer factor to about
512 on the long side and brought back after; eight steps so 0.2/0.35/0.5 are distinct.

### Measured, fairly

| | PSNR | LPIPS | flicker | s/region |
| --- | ---: | ---: | ---: | ---: |
| input (mosaic) | 26.69 | 0.444 | 0.032 | — |
| bicubic (Fast) | **29.79** | 0.161 | **0.018** | 0.001 |
| **diffusion s=0.2** | 28.12 | **0.092** | 0.029 | 0.13 |
| diffusion s=0.3 | 27.49 | 0.102 | 0.035 | 0.17 |
| diffusion s=0.5 | 25.13 | 0.157 | 0.061 | 0.25 |
| real motion alone (clean, consecutive) | | | 0.054 | |

- **s=0.2 is the first restorer in this project to beat bicubic on LPIPS**, by 43%, at the PSNR
  cost an inventing restorer is expected to pay (−1.7 dB) and with flicker below what real motion
  costs on its own. To the eye it is bicubic without the plastic smoothness.
- s=0.3 begins to grow shapes that are not there; s=0.5 is invention with a vignette.
- A generic prompt (`"photo"`) changes nothing (0.0917 vs 0.0920), so the prompt stays empty —
  which is also how C-4 is satisfied by construction.
- 130 ms per region on the 3080 Ti: at one region a frame the Fast path's 4.25 fps becomes about
  2.9. VRAM 2.2 GB on top of the detector's 3.

### What this does not say

One synthetic clip of blurry CG, twenty-four frames. The gain is perceptual, modest to the eye,
and the content is exactly the kind a generative prior should struggle with — so a photographic
source may do much better or much worse. D-36's limitation applies. The probe also composites
with a hard mask; the worker's feather would remove the elliptical edge visible in the strip.

### Decision

Worth a Phase 1: the diffusion step as a **refiner on top of Fast** — bicubic init, user-chosen
model and optional LoRA/embeddings from the store, strength as the one exposed knob (default
0.2), no prompt field — rather than a fourth preset, because that is what it is. The model
manager (a Hugging Face repo id in settings, fetched by the application with progress) is the
larger part of that work and is not started until the user says so.
