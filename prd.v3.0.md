# Product Requirements Document (PRD)

## Project Name: DeMosaic Studio

**Code Name:** Dynamic Mosaic Auto-Detection & Multi-Frame Restoration
**Target Platform:** Windows 10/11 x64
**Primary Acceleration:** NVIDIA CUDA (PyTorch), NVDEC / NVENC, x265 (CPU) for quality output
**Fallback Acceleration:** CPU (correctness / debug). No DirectML, no ONNX/TensorRT — see D-09, D-11
**Document Status:** Implementation-Ready Draft
**Version:** 3.0 — **personal-use scope confirmed** (supersedes 2.0)
**Distribution:** **Personal use only. This software is not distributed to anyone.** See §2.4 and D-11.
**Version history:** `prd.v1.0-original.md` (original), `prd.v2.0.md` (implementation-ready, distribution-agnostic)
**Last Updated:** 2026-08-22

---

## How to read this document

| If you are... | Read first |
| --- | --- |
| Deciding whether to fund/build this | §0 (review verdict), §1.4 (feasibility envelope), §18 (risk register), §20 (open questions) |
| About to write code | §0.3 (decision register), §3 (architecture), §4.3 (repo layout), §8 (worker protocol), §10 (error codes) |
| Training models | §1.4, §11 (dataset), §12 (evaluation harness), §16 Phase 1-2 |
| Testing / QA | §13 (test strategy), §21 (edge cases), §17 (MVP checklist) |

Requirement language follows RFC 2119: **shall** = mandatory, **should** = strong default that
requires a written justification to deviate from, **may** = optional.

Every functional requirement in §5 carries an **AC** (acceptance criterion) and a **test id**.
A requirement without a falsifiable AC is not a requirement — it is a wish, and v1.0 had many.

---

## Revision Note — what v3.0 changes and why

v2.0 was written without knowing whether the product would be distributed, so it had to assume the
strictest case. **The owner has now confirmed personal use only, with no distribution of any kind**
(D-11). That closes §20 Q2 and changes the plan in three places — and, importantly, does **not**
change it in a fourth place that matters more than the other three.

**What the decision unlocks:**

1. **GPL FFmpeg with x264/x265** (D-12). This is the largest *quality* gain in v3.0 and the one that
   is easiest to overlook. NVENC HEVC is noticeably worse than x265 at slow presets for the same
   bitrate, and §5.1.8's problem — a full re-encode degrading the 85%+ of the picture that was never
   mosaicked — *is* an encoder-quality problem. See §5.1.4 and §5.1.8.
2. **Third-party restoration weights as a training initialization** (D-04 revised). Not as a shipped
   model, and not used as-is: BasicVSR++ is trained to invert bicubic downsampling and pixelation is
   a different forward operator. But its low-level features and its flow-guided alignment module are
   largely degradation-agnostic and should transfer, which is a materially better starting point than
   training from scratch. **How much better is an expectation, not a measurement** — §16 Phase 2 now
   requires it to be measured against a from-scratch run under the §13.5 noise-floor protocol.
3. **Phases 5 and 6 shrink to almost nothing** (§16). No installer, no embedded Python runtime, no
   ONNX/TensorRT export, no per-backend numerical-parity gate, no DirectML op-support gate, no
   clean-room install test, no cross-GPU acceptance matrix. This is the largest *schedule* gain in
   v3.0 — larger than the model licensing — and it removes the project's single most likely
   integration failure (a deformable-convolution TensorRT plugin, §5.7).

**What it deliberately does not change:**

4. **The detector stays clean-room U-Net** (D-03 unchanged). AGPL is now legally available, but the
   quality case for Ultralytics YOLO-seg was always weak here: there are no pretrained mosaic weights
   either way, so the gain is a mature training loop rather than accuracy, and prototype-coefficient
   masks are *coarser* at the boundary than a full-resolution U-Net decoder — which is the opposite
   of what §5.11 blending needs. Taking the stickiest license in the ecosystem into the centre of the
   product to buy an uncertain benefit is a bad trade even when it is free.

**What it changes not at all:**

5. **§1.4's feasibility ceiling is physics, not licensing.** An object-anchored grid destroys the
   phase diversity that multi-frame restoration runs on, and a block size above ~24 px destroys the
   information outright. No choice of weights recovers what is not in the frames. **The Phase 0 gate
   is exactly as mandatory in v3.0 as it was in v2.0.** Likewise the domain gap (R-03) and the
   false-positive requirements (§5.2.5) are unchanged: the training data is still synthetic,
   recompressed, and ours.

**The cost being accepted (D-11):** AGPL and non-commercial-licensed weights are a **one-way door**.
Fine-tuned derivatives inherit their source license, so the model produced in Phase 2 becomes a
non-distributable asset. If the distribution decision ever changes — including giving a build to one
person, free — the project must either open-source in full, buy a commercial license, or retrain from
clean-room components. §5.9.1's backend interface is what keeps that last option affordable, and
§18 R-05 now tracks the door rather than the license.

---

## Revision Note — what v2.0 changed and why (retained)

v1.0 was a good *statement of intent*. It was not implementable: an engineer handed v1.0 could not
have created the first file of the project without inventing a dozen decisions the document
deliberately left open ("may", "candidate", "or equivalent", "where appropriate"). Those open
choices are not neutral — several of them determine whether the product can ship at all.

v2.0 keeps v1.0's scope and principles and closes the gaps. The substantive changes:

1. **Architecture changed from `WPF + C++20 native engine over a C ABI` to `WPF + out-of-process
   Python worker over a JSON Lines protocol`, with the native engine deferred to Phase 5** (D-01).
   Rationale, cost, and the reversal path are in §3.4. This is the single biggest change in v2.0 and
   the one most worth arguing with; §20 Q1 tells you how to override it in one line.
2. **A feasibility envelope with kill criteria** (§1.4). v1.0 promised "multi-frame restoration"
   without stating when multi-frame *can* work. It can only work when neighbouring frames observe
   the hidden signal at different mosaic-grid phases. That condition is measurable, is often absent,
   and if Phase 0 shows it is absent on representative material, the product's stated differentiator
   (§19) does not exist and the project must re-scope rather than proceed.
3. **A license audit** (§4.2). Two of v1.0's named candidate technologies — Ultralytics YOLO
   (AGPL-3.0) and BasicVSR++ (S-Lab non-commercial license) — cannot legally ship inside a
   closed-source Windows application. v1.0 named both on the front page of the tech stack.
4. **A complete host-to-engine protocol** (§8), a **job/checkpoint schema with per-artifact
   fingerprint invalidation** (§9), and a **numbered error-code table** (§10). These three artifacts
   are what let UI work and engine work proceed in parallel; without them the two halves cannot be
   integrated and cannot be tested independently.
5. **Concrete dataset and evaluation specifications** (§11, §12), including a **hard-negative
   corpus** — v1.0 never mentioned false positives, and defocus/bokeh/low-bitrate blocking is the
   dominant real-world false-positive source for any mosaic detector.
6. **A test strategy** (§13) including the rule that *GPU output is non-deterministic and any A/B
   claim must first measure its own noise floor.*
7. **Reference hardware corrected** from RTX 4070 to the actual development machine
   (RTX 3080 Ti 12 GB), and performance targets restated as two tiers — MVP (Python engine) and
   Optimized (native engine) — because a single number that no build can meet is not a target.
8. **Lawful-use and content constraints** (§2.3), which a tool of this class needs stated in its
   requirements rather than discovered at release review.

---

## 0. Review Verdict and Gap Register

### 0.1 Verdict

**v1.0 is not sufficient to implement.** It is an accurate description of a *correct* pipeline and a
genuinely good list of engineering principles — v1.0's §17 and §19 are the strongest parts of the
document and are kept nearly unchanged here. But it specifies the *shape* of the system and almost
none of its *substance*.

Concretely: v1.0 contains 0 API signatures, 0 file formats, 0 pinned versions, 0 licenses, 0 named
model weights, 0 data sources, 0 repository paths, 0 build commands, and 0 falsifiable acceptance
criteria that reference a defined artifact. Its numeric targets (`mAP50 >= 0.92`, `>= 15 FPS`) are
stated against "the agreed representative test dataset", which does not exist and is not defined, so
they can neither pass nor fail.

The most dangerous property of v1.0 is not what it omits but what it *assumes without testing*: that
useful information survives in neighbouring frames. The entire architecture (v1.0 §18/§19) is built
on that assumption, and the document never asks under what conditions it holds. §1.4 of this
revision makes measuring it the first thing the project does.

### 0.2 Gap register

Each gap is closed by the section named in the last column.

| # | Gap in v1.0 | Why it blocks implementation | Closed by |
| --- | --- | --- | --- |
| G-01 | No feasibility bound on multi-frame recovery | The core premise may be false for most real input, and nothing in the plan detects that before Phase 3 | §1.4, §16 Phase 0 gate |
| G-02 | "Stable C ABI + P/Invoke" with no header, ownership, threading, or error contract | UI and engine cannot be built or tested in parallel; integration becomes a big-bang merge | §3.5, §8 |
| G-03 | No named detector model, weights, or training data source | "Segmentation model, YOLO-family candidate" is not a buildable instruction, and no public mosaic-segmentation weights exist | §4.2, §5.2, §11 |
| G-04 | No license audit | Ultralytics YOLO is AGPL-3.0; BasicVSR++ is S-Lab non-commercial. Both were named as primary candidates. **v3.0: obligations do not trigger under D-11, but the audit still governs what may be used and records the one-way door** | §4.2, §18 R-05 |
| G-05 | Restoration model unspecified, and the named candidate is trained for the wrong degradation | BasicVSR++ is trained for bicubic downsampling, not pixelation; adopting it as-is will underperform and mislead Phase 2 | §5.9, §16 Phase 2 |
| G-06 | No false-positive requirement or negative corpus | Detectors of this kind fire on bokeh, defocus, and low-bitrate blocking; a high-recall detector that mangles clean footage is worse than no product | §5.2.5, §11.4 |
| G-07 | `JobState` sketched with no schema, path, versioning, or invalidation rule | Resume that silently reuses artifacts produced under different settings is a data-corruption bug, not a UX bug | §9 |
| G-08 | Error categories listed but not enumerated, numbered, or classified recoverable/fatal | The retry policy, the UI strings, and the parity tests cannot be written | §10 |
| G-09 | No repository layout, build system, toolchain versions, or CI | No first commit is possible, and the dev machine currently has none of the required native toolchain (§4.5) | §4.3, §4.4, §4.5 |
| G-10 | No test strategy, no fixtures, no GPU-free CI subset, no determinism policy | GPU non-determinism makes every quality claim unfalsifiable without a measured noise floor | §13 |
| G-11 | Acceptance numbers reference an undefined dataset | Targets cannot pass or fail, so phase gates are unenforceable | §11.5, §12 |
| G-12 | No model packaging, versioning, or verification plan | Weights are hundreds of MB to GB, and without versioning a retrained model silently mixes with older output | §14. **v3.0: distribution parts dropped; provenance parts retained** |
| G-13 | VFR handling stated as a goal with no output-timing rule | "PTS-based" does not say what PTS the *output* frames get, which is where A/V drift actually originates | §5.1.7 |
| G-14 | Whole-frame re-encode degrades the 85%+ of the picture that has no mosaic | Users compare output to source; a globally softer picture reads as "the tool ruined my video" | §5.1.8 |
| G-15 | DirectML fallback promised for models that may use custom or deformable ops | The fallback may not run the model at all; promising it creates a support burden | §5.17. **v3.0: closed outright by D-11 — DirectML dropped** |
| G-16 | No preview-path specification | Preview is a second consumer of the engine with different latency requirements; unspecified means it gets bolted on | §5.16.3, §8.6 |
| G-17 | Progress/callback contract unspecified | Out-of-order and post-completion progress callbacks are a known failure mode in this exact application shape | §8.4 |
| G-18 | No lawful-use or content constraints | A restoration tool of this class needs its use boundaries in requirements, not in a release-review surprise | §2.3 |
| G-19 | Milestones have durations but no dependencies, entry/exit criteria, or kill criteria | Phases cannot gate each other, so a failed premise propagates to Phase 5 | §16 |
| G-20 | No risk register | The project's three existential risks (feasibility, licensing, detector domain gap) are unowned | §18 |

### 0.3 Decision register

These are the decisions v1.0 left open. Each is now **decided** so implementation can begin. Each
records its reversal cost, because a decision that is cheap to undo should not be re-litigated and
one that is expensive to undo should be argued about now.

| ID | Decision | Alternative rejected | Reversal cost |
| --- | --- | --- | --- |
| D-01 | Engine runs **out-of-process in Python**, host-to-engine over stdio JSON Lines (§8). Native C++ engine deferred to Phase 5 behind the same protocol | C++20 core + C ABI + P/Invoke from day one (v1.0) | **Low.** The boundary is a process boundary plus a wire protocol; a native engine speaking the same protocol is a drop-in replacement. That is why this boundary was chosen |
| D-02 | **.NET 10** (LTS) for the WPF host | .NET 8 (v1.0) | Trivial. Chosen because SDK 10.0.302 is the only SDK on the dev machine and the team's other WPF product already targets it |
| D-03 | Detector = **binary semantic segmentation**, U-Net decoder over a `timm` encoder (Apache-2.0), trained in-house. **Unchanged in v3.0** | Ultralytics YOLOv8-seg. Now legally available under D-11 and still rejected: no pretrained mosaic weights exist either way, so the gain is a training loop rather than accuracy, and prototype-coefficient masks are coarser at the boundary than a full-resolution decoder (§5.11 needs the opposite) | Medium. An encoder/decoder swap is a training run, not an architecture change |
| D-04 | Restoration = **in-house recurrent degradation-conditioned model** (§5.9.2), **initialized from third-party VSR pretrained weights where that measurably helps** (§16 Phase 2), trained on the synthetic generator's degradation | (a) Training from scratch — slower start, no measured benefit either way until Phase 2 says so. (b) Using third-party weights as-is — wrong degradation prior, produces confidently wrong texture | High. This is most of Phase 2. Initialization is cheap to abandon; the S-Lab license inherited by the fine-tuned derivative is not (§18 R-05) |
| D-05 | Training data = **synthetic degradation over licensed clean video**; no real mosaicked material is required for training | Collecting and pairing real mosaicked material | Low; the generator is needed regardless |
| D-06 | Checkpoint state is **JSON on disk in a per-job directory**, with **per-artifact settings fingerprints** (§9.3) | One monolithic fingerprint, or a database | Low |
| D-07 | Output default is **full re-encode** at a quality-preserving rate target, with **stream-copy pass-through** when a job detects nothing; smart-cut segment splicing deferred (§5.1.8) | Always smart-cut | Low |
| D-08 | MVP decode/encode goes through **FFmpeg** (PyAV bindings), hardware decode where available, full frames on CPU | Fully GPU-resident pipeline from day one | Medium; it is Phase 5's main work item |
| D-09 | Inference runtime is **PyTorch + CUDA, permanently**. ONNX/TensorRT export is **dropped** under D-11: with no second machine to deploy to, the export toolchain, the per-backend numerical-parity gate, and the deformable-conv TensorRT plugin are all work with no beneficiary | TensorRT for throughput | Low. Re-adding export is a Phase 5 item if a real need appears; nothing in the design forbids it |
| D-10 | Protocol, error codes, and job schema are **versioned and change-controlled** (§8.1, §10.1) | Ad-hoc evolution | n/a — process, not code |
| **D-11** | **Personal use only. The software, its models, and its derivatives are not distributed, published, hosted, or given to anyone.** AGPL-3.0 and non-commercial obligations therefore do not trigger, and installer/packaging/portability work leaves scope | Distributable product (v2.0's assumption) | **High and asymmetric.** Reversing this after Phase 2 means open-sourcing in full, buying commercial licenses, or retraining from clean-room components. §5.9.1's interface keeps the third option affordable. See §2.4 and §18 R-05 |
| **D-12** | Encoder is **GPL FFmpeg with x264/x265 enabled**, x265 the default for quality-preserving output; NVENC retained as the speed option | LGPL FFmpeg + NVENC only (v2.0, forced by the distribution assumption) | Trivial — a build-flag change |

### 0.4 What v2.0 deliberately does not decide

Listed so nobody mistakes silence for an omission. §20 carries the questions these raise.

- The exact restoration network topology (channel counts, block counts, form of recurrence). That is
  a Phase 2 experimental outcome, not a document decision. §5.9 fixes only its **interface** and its
  **acceptance criteria**.
- Commercial and distribution model, pricing, telemetry, update channel.
- Localization scope beyond the string-externalization requirement in §5.16.9.

---

## 1. Executive Summary

### 1.1 Background

Existing mosaic-restoration tools require manual ROI annotation or rely on unstable frame-by-frame
detection. When the mosaic's position, shape, block size, or target changes over time, those
approaches produce ROI jitter, boundary artifacts, temporal flicker, and inconsistent
reconstruction.

DeMosaic Studio is a standalone Windows desktop application that automatically detects and tracks
partially mosaicked regions in video, analyzes each region's degradation characteristics, gathers
evidence from neighbouring frames, performs temporally aligned multi-frame restoration, and exports
a visually consistent reconstructed video. All processing is local; no cloud inference.

### 1.2 Product Goal

The product shall provide:

- Automatic mosaic detection and segmentation without manual annotation.
- Stable multi-object tracking for dynamically moving or resizing mosaic regions.
- Mosaic degradation analysis including block size, grid offset, **grid anchoring** (§5.4.4), and
  degradation type.
- Adaptive multi-frame temporal reconstruction using neighbouring frames.
- Best-effort restoration when original information cannot be recovered, **explicitly labelled as
  such**.
- Temporal consistency with minimal flicker and ROI boundary artifacts.
- Hardware-accelerated decode, inference, and encode on NVIDIA GPUs.
- Preservation of audio, timestamps, subtitles, and container metadata.
- Reliable processing of long videos with cancellation, checkpoint, and resume.

### 1.3 Restoration Semantics

DeMosaic Studio performs **best-effort visual restoration**, not recovery of the original hidden
content.

Where useful source information exists across neighbouring frames, the engine shall prioritize
observable temporal evidence. Where information has been irreversibly destroyed in all available
frames, the restoration model estimates visually plausible content. Estimated content shall never be
represented — in the UI, in logs, in metadata, or in exported files — as verified recovery of the
original pixels.

The engine shall compute a **restoration confidence map** (§7.4) indicating how strongly each
restored region is supported by observable temporal evidence, and the UI shall surface the per-job
aggregate (§5.16.4). Confidence is a *diagnostic*, never evidence.

**AC-1.3** — For any exported file, `demosaic_studio.confidence_mean` and
`demosaic_studio.synthetic=true` are written to container metadata (§5.1.6), and the job summary
displays the mean confidence bucket. Test: `T-EXPORT-META-01`.

### 1.4 Feasibility Envelope and Kill Criteria — READ BEFORE BUILDING

This section did not exist in v1.0 and is the most important addition in v2.0.

#### 1.4.1 What multi-frame restoration can and cannot do

Pixelation replaces every `B x B` block with its mean. That is a linear, information-destroying
measurement: within a block, all detail finer than `B` is gone from that frame. Multi-frame
restoration recovers *real* detail only when **different frames sample the hidden signal at
different grid phases**, so that the set of frames forms an over-determined linear system in the
underlying signal. Whether that happens depends on a property v1.0 never names:

| Grid anchoring | What happens as the subject moves | Multi-frame value |
| --- | --- | --- |
| **Screen-anchored** — grid fixed to frame coordinates (a fixed censor overlay, or a filter applied to the composed frame) | The subject slides across a stationary grid, so each frame averages a *different* set of subject pixels | **High.** Genuine sub-block information is recoverable |
| **Object-anchored** — grid moves with the tracked object (an automated tracking censor) | The same subject pixels land in the same block every frame | **Near zero.** Averaging frames reduces noise only; there is no new detail to recover |
| **Unknown / mixed** | — | Must be measured, never assumed |

The engine shall estimate grid anchoring per track (§5.4.4) and the strategy router shall consume it
(§5.8). A pipeline that runs multi-frame restoration on an object-anchored mosaic spends the entire
compute budget producing a model hallucination while reporting high effort.

#### 1.4.2 Expected recoverability by block size

Approximate bands at 1080p, assuming a screen-anchored grid and at least 1 px/frame subject motion.
These are **hypotheses to be measured in Phase 0**, not claims.

| Block size `B` | Character of the problem | Expected outcome |
| --- | --- | --- |
| <= 4 px | Deblocking plus mild super-resolution | High-confidence restoration; most output pixels evidence-backed |
| 5-12 px | Genuine multi-frame reconstruction; `K` frames at distinct phases materially over-determine the system | The product's target band. Partially evidence-backed |
| 13-24 px | Prior-dominated; temporal evidence constrains coarse structure only | Plausible, largely fabricated. Must report Low confidence |
| > 24 px | Information destroyed | Fabrication. Report as such; consider refusing to claim restoration at all |

Two further destroyers of sub-block residue are present in real material and are therefore mandatory
in the evaluation set (§11.3): **re-encoding after mosaic application** (quantization erases the tiny
inter-block variations that multi-frame solving depends on) and **chroma subsampling** (4:2:0 halves
chroma resolution before the mosaic is even applied).

#### 1.4.3 Phase 0 kill criteria

Phase 0 (§16) shall produce a measured answer to one question: *does multi-frame beat single-frame on
representative material?* On the frozen evaluation set `eval-v1` (§11.5), restricted to the
screen-anchored `B` in `[6, 12]` band with H.264 recompression across the dataset's CRF ladder:

- **PASS** — multi-frame (`K=5`) exceeds the single-frame baseline by **>= 1.0 dB PSNR** *and*
  **>= 0.03 LPIPS** *and* shows lower warping error, all measured under the noise-floor protocol of
  §13.5.
- **MARGINAL** — the improvement is real but below those thresholds. Proceed only with the temporal
  window reduced to `K` in `{3,5}` and the Phase 2 budget re-planned; record the decision in
  `docs/DECISIONS.md`.
- **KILL** — no measurable improvement, or improvement inside the noise floor. The product's stated
  differentiator (§19) does not exist for this material. **Do not proceed to Phase 2 as specified.**
  Re-scope to single-frame restoration with honest confidence labelling, or to a detection-and-
  masking product, and revise this PRD.

**AC-1.4** — `scripts/eval_multiframe_gate.py` emits a machine-readable verdict
(`PASS` / `MARGINAL` / `KILL`) with the measured deltas and the measured noise floor, and the result
is committed to `docs/phase0-report.md`. Test: `T-GATE-P0-01`.

The cost of skipping this section is the entire Phase 2-5 budget spent on an architecture whose
premise was never checked. v1.0 would have spent it.
---

## 2. Scope

### 2.1 In Scope

- Partially mosaicked video restoration.
- Multiple simultaneous mosaic regions.
- Dynamically moving and resizing mosaic regions.
- Pixelation, blur, and mixed degradation.
- Multi-frame temporal reconstruction with single-frame fallback.
- Local Windows processing on one machine (§4.5), NVIDIA CUDA acceleration, CPU correctness path.
- Hardware video decoding and encoding where supported.
- Audio, subtitle, chapter, and metadata preservation.
- Long-running jobs with checkpoint, resume, pause, and cancel.
- Before/after preview and diagnostic overlays.

### 2.2 Out of Scope for the Initial MVP

- Cloud inference and distributed rendering.
- Guaranteed forensic recovery of destroyed information.
- Manual frame-by-frame painting tools.
- Full professional NLE functionality.
- Model training inside the desktop application (training lives in `training/`, §4.3).
- Batch/folder queue processing. *(Deliberate: it is a straightforward addition once the single-job
  pipeline is stable, and adding it early doubles the state space the scheduler must handle.)*

### 2.3 Lawful Use and Content Constraints

New in v2.0. These are product requirements, not legal boilerplate, and they have implementation
consequences.

- **C-1** The application is for restoring video the user owns or is otherwise authorized to
  process. Under D-11 there is no installer, so this is stated on the first-run screen and in
  `README.md`.
- **C-2** No model built for this product shall be trained on, or fine-tuned toward reconstructing,
  identifiable real individuals. Training data is licensed clean video plus the synthetic degradation
  generator (§11); the degradation is what the model learns to invert, not any person.
- **C-3** Reconstructed content is synthetic. The product shall not describe output as recovered,
  original, uncensored-original, or evidentiary, in UI strings, metadata, filenames, or marketing
  copy. Exported files carry `demosaic_studio.synthetic=true` (§5.1.6, AC-1.3).
- **C-4** No identity-directed features. Specifically out of scope, permanently: face swapping,
  identity insertion or transfer, likeness matching against a person database, "restore to look like
  <named person>" conditioning.
- **C-5** No feature whose purpose is to defeat content-protection or platform-integrity measures.
  Input is a decoded video file the user already possesses; the product performs no DRM circumvention.
- **C-6** Logs shall carry diagnostic context without embedding user media content: no frame
  thumbnails, no pixel dumps, no full source paths at INFO level (§10.4).

**AC-2.3** — A checklist item verifies C-1 through C-6 against the built application and its string
tables. Test: `T-POLICY-STRINGS-01` scans `Strings.resx` for the banned vocabulary of C-3.

Note that D-11 does **not** relax any of C-1 through C-6. Personal use removes *license* obligations;
it removes none of the constraints above, which exist for reasons unrelated to who receives a binary.

### 2.4 Distribution Scope (D-11) — new in v3.0

**This software is built for the owner's personal use and is not distributed.** No binaries, no
source, no models, no derived weights, no hosted or network-accessible instance, to anyone — free or
paid, including a single copy to one other person.

Consequences that the rest of this document depends on:

| Area | Effect |
| --- | --- |
| Licensing | AGPL-3.0 obligations (conveying, §13 network use) and non-commercial restrictions do not trigger. §4.2 remains in force as an *engineering and reversibility* control, not a legal one |
| Encoder | GPL FFmpeg with x264/x265 becomes available (D-12), which is the largest quality gain in v3.0 (§5.1.4, §5.1.8) |
| Restoration weights | Third-party pretrained weights may be used as a training **initialization** (D-04). The resulting fine-tuned model inherits their license and is likewise non-distributable |
| Packaging | No installer, no embedded Python runtime, no code signing, no clean-room install test (§14, §16 Phase 6) |
| Portability | No ONNX/TensorRT export, no DirectML gate, no cross-GPU acceptance matrix. One target machine: §4.5 (D-09, §5.17, §22) |
| Everything else | Unchanged. Notably §1.4's feasibility ceiling, §5.2.5's false-positive requirements, and §2.3's content constraints |

**The reversal trigger.** If the intent to distribute ever returns — including "just send it to a
friend" — **stop and re-plan before writing more code.** After Phase 2 the choices narrow to
open-sourcing the whole application, buying commercial licenses, or retraining the restoration model
from clean-room components. Keeping third-party-derived weights behind `IRestorationBackend`
(§5.9.1) is what keeps the third option from being a rewrite. Tracked as §18 R-05.

---

## 3. High-Level Architecture

### 3.1 Process topology

```text
+---------------------------------------------------------------+
|  DeMosaicStudio.App      (C# / .NET 10 / WPF)                  |
|  - Drag & drop, media metadata, job list                       |
|  - Preview (original / restored / split), diagnostic overlays  |
|  - Progress, ETA, hardware & quality settings                  |
|  - Job control: start / pause / resume / cancel / retry        |
+------------------------------+--------------------------------+
                               |
        stdio JSON Lines, protocol v1.0  (§8)   [D-01]
        child process, one worker per host, one job at a time
                               |
+------------------------------v--------------------------------+
|  demosaic_worker             (Python 3.12, embedded runtime)   |
|                                                                |
|  media.reader   -- FFmpeg/PyAV demux + decode, PTS-exact       |
|       |                                                        |
|  scene.cut      -- histogram + flow discontinuity              |
|       |                                                        |
|  detect.segment -- binary mosaic mask  (U-Net/timm, §5.2)      |
|       |                                                        |
|  track.bytetrack -- persistent Track IDs + Kalman (§5.3)       |
|       |                                                        |
|  analyze.profile -- block size, phase, ANCHORING, type (§5.4)  |
|       |                                                        |
|  roi.stabilize  -- adaptive padding, alignment (§5.5)          |
|       |                                                        |
|  window.builder -- adaptive K, scene-cut truncation (§5.6)     |
|       |                                                        |
|  restore.router -- multi-frame / single-frame / passthrough    |
|       |             (§5.8; consumes grid anchoring)            |
|  restore.backend -- IRestorationBackend impls (§5.9)           |
|       |                                                        |
|  post.temporal  -- flicker suppression (§5.10)                 |
|       |                                                        |
|  post.blend     -- mask-aware compositing (§5.11)              |
|       |                                                        |
|  media.writer   -- NVENC/software encode, mux A/V/subs (§5.1)  |
+---------------------------------------------------------------+
```

### 3.2 Layer rules

- The host **never** performs image processing, model inference, or media I/O. Its only media
  responsibility is displaying frames the worker hands it (§8.6).
- The worker **never** performs UI decisions. It reports facts and errors; the host decides what the
  user sees and what is retried.
- Everything crossing the boundary is defined in §8. If it is not in §8, it does not cross.
- Domain logic that both sides need (error-code semantics, settings fingerprint rules,
  restoration-confidence buckets) is specified once here and implemented twice, guarded by parity
  tests (§13.4).

### 3.3 Threading and cancellation model

- Worker: one asyncio/thread pipeline of **bounded** queues with back-pressure. Stage boundaries are
  the queues in §3.1. Queue depth is configurable per stage; defaults in §5.13.
- Host: all engine interaction is async. Progress is delivered to the UI through an **inline,
  ordered, gated** progress channel — not a free-threaded callback. A progress report that arrives
  after the job has reached a terminal state is **dropped**, never applied. (This exact bug —
  post-completion progress resurrecting a finished job into "processing 65%" — is a known failure
  mode of this application shape.)
- Cancellation is cooperative and one-directional: host sends `cancel`, worker acknowledges, drains,
  writes a checkpoint, and exits with a terminal `result` carrying `status="cancelled"`. The host
  does not kill the process except after the grace period in §8.5.

**AC-3.3** — A test drives 10,000 out-of-order and post-terminal progress messages into the host's
progress channel and asserts the job's observable state never moves backwards and never leaves a
terminal state. Test: `T-HOST-PROGRESS-ORDER-01`.

### 3.4 Why the MVP engine is Python, not C++ (D-01)

v1.0 specified a C++20 native engine behind a C ABI. That is the right *end state* and the wrong
*starting point*. The argument, stated plainly so it can be argued with:

**For deferring native:**

1. Every candidate model, every training loop, every metric (LPIPS, warping error), and every
   ablation lives in the PyTorch ecosystem. Phases 0-2 are almost entirely research, and they are
   research in Python whichever engine ships. Writing the C++ engine first means writing it against
   a model that does not exist yet.
2. The three hardest requirements in this document — the feasibility gate (§1.4), the detector's
   domain gap (§18 R-03), and temporal consistency (§5.10) — are all *model* problems. None of them
   gets easier with a native engine, and all of them get slower to iterate on.
3. Temporal alignment (§5.7) is the piece most likely to need a custom op. In PyTorch it is an
   afternoon; in TensorRT it is a plugin, a serialization format, and a per-GPU engine cache.
4. The dev machine has **no** native toolchain installed today (§4.5): no Visual Studio C++ tools, no
   CUDA Toolkit, no CMake. Phase 0 would begin with a multi-hour, ~20 GB toolchain bootstrap before
   the first line of product code.
5. The team has already shipped this exact topology — WPF host, embedded Python child process, stdio
   JSON Lines protocol, checkpoint/resume, model download and verification — in another product. The
   protocol pitfalls (progress ordering, CUDA support-library loading on Windows, settings-fingerprint
   invalidation) are known and are folded into §8, §9, and §14 of this document.

**Against (the real costs, accepted):**

1. Full-frame decode and encode go through CPU memory in the MVP, so end-to-end throughput will be
   materially below the native target. §6.1 therefore states two tiers rather than pretending one.
2. Python adds per-frame overhead in the scheduler. Mitigated by keeping per-frame Python work to
   orchestration only, with all pixel work in torch/NumPy/FFmpeg.
3. ~~An embedded Python runtime enlarges the installer and complicates packaging.~~ **Removed in v3.0:** D-11 drops packaging entirely, so this cost no longer exists (§14.4).

**Reversal path (why this is a Low-cost decision):** the boundary is a process boundary and a wire
protocol, not a language binding. A native engine that speaks protocol v1.x is a drop-in replacement
for the Python worker, selectable per-install. §5.13's scheduler requirements and §8's protocol are
written to be implementable in either language, and §13.4's parity tests already assume two
implementations of the shared semantics.

**AC-3.4** — The host resolves its worker through a single indirection (`IEngineLauncher`) with two
implementations (`PythonWorkerLauncher`, `NativeWorkerLauncher`), and a fake in-process launcher for
tests. Swapping engines is configuration, not code. Test: `T-HOST-LAUNCHER-SWAP-01`.

### 3.5 Engine boundary contract (replaces v1.0's unspecified C ABI)

The contract is §8 in full. Summary of its guarantees:

| Property | Guarantee |
| --- | --- |
| Transport | UTF-8 JSON Lines on the child's stdin/stdout. stderr is unstructured log text, captured but never parsed for control flow |
| Framing | Exactly one JSON object per line, no embedded newlines, `\n` terminated |
| Versioning | Handshake exchanges `protocolVersion`; host refuses a worker whose major version differs (§8.1) |
| Ordering | All messages for a job are ordered; the host may drop but never reorder |
| Backpressure | Worker throttles `progress` to at most 4/s per job and coalesces |
| Errors | Every failure surfaces as a numbered code from §10, never as a free-text string alone |
| Lifetime | One worker process per host process, restartable; a crashed worker fails the current job only |
| Large payloads | Frames and masks are never inlined in JSON; they are written to the job's temp directory and referenced by path (§8.6) |

---

## 4. Technology Stack, Licensing, and Project Setup

### 4.1 Pinned stack

Versions are pinned per release, not floating. Update in `Directory.Packages.props` and
`worker/requirements.lock` only, never per-project.

| Area | Technology | Pinned for v0.1 | Note |
| --- | --- | --- | --- |
| Desktop UI | C# / .NET 10 / WPF | SDK 10.0.3xx, `net10.0-windows` | D-02 |
| Host DI/logging | Microsoft.Extensions.* , Serilog | pin in `Directory.Packages.props` | |
| Engine | Python | 3.12.x, local venv (§14.4) | D-11 |
| Tensor runtime | PyTorch + CUDA | torch 2.x + cu12x wheel | D-09 |
| Encoder/decoder | FFmpeg via PyAV | FFmpeg 7.x **GPL build, x264 + x265 enabled** | D-12, §5.1.4 |
| Detection | U-Net decoder + `timm` encoder | timm 1.x | D-03 |
| Tracking | ByteTrack algorithm, re-implemented | n/a (algorithm, not package) | §4.2 |
| Optical flow | RAFT-small or PWC-lite, license-checked | Phase 2 decision | §5.7 |
| Metrics | `torchmetrics`, `lpips` | Phase 1 | §12 |
| Export / portability | — | **dropped** (D-09, D-11) | §5.17 |
| Installer | — | **dropped** (D-11) | §14 |
| Native engine | C++20, CMake, vcpkg | deferred indefinitely | §3.4 |

### 4.2 License audit — reversibility control (was: blocking)

Under D-11 nothing here is a legal blocker any more. The table stays because it now answers a
different and still-important question: **if the distribution decision ever changes, what does each
dependency cost us?** The "Contaminating" column is the one to read.

| Component | License | Use under D-11 | Contaminating? |
| --- | --- | --- | --- |
| FFmpeg **GPL build, x264/x265** | GPL-2.0+ | **Adopted (D-12).** x265 slow is the default quality encoder; NVENC is the speed option | Yes — the app would become GPL if distributed. Cheap to undo: rebuild FFmpeg LGPL and lose x265 |
| **BasicVSR++ / BasicSR model-zoo weights** | S-Lab License 1.0 (non-commercial) | **Allowed as Phase 2 training initialization** (D-04) and as an internal baseline | **Yes, and stickily** — the fine-tuned derivative inherits it. This is the expensive door (§18 R-05) |
| **Ultralytics YOLOv8/v11 (incl. `-seg`)** | AGPL-3.0 | **Still not used** — rejected on engineering grounds, not legal ones (D-03) | Would be the stickiest in the stack. Avoided by choice |
| PyTorch | BSD-3 | Adopted | No |
| `timm` encoders | Apache-2.0 | Adopted. Each pretrained checkpoint carries its own license — check per checkpoint | No, if the checkpoint is permissive |
| ByteTrack (reference impl) | MIT | Algorithm reimplemented in-repo | No |
| LPIPS / evaluation weights | BSD-2 + varied backbone weights | Evaluation only | No (never in the product path) |
| RAFT / optical-flow weights | check per source before adoption | Phase 2 decision (§5.7) | Depends — prefer a permissive source, all else equal |

Two rules that survive D-11 and are worth the small effort:

- **R-4.2a** Anything with a contaminating license is used **only behind an interface** — restoration
  weights behind `IRestorationBackend` (§5.9.1), the encoder behind the media writer. Never woven
  through the codebase. This is what makes §2.4's reversal trigger a swap instead of a rewrite.
- **R-4.2b** `THIRD_PARTY_NOTICES.md` is maintained anyway, listing every dependency with its license
  and source URL. It costs minutes and it is the only artifact that makes the reversal question
  answerable later. A dependency in a lockfile without a notice entry fails the check.

**AC-4.2** — `T-LICENSE-NOTICE-COVERAGE-01` fails when a lockfile dependency has no notice entry.
`T-CONTAMINANT-ISOLATION-01` asserts that no module outside the media writer imports the encoder
directly and no module outside `restore/` imports third-party-derived model code.

### 4.3 Repository layout

```text
DeMosaicStudio/
  DeMosaicStudio.sln
  Directory.Build.props            # shared C# settings, warnings-as-errors
  Directory.Packages.props         # central package version pinning
  global.json                      # SDK pin
  prd.md                           # this document
  prd.v1.0-original.md             # preserved v1.0
  AGENTS.md                        # layer rules, coding rules, change procedures
  CLAUDE.md                        # working handover notes (what broke on real hardware and why)
  THIRD_PARTY_NOTICES.md
  docs/
    ARCHITECTURE.md
    DECISIONS.md                   # ADRs; D-01..D-10 seeded from §0.3
    WORKER_PROTOCOL.md             # §8, expanded, authoritative
    ERROR_CODES.md                 # §10, authoritative
    TROUBLESHOOTING.md
    phase0-report.md               # produced by the §1.4.3 gate
  src/
    DeMosaicStudio.Domain/         # pure: settings, fingerprints, error codes, policies. No I/O
    DeMosaicStudio.Application/    # job orchestration, engine client, progress gating
    DeMosaicStudio.Infrastructure/ # process launcher, file system, model store, persistence
    DeMosaicStudio.App/            # WPF; net10.0-windows
  worker/
    demosaic_worker/
      __init__.py                  # re-exports PROTOCOL_VERSION; never a second copy of it
      protocol.py                  # message schemas, single source of PROTOCOL_VERSION
      errors.py                    # error codes, mirrored by Domain
      main_loop.py                 # stdio dispatch
      media/ scene/ detect/ track/ analyze/ roi/ window/ restore/ post/
      hardware.py cuda_setup.py checkpoint.py
    requirements.lock
    tests/
  training/                        # not part of the app. datasets, degradation generator, train, eval
    degradation/ datasets/ models/ train_detector.py train_restorer.py
  scripts/
    setup-worker.ps1 check-environment.ps1 smoke-gpu.ps1
    eval_multiframe_gate.py eval_report.py bench.py
  tests/
    DeMosaicStudio.Domain.Tests/ DeMosaicStudio.Application.Tests/
    DeMosaicStudio.Integration.Tests/    # protocol round-trip against a fake worker
  fixtures/
    media/                         # tiny generated clips, committed
    protocol/                      # golden protocol transcripts
    parity/                        # shared C#/Python fixtures (error codes, fingerprints)
```

(No `installer/` under D-11. The application runs from its build output on the machine in §4.5.)

Rules that make the layout load-bearing rather than decorative:

- `Domain` has no project references and no I/O. Anything testable without a GPU or a file system
  belongs there, which is what keeps CI meaningful (§13.2).
- `App` (`net10.0-windows`) is **not** referenced by any test project that must run on Linux CI.
  Policy and validation logic therefore lives in `Domain`, never in code-behind.
- `PROTOCOL_VERSION` has exactly one definition (`worker/demosaic_worker/protocol.py`) and one
  mirror (`Domain`), guarded by a parity test. A second copy will drift — it always does.

### 4.4 Build, test, and CI

Local verification, all three of which shall pass before any commit:

```powershell
dotnet build DeMosaicStudio.sln -c Release   # warnings-as-errors; keep at 0
dotnet test  DeMosaicStudio.sln -c Release
python -m pytest worker/tests training/tests -q
```

- Under D-11 there is one machine and one developer, so "CI" means **a local pre-commit script that
  runs the three commands above**. Hosted CI is optional; if it is set up, run it on Linux with
  `EnableWindowsTargeting=true`, which compiles WPF without running it and catches XAML and
  binding-signature breakage.
- The **GPU-free subset** (§13.2) is still worth maintaining even with a GPU always available: it is
  what keeps the feedback loop in seconds rather than minutes, and it is what forces policy logic
  into testable places. GPU-dependent tests are marked and covered by `scripts/smoke-gpu.ps1`.
- **Baseline discipline:** platform-dependent tests will fail on one OS or the other. Record the
  known-failing baseline in `CLAUDE.md` and compare against it. Only *newly* failing tests are a
  regression.
- Scripts (`.ps1`) shall be saved as **UTF-8 with BOM**. Windows
  PowerShell 5.1 reads BOM-less files as ANSI (CP949 on this machine), which corrupts multi-byte
  characters and causes *parse* failures, not runtime failures. `pwsh` 7 reads them fine, so this
  never reproduces under a modern shell. Guard with a byte-level test over every script in the repo.

**AC-4.4** — `T-SCRIPT-ENCODING-01` byte-inspects every `.ps1`/`.iss` in the repo for a UTF-8 BOM and
fails on any new script that lacks one. (Applies to `.iss` too, should an installer ever return.)

### 4.5 Development machine baseline (measured 2026-08-22)

Recorded because it changes Phase 0's first task list, and because v1.0's reference hardware was not
the hardware this will be built on.

| Item | State |
| --- | --- |
| GPU | NVIDIA RTX 3080 Ti, 12 GB, driver 591.86 |
| .NET SDK | **10.0.302 present.** No .NET 8 SDK → v1.0's `.NET 8` target could not build here (D-02) |
| Visual Studio / MSVC C++ | **absent** |
| CUDA Toolkit / `nvcc` | **absent** |
| CMake | **absent** |
| Python on PATH | **absent** |
| FFmpeg on PATH | **absent** |
| Repo | **not a git repository yet** — Phase 0 task 0.1 |

Phase 0 therefore begins with `git init`, the Python embedded runtime, and FFmpeg, and — under D-01 —
does **not** need the ~20 GB MSVC + CUDA Toolkit bootstrap until Phase 5.

**AC-4.5** — `scripts/check-environment.ps1` reports every item above with PASS/FAIL and the exact
remediation command, and is the first thing a new machine runs. Test: `T-ENV-CHECK-01`.
---

## 5. Functional Requirements

Format: requirement, then **AC** (falsifiable acceptance criterion) and **test id**. Where v1.0 gave
a value without a unit, a default, or a range, v2.0 supplies all three.

### 5.1 FR-1: Media I/O

#### FR-1.1 Containers

Input: MP4, MKV, AVI, MOV. Output: MP4 and MKV (MKV required when the source carries streams MP4
cannot hold — e.g. multiple subtitle formats, PGS).
**AC** — Each container round-trips a fixture through probe → process → mux with stream counts
preserved. `T-IO-CONTAINER-01..04`.

#### FR-1.2 Video codecs

Input: H.264/AVC, H.265/HEVC, and AV1 where the installed decoder supports it. Unsupported input
fails with `E1003` before any processing begins, never mid-job.
**AC** — Probing an unsupported profile yields `E1003` within 2 s and creates no job directory.
`T-IO-CODEC-REJECT-01`.

#### FR-1.3 Decode backend

Priority: NVDEC → D3D11VA → FFmpeg software. The selected backend is reported in `probe` (§8.3) and
recorded in the job state. A hardware-decode failure shall fall back to software **once**, log
`W1101`, and continue — not fail the job.
**AC** — With NVDEC forcibly disabled, the same fixture decodes to bit-identical frames via software
and the job reports `decoder="software"`. `T-IO-DECODE-FALLBACK-01`.

#### FR-1.4 Encode backend

Two encoder profiles, user-selectable, **x265 the default** (D-12):

| Profile | Encoder | When |
| --- | --- | --- |
| **Quality (default)** | **x265**, `slow` preset, CRF-based | Final output. Reaches the FR-1.8 transparency threshold at a materially lower bitrate than NVENC HEVC, which is what keeps the untouched 85%+ of the picture intact |
| **Speed** | NVENC H.265 / H.264 | Long files, iteration, preview export |
| Fallback | x264 | H.264 output when required for compatibility |

Rate control is **constant-quality** targeting visual transparency on untouched regions (FR-1.8), not
a fixed bitrate. Encoding is CPU-bound in the Quality profile; the 32-thread CPU in §4.5 makes that
acceptable, and §6.1's throughput targets are stated per profile.

**AC** — Encoder selection is observable in the result payload, the produced stream metadata matches
the request, and both profiles independently satisfy FR-1.8's transparency threshold.
`T-IO-ENCODE-SELECT-01`, `T-QUALITY-NULLRUN-01`.

#### FR-1.5 Audio preservation

Audio streams shall be **stream-copied**, never transcoded, unless the user explicitly opts in. All
audio tracks are preserved where the output container supports them. Audio is copied from the source
with its original timestamps; the video timeline is not resampled.
**AC** — A 3-track fixture exports with 3 bit-identical audio streams (hash comparison of extracted
streams). `T-IO-AUDIO-COPY-01`.

#### FR-1.6 Subtitles and metadata

Preserve subtitle streams, chapters, rotation/display-matrix metadata, and language tags where the
destination container supports them. Add the product's own metadata:
`demosaic_studio.version`, `demosaic_studio.model_versions`, `demosaic_studio.synthetic=true`,
`demosaic_studio.confidence_mean`.
**AC** — `T-EXPORT-META-01` asserts all four keys are present and `synthetic` is `true` for any job
that restored at least one region. Rotation metadata survives round-trip: `T-IO-ROTATION-01`.

#### FR-1.7 Timestamps, CFR and VFR — **output timing rule** (closes G-13)

Processing is PTS-driven. v1.0 stopped there; the rule that actually prevents drift is about output:

- The output frame for source frame *f* carries **the source PTS of *f***, rescaled to the output
  time base. No frame is dropped, duplicated, or retimed by the restoration pipeline.
- **CFR input → CFR output.** **VFR input → VFR output**, preserving per-frame durations; the muxer
  writes the original timestamps rather than a synthesized constant rate.
- Frame *count* is invariant: `output_frames == input_frames`. This is asserted, not assumed.
- Audio is never resampled or shifted; A/V sync is therefore preserved by construction rather than by
  correction.
- B-frame reordering means decode order != presentation order. The temporal window (§5.6) is built in
  **presentation order**, and the scheduler shall not assume decode order.

**AC** — For CFR and VFR fixtures, output frame count equals input frame count, and per-frame PTS
deltas match the source within one time-base tick. `T-IO-PTS-CFR-01`, `T-IO-PTS-VFR-01`.

#### FR-1.8 Untouched-region quality (closes G-14) — new in v2.0

Restoration touches a small fraction of the frame (typically <=15%), but a naive full re-encode
degrades **100%** of it. Users compare against the source and read a globally softer picture as
damage.

- **R-1.8a** When a job produces **zero** restored regions across the whole file, the engine shall
  **stream-copy** the video (no re-encode) and report `passthrough=true`.
- **R-1.8b** Otherwise the default is a full re-encode at a quality-preserving target: CRF/CQ chosen
  so that on a **null run** (restoration disabled, same encode settings) the output scores
  **>= 42 dB PSNR / >= 0.99 SSIM** against the source. That figure defines "visually transparent" for
  this product, and the default is *derived from the measurement per encoder profile*, not guessed.
  Expect x265 `slow` to reach it at a substantially lower bitrate than NVENC — that gap is the
  practical payoff of D-12, and it shall be measured and recorded in `docs/benchmarks/`, not assumed.
- **R-1.8c** Segment-wise smart-cut (re-encode only GOPs containing restored frames, stream-copy the
  rest) is **deferred**, not rejected. D-07. It is the correct long-term answer and it needs
  keyframe-accurate splicing that is out of MVP scope.

**AC** — The null-run test asserts the transparency threshold for each encoder profile, and the
zero-detection test asserts byte-identical video stream between source and output.
`T-QUALITY-NULLRUN-01`, `T-IO-PASSTHROUGH-COPY-01`.

---

### 5.2 FR-2: Mosaic Detection and Segmentation

#### FR-2.1 Output

The detector produces, per frame: a **pixel-level mosaic mask** (primary), per-region bounding boxes
(for tracking/scheduling only), per-region confidence, and an optional degradation class.
Masks — not boxes — define restoration boundaries.
**AC** — The engine's region records always carry a mask; a region with a box and no mask is a
schema violation. `T-DET-SCHEMA-01`.

#### FR-2.2 Model and input geometry (closes G-03)

- Architecture: **binary semantic segmentation**, U-Net decoder over an ImageNet-pretrained `timm`
  encoder (Apache-2.0). D-03.
- Inference resolution: fixed short side of **512 px** with aspect preserved, tiled at 512x512 with
  64 px overlap for inputs above 1080p; mask upsampled to source resolution with bilinear + threshold.
- Precision: FP16 on CUDA, FP32 elsewhere.
- Output stride and mask post-processing: connected components, drop components below
  `min_region_area` (default **256 px²** at source resolution), morphological close with a 3x3 kernel.

#### FR-2.3 Thresholds

| Parameter | Default | Range | Exposed |
| --- | --- | --- | --- |
| Detection confidence | `0.45` | `0.10 - 0.90` | Main settings |
| NMS IoU (box-level, for track association) | `0.50` | `0.30 - 0.80` | Advanced |
| `min_region_area` | `256 px²` | `64 - 4096` | Advanced |
| Mask binarization threshold | `0.50` | `0.30 - 0.70` | Advanced |

#### FR-2.4 Multi-region

Multiple simultaneous mosaic regions per frame shall be supported, with a configurable
`max_regions_per_frame` (default **16**) to bound worst-case memory. Exceeding it keeps the
highest-confidence regions and logs `W3101`.
**AC** — A synthetic 20-region frame yields exactly 16 processed regions plus the warning.
`T-DET-MAXREGIONS-01`.

#### FR-2.5 False positives (closes G-06) — new in v2.0

A detector optimized only for recall will fire on defocus blur, bokeh, low-bitrate blocking, motion
blur, pixel-art content, and LED/tile textures, and then the restoration stage will *alter footage
that was never mosaicked*. That is the worst user-visible failure this product can produce, and v1.0
had no requirement against it.

- **R-2.5a** Against the hard-negative corpus (§11.4), at the default threshold, **<= 0.5%** of
  negative frames may produce any region, and **<= 0.1%** may produce a region larger than 0.5% of
  frame area.
- **R-2.5b** Temporal gating: a newly detected region shall not be restored until it has been
  confirmed on **>= 2 consecutive frames** (`min_confirm_frames`, default 2, range 1-5). A
  single-frame flash is a false positive far more often than it is a real one-frame mosaic.
- **R-2.5c** The UI shall provide a per-job "detected regions" summary before processing (the
  `analyze` command, §8.3), so the user can see *what* will be altered rather than discovering it in
  the output.

**AC** — `T-DET-FPRATE-01` runs the negative corpus and asserts both rates.
`T-DET-CONFIRM-FRAMES-01` asserts a one-frame detection produces no restoration.

---

### 5.3 FR-3: Tracking and Temporal Smoothing

#### FR-3.1 Tracker

ByteTrack-style two-stage association (high-confidence, then low-confidence leftovers) with a Kalman
filter per track, reimplemented in-repo (§4.2). Track IDs are stable and monotonic per job.

#### FR-3.2 Kalman state

`x = [cx, cy, w, h, vx, vy, vw, vh]^T`, constant-velocity transition, measurement `H` selecting
`[cx, cy, w, h]`. Process/measurement noise are per-preset constants, tunable in advanced settings,
with defaults calibrated on the Phase 1 tracking set.

#### FR-3.3 Track states

`TENTATIVE → ACTIVE → {OCCLUDED, LOST} → {REACQUIRED → ACTIVE, TERMINATED}`. Transitions are a
**table-driven rule set**, not scattered conditionals: forward progress along the active path is
allowed, backward transitions are rejected, and `TERMINATED` is reachable only from `LOST` or by
end-of-stream. State-machine violations raise `E3201` rather than silently correcting.
**AC** — Every transition in the table has a test; every transition absent from the table is asserted
to throw. `T-TRACK-STATE-TABLE-01`.

#### FR-3.4 Missing detections

`max_missing_frames` default **3**, range `0 - 15`. While missing, the mask is propagated by the
Kalman prediction plus optical-flow warp, and the region remains eligible for restoration with a
confidence penalty. Beyond the limit the track goes `LOST`.
**AC** — Removing detections for 3 consecutive frames keeps the track ACTIVE/OCCLUDED and continues
restoration; removing 4 sends it LOST. `T-TRACK-MISSING-01`.

#### FR-3.5 Smoothing

Kalman + EMA on box geometry, with the EMA factor **motion-adaptive**: heavy smoothing at low motion,
light at high motion, so that fast-moving regions do not lag behind the content.
**AC** — On a synthetic constant-velocity sequence, steady-state box lag is <= 1 px at 2 px/frame and
<= 3 px at 20 px/frame. `T-TRACK-LAG-01`.

---

### 5.4 FR-4: Mosaic Degradation Analysis

Per active track, the engine estimates a `MosaicProfile`:

```text
MosaicProfile {
    type:                 PIXELATION | GAUSSIAN_BLUR | BOX_BLUR | MIXED | UNKNOWN
    block_width:          float px           # >= 1
    block_height:         float px
    grid_offset_x:        float px           # phase in [0, block_width)
    grid_offset_y:        float px
    grid_anchor:          SCREEN | OBJECT | UNKNOWN     # new in v2.0, §5.4.4
    grid_anchor_conf:     float [0,1]
    degradation_strength: float [0,1]
    temporal_stability:   float [0,1]
    confidence:           float [0,1]
}
```

#### FR-4.1 Type classification

At minimum the five types above. Classification uses the ROI's gradient/edge statistics: pixelation
produces a strongly periodic gradient comb, blur does not.

#### FR-4.2 Block geometry

For pixelated regions, estimate block width, block height (**non-square supported**), and grid phase.
Method: 2-D autocorrelation or FFT of the ROI's gradient magnitude; the dominant periodicity gives
block size, and the argmax of the edge-position histogram modulo block size gives phase.
**AC** — On synthetic mosaics with known `B` in `[4, 32]` and known phase, estimate block size within
**±1 px** and phase within **±1 px** on >= 95% of samples. `T-PROFILE-BLOCK-EST-01`.

#### FR-4.3 Temporal profile stabilization

Profiles are aggregated per Track ID with a running robust estimator. A profile change is accepted
only with sufficient evidence: **>= 3 consecutive frames** disagreeing with the stable value by more
than the estimator's tolerance. This prevents per-frame parameter oscillation.
**AC** — Injecting a single-frame outlier does not move the stabilized profile; a sustained change
does, within 3-5 frames. `T-PROFILE-STABILITY-01`.

#### FR-4.4 Grid anchoring estimation — new in v2.0, closes G-01 operationally

Estimate whether the mosaic grid is fixed to frame coordinates or moves with the tracked object, by
comparing the per-frame estimated grid phase against (a) frame coordinates and (b) the track's
predicted motion:

- Phase constant in **frame** coordinates while the box moves → `SCREEN`.
- Phase constant relative to the **box origin** → `OBJECT`.
- Neither stable, or the track is static so the two hypotheses are indistinguishable → `UNKNOWN`.

The router (§5.8) shall treat `OBJECT` as "temporal evidence is unavailable" and prefer single-frame
restoration with reduced confidence, rather than spending the multi-frame budget for nothing.
**AC** — Synthetic clips generated with each anchoring mode are classified correctly on >= 90% of
tracks with >= 8 frames of motion; static-subject clips are classified `UNKNOWN` rather than guessed.
`T-PROFILE-ANCHOR-01..03`.

---

### 5.5 FR-5: Temporal ROI Stabilization

#### FR-5.1 Adaptive padding

```text
padding = max(minimum_padding_px, bbox_short_side * padding_ratio, estimated_block * 2)
```

Defaults: `minimum_padding_px = 16`, `padding_ratio = 0.15` (range `0.10 - 0.20`). Padding is never
hard-coded to a single ratio, because a 20 px region and a 600 px region need different absolute
context, and a large mosaic block needs at least two blocks of surrounding context to estimate phase.

#### FR-5.2 Boundary handling

When the padded ROI crosses the frame edge, clamp the read window and use **replication or
reflection** padding to reach the model's expected size. Zero-padding is prohibited as a default: it
injects a hard black edge the model reads as content.
**AC** — A region touching each of the four edges (and each corner) restores without an edge seam
exceeding the boundary-quality threshold of §12.5. `T-ROI-EDGE-01..08`.

#### FR-5.3 Tensor alignment

ROI dimensions are aligned up to a multiple of **32**, with the alignment padding taken from real
neighbouring pixels where available and by replication otherwise. The restored output is cropped back
to the unaligned ROI before blending.
**AC** — Alignment padding is never composited into the output frame. `T-ROI-ALIGN-CROP-01`.

#### FR-5.4 ROI jitter

The ROI used for restoration is derived from the **smoothed** track (§5.3.5), not the raw detection,
and its position is quantized to the mosaic grid phase where a reliable phase estimate exists, so
that the model sees a consistent grid alignment across the temporal window. Frame-to-frame ROI
displacement in a static scene shall not exceed 1 px.
**AC** — `T-ROI-JITTER-01` measures ROI displacement on a static-subject clip.

---

### 5.6 FR-6: Adaptive Temporal Window

Window sizes `K in {3, 5, 7, 9}`, default **5**, centered on the target frame where possible
(`floor(K/2)` before and after), truncated at stream and scene boundaries.

| Condition | K |
| --- | --- |
| Low motion (median flow < 1 px/frame) | 7-9 |
| Medium motion (1-6 px/frame) | 5 |
| High motion (> 6 px/frame) | 3 |
| Within `floor(K/2)` frames of a scene cut | truncate to the available same-scene frames |
| `grid_anchor == OBJECT` | 1 (single-frame; see §5.4.4) |
| VRAM pressure (§5.14) | step down |

Look-ahead of `floor(K/2)` frames is a **pipeline latency**, not a stall: the scheduler keeps a
presentation-ordered ring buffer and emits frame *f* once *f + floor(K/2)* has been decoded.
**AC** — For each condition the selected K is observable in diagnostics and matches the table.
`T-WINDOW-POLICY-01..06`.

---

### 5.7 FR-7: Temporal Alignment

The engine shall align neighbouring frames to the target frame with **sub-pixel** accuracy inside the
ROI. Permitted implementations: optical flow, deformable alignment, feature-space alignment, or
learned implicit alignment. DCNv2 is explicitly **not** required.

**v3.0 relaxation.** v1.0's constraint — that the implementation be supported by every shipped backend
or carry a tested fallback — was the single most likely source of a "works in PyTorch, fails in
ONNX/TensorRT/DirectML" break. D-09 and D-11 remove that risk entirely: the only runtime is PyTorch +
CUDA, so a custom or deformable op is an ordinary implementation choice rather than an export hazard.
This is the largest single simplification v3.0 buys, and it lands on the hardest requirement in the
document. The remaining constraint is only that the op run correctly on the CPU path too (§5.17a),
which is a correctness matter, not a portability one.

Each aligned neighbour carries an **alignment confidence** in `[0,1]` derived from forward-backward
flow consistency and photometric residual. Neighbours below `align_conf_min` (default **0.35**) are
excluded from fusion rather than down-weighted to near-zero.
**AC** — On synthetic sequences with known ground-truth motion, mean endpoint error inside the ROI is
below 0.5 px for motion up to 8 px/frame; occluded neighbours are excluded, not fused.
`T-ALIGN-EPE-01`, `T-ALIGN-OCCLUSION-EXCLUDE-01`.

---

### 5.8 FR-8: Restoration Strategy Router

Per track, per frame, the router selects one path:

| Path | Selected when |
| --- | --- |
| **A. Multi-frame** | `>= 2` valid aligned neighbours, `grid_anchor != OBJECT`, no scene cut inside the window, mean alignment confidence `>= align_conf_min`, VRAM budget sufficient |
| **B. Single-frame** | Scene cut truncates the window, alignment confidence poor, only one valid frame, occlusion invalidates neighbours, `grid_anchor == OBJECT`, or VRAM forced a step-down past K=3 |
| **C. Pass-through** | No region, region below `min_region_area`, region unconfirmed (§5.2.5b), or user disabled restoration for that region |

The chosen path, and **the reason it was chosen**, are recorded per frame in diagnostics. A router
that cannot explain itself cannot be debugged on a two-hour file.
**AC** — Every routing decision emits a reason code from a closed enum, and a test asserts the enum is
exhaustive over the branch conditions. `T-ROUTER-REASON-01`.

---

### 5.9 FR-9: Restoration Backends

#### FR-9.1 Interface, not model (kept from v1.0, made concrete)

```text
IRestorationBackend
    name, version, supported_windows, min_roi, max_roi, precision, runtimes
    prepare(profile, window_size, roi_size) -> plan
    restore(target_roi, aligned_neighbours[], mask, profile, plan) -> (roi_out, confidence_map)
    vram_estimate(roi_size, window_size) -> bytes
```

Three implementations, one per quality preset (§15): `Fast`, `Balanced`, `Quality`. The router
selects by preset and by `vram_estimate`.

#### FR-9.2 Model choice (D-04, closes G-05)

The restoration model is **trained in-house on the degradation this product actually faces**. This
matters more than any architecture choice: BasicVSR++ and its relatives are trained to invert
*bicubic downsampling*. Pixelation is box-averaging with a phase, followed by codec quantization —
a different forward operator with a different null space. A model with the wrong degradation prior
produces confidently wrong texture.

**v3.0 (D-11, D-04):** third-party VSR weights may now also be used as a **training initialization**,
not only as a baseline. The reasoning is that the wrong *degradation prior* lives mostly in the
reconstruction head, while the low-level feature extractor and the flow-guided alignment module are
largely degradation-agnostic and should transfer. That is a defensible expectation and nothing more.
Phase 2 therefore runs **initialized vs. from-scratch as a measured comparison** under §13.5, and if
the difference sits inside the noise floor the initialization is dropped — carrying a contaminating
license (§4.2, §18 R-05) for an unmeasured benefit is the worst of both outcomes.

Third-party weights are used **only behind `IRestorationBackend`** (R-4.2a).

#### FR-9.3 Degradation conditioning

The `MosaicProfile` (block size, phase, type, strength) shall be supplied to the model, as
conditioning channels or FiLM-style modulation. The Phase 2 report shall include an **ablation**
showing the measured benefit of conditioning; if it is inside the noise floor, conditioning is
dropped rather than kept for narrative reasons.
**AC** — `T-RESTORE-CONDITIONING-ABLATION-01` is a reported experiment, not a pass/fail gate.

#### FR-9.4 Restoration confidence

Per-pixel confidence in `[0,1]` combining: number of valid temporal observations, their alignment
confidence, phase diversity across the window (the §1.4.1 quantity), mosaic severity relative to
§1.4.2's bands, and model uncertainty where available. Aggregated per region, per track, and per job.
Buckets: `High >= 0.66`, `Medium >= 0.33`, `Low < 0.33`.
**AC** — Confidence is monotone in phase diversity and in block size on synthetic sweeps: larger
blocks and lower diversity never produce higher confidence. `T-CONF-MONOTONE-01`.

---

### 5.10 FR-10: Temporal Consistency

Flicker is a first-class defect, not a polish item. A restored ROI shall not be accepted on
single-frame perceptual quality alone if it destabilizes across frames.

Mechanisms, at least two of which shall be implemented:
recurrent hidden state carried along the track, previous-output guidance warped by flow into the
current frame, flow-guided consistency loss during training, and post-restoration temporal
stabilization of the restored ROI.

**AC** — On the temporal evaluation set, warping error of the restored region is **<= 1.3x** the
warping error of the same region in the ground-truth clip, and strictly lower than the single-frame
baseline's. Measured with the §13.5 noise-floor protocol. `T-TEMPORAL-WARP-01`.

---

### 5.11 FR-11: Mask-Aware Blending

```text
segmentation mask -> controlled dilation -> edge-aware feathering
                  -> temporal alpha smoothing -> compositing
```

- Dilation: default **2 px**, plus `ceil(block_size / 4)` for pixelated regions, because the mosaic's
  influence bleeds to the enclosing block boundary.
- Feathering: edge-aware (guided by source gradients), width default **3 px**, range `1 - 9`.
  Plain Gaussian feathering is permitted as one component but shall not be the whole strategy.
- Temporal alpha smoothing: the alpha map is EMA-smoothed along the track to prevent the mask edge
  from breathing.
- Compositing is done in **linear light** at the working bit depth, not in gamma-encoded 8-bit, to
  avoid a visible edge in gradients.

Minimize: halo, visible rectangular ROI borders, ghost edges, temporal alpha flicker.
**AC** — Boundary metrics of §12.5 pass on the boundary test set, and a rectangle-detector test
asserts no axis-aligned discontinuity at ROI borders. `T-BLEND-NO-RECT-01`, `T-BLEND-HALO-01`.

---

### 5.12 FR-12: Scene Cut Detection

Signals: HSV histogram divergence, mean absolute frame difference, and optical-flow discontinuity;
combined with hysteresis to avoid firing on camera flashes and fast pans.

On a cut: reset temporal buffers, terminate or re-initialize affected tracks, and use truncated-window
or single-frame restoration until enough same-scene context exists.

**Camera flash is not a cut.** A one-to-three frame global luminance spike with unchanged structure
shall be classified as a flash and shall not reset temporal context.
**AC** — On the scene-cut fixture set: recall >= 0.95, precision >= 0.90, and **zero** cuts reported
on the flash-only clip. `T-SCENE-CUT-01`, `T-SCENE-FLASH-01`.

---

### 5.13 FR-13: Pipeline Scheduler

Asynchronous, bounded, back-pressured stages as in §3.1. Requirements:

- Bounded producer/consumer queues; default depth **8** frames per stage, configurable.
- Back-pressure propagates to the decoder; the decoder never runs unboundedly ahead.
- Cancellation and pause propagate to every stage within **500 ms**.
- Deterministic resource cleanup: every stage owns its buffers and releases them on shutdown, including
  on the error path. Cancellation-token ownership belongs to the pump that created it — a stage's
  `finally` block shall not cancel or dispose a token it does not own, because that turns a normal
  shutdown into an exception that skips the state transition to `Idle` and hangs the queue.
- Reusable tensor/buffer pools; pinned host memory and CUDA streams where beneficial.
- Presentation-order ring buffer for the temporal window (§5.6, FR-1.7).

**AC** — A cancel issued mid-job returns a terminal `result` within 500 ms with no leaked threads,
no leaked file handles, and a valid checkpoint. Repeated 100x without growth.
`T-SCHED-CANCEL-01`, `T-SCHED-LEAK-01`.

---

### 5.14 FR-14: GPU Memory Management

VRAM budget modes: `Auto`, `4 GB`, `6 GB`, `8 GB`, `12 GB+`. `Auto` uses a safe fraction of *free*
VRAM (default **70%**, floor 2 GB) measured at job start, never all of it, and never total VRAM.

OOM mitigation ladder, applied in order, each step logged with the code from §10:

1. Reduce restoration batch size.
2. Reduce temporal window `K` (9 → 7 → 5 → 3).
3. Enable tiling.
4. Reduce tile dimensions.
5. Switch to a lower-memory backend (`Quality` → `Balanced` → `Fast`).
6. Switch to the fallback runtime if appropriate.
7. Fail the job with `E4401` and actionable diagnostics.

**The engine shall not silently produce corrupted output after OOM recovery.** Any frame whose
restoration was interrupted by OOM is either fully re-restored at the reduced setting or passed
through unmodified — never partially composited.

**AC** — A fault-injection test raises OOM at each ladder step and asserts (a) the ladder advances in
order, (b) every output frame is either fully restored or byte-identical to source, (c) the settings
downgrade appears in diagnostics. `T-VRAM-LADDER-01..07`, `T-VRAM-NO-PARTIAL-01`.

Note: `computeType`/precision is a **performance knob** and therefore excluded from checkpoint
fingerprints (§9.3). An OOM-driven precision downgrade must not invalidate the work already done.

---

### 5.15 FR-15: Job Checkpoint and Resume

Specified in full in §9. Summary of the requirement: long jobs checkpoint at a bounded interval
(default every **10 s** or **300 frames**, whichever first) and on cancellation; resume verifies
source identity, application version, model versions, and **per-artifact settings fingerprints**, and
discards only the artifacts that are actually invalidated.
**AC** — Kill the worker process (SIGKILL equivalent) at 50% of a job and resume: the output is
equivalent to an uninterrupted run within the noise floor, and no work before the last checkpoint is
repeated. `T-RESUME-KILL-01`.

---

### 5.16 FR-16: User Interface

#### 5.16.1 Drag and drop
Supported files dropped anywhere on the main window are accepted; unsupported files are rejected with
a specific reason, not a generic beep.

#### 5.16.2 Metadata display
Resolution, duration, nominal FPS, CFR/VFR indication, video codec, audio codec(s), subtitle
stream(s), container, file size, and the detected decode/encode backends.

#### 5.16.3 Preview (closes G-16)
Three modes: original, restored, split-view with a draggable divider. Preview is served by the same
engine via the `preview` command (§8.6) so that what the user sees is what the pipeline produces —
never a second, subtly different code path. Preview requests are **cancellable and coalesced**: a
scrub gesture issues many requests and only the latest is honoured.
**AC** — Scrubbing rapidly across a 2-hour file never queues more than one outstanding preview request
and never blocks the UI thread. `T-UI-PREVIEW-COALESCE-01`.

#### 5.16.4 Diagnostic overlay
Toggleable: mosaic mask, bounding box, Track ID, detection confidence, estimated block size,
**grid anchoring**, restoration confidence, selected path and reason (§5.8), processing FPS.

#### 5.16.5 Quality presets
`Fast`, `Balanced`, `Quality` (§15). Everything else lives in an advanced panel.

#### 5.16.6 Hardware selection
`Auto` (default), `NVIDIA CUDA`, `CPU`. DirectML is dropped under D-11/D-09 (§5.17). Unavailable
options are disabled **with the reason shown**, not hidden. A precision explicitly saved as
`float16` on an NVIDIA machine shall not be
applied verbatim on a CPU-only machine — the engine re-resolves precision against the active device
and logs the substitution. (Carrying a saved `float16` onto a CPU backend is a hard load failure that
no OOM ladder recovers from.)

#### 5.16.7 Detection sensitivity
Slider `0.10 - 0.90`, default `0.45`, with a live "regions found on this frame" readout so the number
means something to the user.

#### 5.16.8 Job control
Start, Pause, Resume, Cancel, Retry-after-failure, Open output folder. Buttons are enabled/disabled
per the job's state, and an action that is unavailable explains why. "Nothing selected" and "selected,
but this action does not apply to it" are **different messages**.
**AC** — A state/action matrix test asserts the enablement and the message for every (state, action)
pair. `T-UI-ACTION-MATRIX-01`.

#### 5.16.9 Strings
All user-visible text is externalized to resource files; no literal user-facing strings in code. If
the project ships a hand-maintained designer file alongside the resource file, a parity test shall
assert the two agree.
**AC** — `T-UI-STRING-PARITY-01`.

---

### 5.17 FR-17: Runtime and Backend Fallback (closes G-15) — new in v2.0

**Substantially reduced in v3.0.** With one known machine (§4.5) and no distribution (D-11), backend
portability stops being a requirement and becomes speculative work. Dropped: ONNX export, TensorRT
engines and their per-GPU cache, the DirectML path, the per-model op-support gate, and §13.6's
numerical-parity tolerances (retained in the document for the day export returns, marked as inactive).

What remains:

- **R-17a** Supported backends are **CUDA** (primary) and **CPU** (debug, and correctness comparison).
  Nothing else is offered, so nothing else can fail at load.
- **R-17b** Precision is **re-resolved against the active device at job start**, never applied
  verbatim from saved settings. A saved `float16` carried onto the CPU backend is a hard load failure
  that no OOM ladder recovers from, and the substitution is logged (`W6101`, §5.16.6).
- **R-17c** The CPU path is not a performance path. When selected, the UI states an honest speed
  estimate before the job starts rather than appearing to hang.

**AC** — `T-BACKEND-CPU-PRECISION-01` asserts a saved `float16` runs on CPU by substitution, not by
failure. `T-BACKEND-SET-01` asserts only CUDA and CPU are selectable.
---

## 6. Non-Functional Requirements

### 6.1 Performance

Reference configuration (corrected to the actual dev machine, §4.5):

```text
Input:  1920x1080, 30 FPS, H.264, CFR
GPU:    NVIDIA RTX 3080 Ti 12 GB, driver 591.86
Mosaic: <= 15% of frame area, screen-anchored, B in [6,12]
Mode:   Balanced
```

Two tiers, because one number that no build can meet is not a target (D-01, §3.4). Targets are stated
per **encoder profile** (§5.1.4), because x265 `slow` is CPU-bound and will often be the binding
constraint rather than the GPU:

| Component | MVP target (Python engine) | Optimized target (Phase 5) |
| --- | ---: | ---: |
| Detector only | >= 60 FPS | >= 90 FPS |
| End-to-end, restoration active, **NVENC (Speed)** | **>= 8 FPS** | >= 15 FPS |
| End-to-end, restoration active, **x265 slow (Quality)** | **>= 4 FPS** | >= 8 FPS |
| Pass-through path (no detections) | >= 100 FPS | stream-copy, I/O bound |
| UI responsiveness | no blocking during processing | same |

The Optimized tier's mechanism changed in v3.0: TensorRT is dropped (D-09), so the remaining levers
are tiling, batching, GPU-resident decode/encode, `torch.compile` / CUDA graphs, and — only if
measurement justifies it — the native engine (§3.4).

**Every performance result shall report ROI coverage**, because restoration throughput scales with
processed area and a number without coverage is meaningless. Results shall also report the resolved
backend, precision, and window size.

**AC-6.1** — `scripts/bench.py` produces a table with all of the above for the reference clip set and
writes it to `docs/benchmarks/<version>.md`. A performance claim not produced by this script is not a
performance claim. `T-BENCH-SCHEMA-01`.

#### 6.1.1 Startup latency

Startup and buffering latency is measured **separately** from steady-state throughput. v1.0's
`<200 ms` five-frame requirement is withdrawn as a universal guarantee: decoder buffering, B-frame
reordering, model initialization, first-call CUDA kernel compilation, and temporal look-ahead all vary by
source and backend. Replacement requirement: **time-to-first-restored-frame is reported**, and the UI
shows a determinate "preparing" state rather than an apparently frozen progress bar.

### 6.2 Stability

- Jobs longer than one hour shall run without unbounded memory growth.
- Measurable criterion: after a 10-minute warm-up on a 2-hour input, host RSS and worker RSS shall
  each stay within **±10%** of their warm-up plateau, and peak VRAM shall not trend upward. No
  resource whose count grows proportionally to processed frames.
- v1.0's "0% memory leak" is withdrawn as unmeasurable and replaced by the above.

**AC-6.2** — `T-LONGRUN-MEMORY-01` (nightly, GPU machine) samples RSS/VRAM every 30 s across a 2-hour
run and asserts the bounds, emitting the trace on failure.

### 6.3 Compatibility

**Scope narrowed in v3.0 (D-11): one target machine, the one in §4.5.** Compatibility beyond it is
not a requirement and shall not consume effort.

- Windows 11 x64 (the dev machine). Windows 10 build 19041+ is expected to work and is not tested.
- NVIDIA CUDA pinned in `docs/RUNTIME_MATRIX.md` for the installed driver. The product does not claim
  "CUDA 12.x" in general and does not need to.
- Non-NVIDIA GPUs: **not supported.** CPU only, per §5.17.
- Windows GPU note carried from hard-won experience: CTranslate2-style native CUDA consumers on
  Windows need the CUDA support DLLs (cuBLAS/cuDNN) both **present** and **actually loaded** —
  registering a DLL directory is not sufficient, and Python 3.8+ ignores `PATH` for DLL resolution.
  Whatever native runtime the engine ends up using, the CUDA bring-up code shall *load* its support
  libraries explicitly at startup and shall have a test that fails if it only registers directories.

**AC-6.3** — `T-CUDA-BRINGUP-01` asserts the bring-up path loads (not merely registers) each required
support library, in dependency order.

### 6.4 Reliability

A failed job shall never require restarting the application. Errors are categorized and numbered in
§10. Logs carry diagnostic context without user media content (§2.3 C-6).

### 6.5 Security and privacy

- No network access during processing. The only network operations are model download (§14.2) and an
  explicit user-initiated update check.
- Model downloads are verified by SHA-256 against a signed manifest before use.
- Temp/job directories are created under the user's local app data with default ACLs and are removed
  on successful completion unless the user opted to keep artifacts.

**AC-6.5** — `T-NET-QUIET-01` asserts zero outbound connections during a full offline job run.

---

## 7. Algorithmic Specifications

### 7.1 Bounding-box Kalman filter

State `x_k = [cx, cy, w, h, vx, vy, vw, vh]^T`, constant-velocity model.

```text
predict:  x_k^- = A x_(k-1)              P_k^- = A P_(k-1) A^T + Q
gain:     K_k   = P_k^- H^T (H P_k^- H^T + R)^-1
update:   x_k   = x_k^- + K_k (z_k - H x_k^-)
          P_k   = (I - K_k H) P_k^-
```

`Q` and `R` are per-preset and motion-adaptive (§5.3.5). Numerical requirement: `P` is kept symmetric
positive-definite via the Joseph form or explicit symmetrization; a filter that diverges on a long
job is a correctness bug, not a tuning issue.
**AC** — `T-KALMAN-PSD-01` runs 100k steps on adversarial measurements and asserts `P` stays SPD.

### 7.2 Mosaic profile aggregation

Per Track ID, robust running estimates (median-of-window for geometry, EMA for strength). Change
acceptance per §5.4.3.

### 7.3 Alignment confidence

`conf = f(forward_backward_flow_error, photometric_residual, occlusion_mask)`, normalized to `[0,1]`,
with the mapping's constants fixed in code and covered by a table test so tuning is visible in diffs.
Neighbours below `align_conf_min` are **excluded**, not merely down-weighted (§5.7).

### 7.4 Restoration confidence

Inputs per §5.9.4. Interpretation:

- **High** — substantial observable information from neighbouring frames.
- **Medium** — temporal inference contributes significantly.
- **Low** — the result depends heavily on model estimation.

Confidence is a diagnostic. It is never proof that reconstructed content matches the original hidden
content (§1.3, §2.3 C-3).

---

## 8. Worker Protocol v1.0 (closes G-02, G-17)

Authoritative expansion lives in `docs/WORKER_PROTOCOL.md`; this section is the specification that
file elaborates.

### 8.1 Transport, framing, versioning

- Transport: the worker's **stdin** (host → worker) and **stdout** (worker → host), UTF-8, one JSON
  object per line, `\n`-terminated, no embedded raw newlines. **stderr** is free-form log text: it is
  captured to the job log and never parsed for control flow.
- Every message: `{"v": "1.0", "type": "...", "id": "<uuid>", "jobId": "<uuid|null>", ...}`.
- `PROTOCOL_VERSION` has exactly one definition per side (§4.3). Bumping it follows the change
  procedure in `AGENTS.md`: bump → change both sides → update `docs/WORKER_PROTOCOL.md` → add a
  round-trip test.
- **Compatibility rule:** the host refuses a worker whose **major** version differs and reports
  `E7001`. Minor differences are accepted; unknown fields are **ignored**, never rejected. This is
  what allows a newer worker to add fields without invalidating an older host's checkpoints.

### 8.2 Message catalogue

Host → worker:

| type | Purpose | Key fields |
| --- | --- | --- |
| `hello` | Handshake | `hostVersion`, `protocolVersion` |
| `probe` | Media + hardware inspection, no processing | `sourcePath` |
| `analyze` | Detection/tracking pass only; produces the region summary for §5.2.5c | `jobId`, `sourcePath`, `settings`, `sampleEvery` |
| `process` | Full pipeline | `jobId`, `sourcePath`, `outputPath`, `settings`, `resume` |
| `preview` | Render one frame, original and restored | `jobId`, `pts`, `settings`, `overlay` |
| `pause` / `resume` | Suspend/continue the running job | `jobId` |
| `cancel` | Cooperative cancel | `jobId` |
| `shutdown` | Terminate the worker | — |

Worker → host:

| type | Purpose | Key fields |
| --- | --- | --- |
| `ready` | Handshake reply | `workerVersion`, `protocolVersion`, `capabilities` |
| `probeResult` | Media/hardware facts | `media{...}`, `hardware{...}` |
| `progress` | Bounded-rate progress | `stage`, `pts`, `fraction`, `fps`, `eta` |
| `log` | Structured log line | `level`, `code?`, `message`, `context{}` |
| `trackUpdate` | Diagnostics/overlay data | `frames[{pts, regions[...]}]` |
| `checkpoint` | Checkpoint written | `lastCompletedPts`, `path` |
| `previewResult` | Rendered preview | `pts`, `originalPath`, `restoredPath`, `regions[]` |
| `result` | **Terminal** for a job | `status: completed\|cancelled\|failed`, `summary{}`, `error?` |
| `error` | Failure detail | `code`, `recoverable`, `message`, `context{}` |

### 8.3 `probe` and `probeResult`

`probeResult.media`: `durationSeconds`, `width`, `height`, `nominalFps`, `isVfr`, `frameCount?`,
`videoCodec`, `pixelFormat`, `bitDepth`, `colorPrimaries/transfer/matrix`, `rotation`,
`audioStreams[]`, `subtitleStreams[]`, `chapters`, `container`, `sizeBytes`.

`probeResult.hardware`: `gpus[{name, vramTotalBytes, vramFreeBytes, driver}]`, `cudaAvailable`,
`nvdecAvailable`, `nvencAvailable`, `directmlAvailable`, `cpuThreads`, `ramBytes`, and
`supportLibraries{}` with per-library load status.

**`cudaAvailable` means "we loaded the libraries and ran a test kernel", not "a driver reports a
device".** A device count from a driver query is not availability; models load and *then* fail. The
probe shall actually exercise the path it is reporting on.
**AC** — `T-PROBE-CUDA-HONEST-01` asserts `cudaAvailable` is false when a support library is present
but unloadable.

### 8.4 Progress contract (closes G-17)

- `progress` is emitted at most **4/s per job**, coalesced. It is advisory.
- `fraction` is monotonically non-decreasing within a job. The worker shall not emit a lower fraction
  than it has already emitted.
- `stage` transitions follow the closed enum `probing → analyzing → restoring → encoding → muxing →
  finalizing`, forward only.
- The host **drops** any `progress` that arrives after a terminal `result` for that job, and any whose
  `stage` or `fraction` moves backwards. It does not apply them and does not error.
- The host's UI progress channel is ordered and inline; progress is never marshalled through a
  free-threaded callback that can reorder it.

**AC** — `T-HOST-PROGRESS-ORDER-01` (also AC-3.3) and `T-WORKER-PROGRESS-RATE-01`.

### 8.5 Lifecycle

1. Host launches the worker, sends `hello`, expects `ready` within **10 s** or fails `E7002`.
2. One job at a time per worker. `process` is rejected while another job is active (`E7003`).
3. `cancel` → worker acknowledges via `log`, drains, writes a checkpoint, emits terminal `result`
   with `status="cancelled"` within **5 s**. After a **10 s** grace period the host may kill the
   process and record `E7004`.
4. Worker crash (non-zero exit without a terminal `result`) fails only the current job with `E7005`;
   the host relaunches the worker for the next job.
5. `shutdown` → worker exits 0 within 5 s.

**AC** — `T-PROTO-LIFECYCLE-01..05`, driven against a fake worker that can be told to misbehave at
each step.

### 8.6 Large payloads: frames, masks, previews (closes G-16)

Pixel data never travels inside JSON. The worker writes to the job's temp directory and sends a path:

```text
%LOCALAPPDATA%\DeMosaicStudio\jobs\<jobId>\
    job.json               # §9
    preview\<pts>.orig.png
    preview\<pts>.rest.png
    diag\<pts>.mask.png    # only when the diagnostic overlay is enabled
    output.tmp.<ext>
    worker.log
```

Preview images are PNG (lossless, so the user is judging the pipeline and not the preview codec).
The host deletes preview artifacts on job close. Preview requests are coalesced and cancellable
(§5.16.3).

### 8.7 Reference transcript

A golden transcript of a complete successful job, a cancelled job, and each failure class lives in
`fixtures/protocol/`. The integration tests replay them against both sides.
**AC** — `T-PROTO-GOLDEN-01..07`.

---

## 9. Job State, Checkpoint, and Invalidation (closes G-07)

### 9.1 Location and format

One directory per job (§8.6), one `job.json`, written atomically (temp file + rename) so an
interrupted write cannot corrupt state.

### 9.2 Schema

```jsonc
{
  "schemaVersion": 1,
  "jobId": "…",
  "appVersion": "0.1.0",
  "protocolVersion": "1.0",
  "source": {
    "path": "…",
    "sizeBytes": 0,
    "mtimeUtc": "…",
    "quickHash": "sha256 of first+last 8 MiB and size"
  },
  "media": { /* probeResult.media snapshot */ },
  "settings": { /* full resolved settings, post-`auto` resolution */ },
  "fingerprints": {
    "detection":   "sha256:…",
    "restoration": "sha256:…",
    "encode":      "sha256:…"
  },
  "models": {
    "detector":   { "id": "det-unet-b0", "version": "1.2.0", "sha256": "…" },
    "restorer":   { "id": "rest-balanced", "version": "0.4.1", "sha256": "…" }
  },
  "progress": {
    "lastCompletedPts": 123456,
    "framesWritten": 3600,
    "stage": "restoring"
  },
  "artifacts": {
    "analysis": { "path": "analysis.jsonl", "complete": true },
    "video":    { "path": "output.tmp.mp4", "complete": false }
  },
  "confidence": { "mean": 0.41, "bucketCounts": { "high": 120, "medium": 900, "low": 300 } }
}
```

### 9.3 Per-artifact fingerprint invalidation — the rule that prevents silent corruption

A single global fingerprint forces a full restart whenever anything changes; no fingerprint lets a
resume mix artifacts produced under different settings. Neither is acceptable. Therefore:

- Each **artifact** records the fingerprint of the settings that **produced** it.
- On resume, artifacts are invalidated **top-down from the first changed stage**: a changed detection
  setting discards analysis *and* video; a changed restoration setting discards video only; a changed
  encode setting discards video only.
- **Performance knobs are excluded from every fingerprint**: VRAM budget, tile size, batch size,
  precision/compute type, thread counts, backend selection. These change during a run (the OOM ladder
  in §5.14 changes precision) and including them would make every resume after a downgrade discard
  completed work.
- **Fingerprints are rewritten immediately after the discard**, not at job completion. Deferring the
  rewrite means a run that fails again re-triggers the same discard on every subsequent retry, so the
  job restarts from zero forever.
- Adding a key to a fingerprint's input set **invalidates every existing checkpoint** (dict equality).
  That is sometimes correct, but it is always a cost, and it shall be a deliberate, noted decision in
  `docs/DECISIONS.md`.
- Resume also verifies `source.quickHash`, `appVersion` major, `protocolVersion` major, and each model
  `sha256`. Any mismatch discards the dependent artifacts with a logged reason.
- Null-safety rule: a missing or unknown fingerprint compares as **"changed"**, never as "equal".
  A null-lifting comparison that evaluates to `false` on unknown data will silently reuse a previous
  file's artifacts for a different source — a data-corruption bug, not a UX bug.

**AC** — `T-RESUME-FINGERPRINT-01..06`: change one setting from each class and assert exactly the
expected artifacts are discarded; assert performance knobs discard nothing; assert an unknown
fingerprint discards; assert the rewrite happens immediately (a second consecutive failed retry does
not re-discard).

### 9.4 Retention

Job directories persist until the user removes the job or the output is confirmed written. A
"remove job" action deletes the directory including cached artifacts, and the UI shall say that it
does.

---

## 10. Error Codes (closes G-08)

### 10.1 Rules

- Every failure crossing the boundary carries a numbered code. Free-text-only errors are a defect.
- Codes are defined once in `worker/demosaic_worker/errors.py`, mirrored in `Domain`, and locked
  together by a parity test (§13.4). Adding a code requires updating both plus
  `docs/ERROR_CODES.md` and `docs/TROUBLESHOOTING.md`.
- Each code carries: `recoverable` (may the host auto-retry?), a user-facing message key, and a
  troubleshooting anchor.
- `W`-prefixed entries are warnings and never fail a job.

### 10.2 Table

| Code | Meaning | Recoverable |
| --- | --- | --- |
| **E1xxx — media / input** | | |
| E1001 | File not found or unreadable | No |
| E1002 | Unsupported container | No |
| E1003 | Unsupported video codec or profile | No |
| E1004 | Corrupt source: demux failure | No |
| E1005 | Source has no video stream | No |
| E1006 | Source metadata inconsistent (declared duration vs. decoded) | Yes (warn + continue) |
| **E2xxx — decode** | | |
| E2001 | Hardware decoder init failed | Yes (→ software) |
| E2002 | Decode error mid-stream, frame unrecoverable | Yes (skip + log) |
| E2003 | Decode error mid-stream, stream unrecoverable | No |
| E2004 | Timestamp discontinuity beyond tolerance | Yes |
| **E3xxx — detection / tracking** | | |
| E3001 | Detector model load failed | No |
| E3002 | Detector inference failure | Yes (retry once, then pass-through frame) |
| E3003 | Detector output shape mismatch (model/runtime mismatch) | No |
| E3201 | Track state-machine violation (internal) | No |
| **E4xxx — restoration** | | |
| E4001 | Restoration model load failed | No |
| E4002 | Restoration inference failure | Yes (→ single-frame → pass-through) |
| E4003 | Alignment failure for the whole window | Yes (→ single-frame) |
| E4004 | ROI smaller than model minimum | Yes (→ pass-through) |
| E4401 | GPU OOM, mitigation ladder exhausted | No |
| E4402 | Backend/runtime unsupported for this model (§5.17) | No |
| **E5xxx — encode / mux** | | |
| E5001 | Encoder init failed | Yes (→ software) |
| E5002 | Encode failure mid-stream | No |
| E5003 | Mux failure | No |
| E5004 | Output container cannot carry a source stream | Yes (drop stream + warn) |
| **E6xxx — system** | | |
| E6001 | Disk full | Yes (after user frees space) |
| E6002 | Output path not writable / permission denied | Yes |
| E6003 | Output file locked by another process | Yes |
| E6004 | Insufficient system RAM | No |
| E6005 | Required support library missing or unloadable | No |
| **E7xxx — protocol / process** | | |
| E7001 | Protocol major version mismatch | No |
| E7002 | Worker handshake timeout | Yes (relaunch) |
| E7003 | Worker busy: a job is already running | No |
| E7004 | Worker did not exit within the cancel grace period | Yes |
| E7005 | Worker crashed | Yes (relaunch, fail current job) |
| E7006 | Malformed protocol message | No |
| **E9xxx** | | |
| E9001 | Unexpected internal error | No |
| **Warnings** | | |
| W1101 | Fell back to software decode | — |
| W3101 | Region count clamped to `max_regions_per_frame` | — |
| W4101 | OOM ladder step applied (carries the step) | — |
| W5101 | Stream dropped for container compatibility | — |
| W6101 | Backend substituted (e.g. precision re-resolved for CPU) | — |

### 10.3 Retry policy

- `recoverable=true` → host may auto-retry **once** per job with the mitigation implied by the code,
  then surfaces it to the user.
- A retried job resumes from the checkpoint; it does not restart.
- Auto-retry requires that the job's status actually be updated as it progresses. A job stuck in an
  early status because a progress update forgot to advance it will make the retry path unreachable
  and turn a recoverable failure into an `E9001`. `T-RETRY-STATUS-01` guards this specific shape.

### 10.4 Logging

`level`, `code`, `message`, `context{}`. Source paths are logged at DEBUG only and as file names at
INFO; no pixel data, no thumbnails (§2.3 C-6). Every job writes `worker.log` in its job directory.
---

## 11. Dataset Specification (closes G-03, G-06, G-11)

### 11.1 Principle

Training data is **generated, not collected**: take clean licensed video, apply the synthetic
degradation generator to chosen regions, and the pre-degradation frames are perfect ground truth with
perfect masks. This gives unlimited paired data, exact labels, and no dependence on collecting real
mosaicked material (D-05, and it is what makes §2.3 C-2 achievable).

The cost is **domain gap**: real mosaics come from unknown tools, at unknown strengths, followed by
unknown re-encoding. §11.3 and §11.6 exist to close it.

### 11.2 Source corpus

| Split | Content | Approx. scale for v1 |
| --- | --- | --- |
| `clean-train` | Licensed/permissive clean video: varied resolution, grain, motion, lighting, skin tones, fabrics, text, faces at varying scale | >= 40 h |
| `clean-val` | Same distribution, **disjoint sources** | >= 4 h |
| `clean-test` | Same distribution, disjoint sources, frozen | >= 4 h |
| `negatives` | §11.4 | >= 6 h |
| `real-mosaic` | Real-world mosaicked clips, **unpaired**, for domain-gap checking and detector recall only | >= 2 h |

**v3.0 (D-11):** models and datasets are never distributed, and training is personal, non-commercial
use, so the corpus is no longer gated on redistribution-friendly licensing — which is what unblocks
§20 Q3 and lets Phase 1 start. Two rules still apply, for reasons that outlive the licensing question:

- Every source file records its origin and known license in `training/datasets/SOURCES.md`. This
  costs minutes and is the only thing that makes §2.4's reversal question answerable later.
- §2.3 C-2 is unaffected: no model is trained on, or fine-tuned toward reconstructing, identifiable
  real individuals. The model learns to invert a *degradation*, and the corpus is chosen accordingly.

### 11.3 Synthetic degradation generator

`training/degradation/` — a deterministic, seeded generator, importable by both training and tests.
Randomizes:

- Block width and height, **independently** (non-square blocks are common in real tools).
- Grid phase/offset, and **grid anchoring** (`SCREEN` / `OBJECT`) — the §1.4.1 property. Anchoring is
  a first-class generator parameter, recorded in the sample's metadata, because the whole feasibility
  question turns on it.
- Degradation type: pixelation, Gaussian blur, box blur, mixed, and pixelation-then-blur.
- Mosaic opacity where applicable; partial mosaic (soft-edged regions).
- Region shape (not just rectangles): ellipses, polygons, and mask shapes traced from real segmentation
  masks.
- Dynamic size and position over time; multiple simultaneous regions; regions crossing frame edges.
- Camera motion, object motion, motion blur.
- Resize/downscale/upscale chains, sensor noise, JPEG, **H.264 and H.265 recompression at a CRF
  ladder**, chroma subsampling (4:2:0 / 4:2:2), bitrate variation.

**Recompression after degradation is mandatory in every evaluation sample.** Real mosaic boundaries
after encoding look substantially different from clean synthetic ones, and a model evaluated only on
clean synthetics will report numbers it cannot reproduce on real files (§1.4.2).

**AC-11.3** — The generator is seeded and reproducible: same seed → byte-identical output.
`T-DEGRADE-DETERMINISM-01`. Every generated sample carries a metadata sidecar with all parameters,
including `grid_anchor` and the recompression settings. `T-DEGRADE-METADATA-01`.

### 11.4 Hard-negative corpus (new in v2.0)

Clean footage that *looks* like mosaic to a naive detector. Required categories:

defocus/bokeh backgrounds · shallow depth of field · heavy motion blur · low-bitrate blocking on flat
areas (dark scenes, sky, walls) · pixel-art and retro game footage · LED walls, tiled surfaces,
mesh/grid fabrics, window blinds · deliberately upscaled low-resolution footage · heavy film grain and
sensor noise · compression breathing on static scenes · screen recordings with UI elements ·
intentionally low-resolution video calls.

Used for the false-positive requirements of §5.2.5 and as negatives during detector training.

### 11.5 Frozen evaluation set `eval-v1`

- Built once from `clean-test` + `negatives` + `real-mosaic`, versioned, hashed, and **frozen**. Its
  manifest (`training/datasets/eval-v1.manifest.json`) records every sample's source, generator
  parameters, and SHA-256.
- All numeric targets in this document refer to `eval-v1`. A target without a named dataset is not a
  target — this is what makes v1.0's `mAP50 >= 0.92` unenforceable and this version's enforceable.
- Stratified so results can be reported per stratum: block-size band (§1.4.2), grid anchoring, motion
  band, ROI coverage band, recompression CRF, resolution.
- Changing `eval-v1` creates `eval-v2`; it never edits `eval-v1`. Historical numbers stay comparable.

### 11.6 Split hygiene

Splits are by **source file and by scene**, never by frame. Near-duplicate frames from the same clip
in different splits inflate every metric. The split tool shall verify with perceptual hashing that no
two frames across splits are within a similarity threshold.
**AC-11.6** — `T-SPLIT-LEAKAGE-01` fails if any cross-split frame pair exceeds the pHash similarity
threshold.

---

## 12. Evaluation Metrics and Harness (closes G-11)

All metrics are produced by checked-in scripts, on `eval-v1`, and written to
`docs/benchmarks/<version>.md`. **A number that was not produced by the harness does not go in a
document, a commit message, or a conversation.**

### 12.1 Detection

Precision, recall, mAP50, mAP50-95, mask IoU, **plus** the false-positive rates of §5.2.5a on the
negative corpus.

Targets on `eval-v1`: mask IoU >= 0.80 and recall >= 0.90 at the default threshold, with FP rates
within §5.2.5a. v1.0's `mAP50 >= 0.92` is retained as a secondary indicator only — box mAP is a weak
proxy for a mask-driven pipeline (§5.2.1).

### 12.2 Tracking

IDF1, ID switches per minute, track fragmentation, lost/reacquired success rate.

### 12.3 Restoration

PSNR, SSIM, LPIPS — reported **per block-size band and per grid-anchoring class**, never as one
aggregate. A single aggregate number hides exactly the effect §1.4 cares about.

Mandatory companion baselines in every restoration report:
1. Pass-through (do nothing).
2. Single-frame restoration.
3. Multi-frame at the evaluated K.

Multi-frame that does not beat single-frame in a band should not be *used* in that band (§5.8).

### 12.4 Temporal quality

At least warping error and temporal LPIPS; flicker metric and flow-consistency error recommended.
Reported for the restored region against the ground-truth region (§5.10 AC).

### 12.5 Boundary quality

Halo score (mean gradient excess in a ring outside the mask), seam visibility (edge energy along the
mask boundary vs. the surrounding), mask-edge temporal consistency, and an axis-aligned-discontinuity
detector for rectangle artifacts (§5.11 AC).

### 12.6 Performance

Decode FPS, detection FPS, restoration FPS, encode FPS, end-to-end FPS, GPU utilization, peak VRAM,
CPU utilization, peak RAM, **ROI coverage %**, resolved backend, precision, window size.

### 12.7 Report format

`scripts/eval_report.py` emits one Markdown table plus a JSON sidecar, both committed. Every row
carries the dataset version, model versions, commit SHA, GPU, and driver.

---

## 13. Test Strategy (closes G-10)

### 13.1 Layers

| Layer | Where | Runs on | Must be |
| --- | --- | --- | --- |
| Domain unit | `tests/DeMosaicStudio.Domain.Tests`, `worker/tests/unit` | CI, no GPU | Fast, deterministic, no I/O |
| Application/orchestration | `tests/DeMosaicStudio.Application.Tests` | CI | Fake engine launcher; covers progress, retry, state |
| Protocol round-trip | `tests/DeMosaicStudio.Integration.Tests` + `worker/tests/protocol` | CI | Replays `fixtures/protocol/` golden transcripts both directions |
| Pipeline integration | `worker/tests/pipeline` | CI (tiny fixtures, CPU) | Real code path on 16x16 synthetic clips |
| Model quality | `training/tests` + §12 harness | GPU machine | Reported, gated at phase boundaries |
| GPU smoke | `scripts/smoke-gpu.ps1` | Dev machine, manual | The only place real GPU throughput is claimed |
| Long-run | nightly, GPU machine | | §6.2 |

### 13.2 GPU-free CI subset

CI shall exercise every layer above the "Model quality" row without a GPU. This requires that
policy/decision logic be free of device dependencies — which is why the router (§5.8), the fingerprint
rules (§9.3), the state machine (§5.3.3), the error mapping (§10), and the window policy (§5.6) all
live in pure, testable code and take their inputs as data.

**If a rule cannot be tested without a GPU, it is in the wrong place.**

### 13.3 Fixtures

- `fixtures/media/`: tiny clips (a few frames, 64x64 up to 320x240) generated by a checked-in script,
  covering CFR, VFR, rotation metadata, multi-audio, subtitles, and a deliberately corrupt file. Small
  enough to commit.
- `fixtures/protocol/`: golden JSON Lines transcripts (§8.7).
- `fixtures/parity/`: shared inputs/outputs for the cross-language parity tests.

### 13.4 Cross-language parity tests

The host and the worker both implement shared semantics. Each pair is locked by a fixture-driven test
that fails if either side drifts:

- Error codes and their `recoverable` flags (§10).
- Settings-fingerprint computation (§9.3) — same settings must hash identically on both sides.
- `PROTOCOL_VERSION` (§8.1).
- Restoration-confidence bucket boundaries (§5.9.4).

### 13.5 Noise-floor protocol — mandatory for every A/B claim

GPU inference is not bit-deterministic: the same input, model, and settings can produce different
output across runs. **Before claiming that a change improved anything, measure how much the metric
moves when nothing changes.**

1. Run configuration X twice, identically. Record the metric delta. That is the noise floor.
2. Run configuration X vs. Y. A difference smaller than the noise floor is **not a result**.
3. Report both numbers together. A claimed improvement stated without its noise floor is rejected in
   review.

This applies to the Phase 0 gate (§1.4.3), the conditioning ablation (§5.9.3), temporal-consistency
claims (§5.10), and every benchmark in §12.

### 13.6 Numerical-parity tolerances

**Export gating is inactive in v3.0** (D-09 dropped ONNX/TensorRT). The tolerances are retained for
the two comparisons that remain live, and are ready if export ever returns:

PyTorch FP32 is the reference. A configuration passes when, on the gate fixture set: mean absolute
error <= `1e-3` and max absolute error <= `1e-2` in `[0,1]` pixel space, and the §12.3 metrics differ
from the reference by less than the §13.5 noise floor.

Live uses: **CUDA FP16 vs. CPU FP32** (catches a precision bug that would otherwise look like a model
quality problem), and **before/after a refactor** of the inference path.

### 13.7 What tests will not catch

Recorded so the team does not mistake a green suite for a working product. The failure classes that
pass CI and break on a user's machine: shell/encoding differences (§4.4), real GPU driver and support
library behaviour (§6.3), real model weights vs. fakes, real container edge cases, and anything
involving actual throughput. Before shipping any change that touches those, run
`scripts/smoke-gpu.ps1` on real hardware. **A green CI run is evidence about the code, not about the
product.**

---

## 14. Model Store and Runtime Setup (was: Packaging and Distribution)

**Substantially reduced in v3.0.** D-11 removes the installer, the download server, the signed
manifest, code signing, and the embedded Python runtime. What survives is the part that was never
really about distribution: **knowing exactly which model produced which output**, which §9.2's
checkpoint fingerprints depend on.

### 14.1 Local model store

```text
models/
  index.json                 # local; generated by the training pipeline, not downloaded
  detector/
    det-unet-b0-1.2.0/
      model.pt
      metadata.json
  restoration/
    fast/  balanced/  quality/
```

`metadata.json` per model declares: model id, version, task, input requirements (size, layout,
normalization), supported temporal windows, precision, required preprocessing, output semantics,
**training dataset version**, **initialization source and its license** (D-04, §4.2), and the SHA-256
of the weights file.

Two rules carry their weight regardless of distribution:

- **R-14.1a** A model's SHA-256 and version are recorded in `job.json` (§9.2) and any mismatch on
  resume discards the dependent artifacts. Without this, a retrained model silently mixes with output
  produced by its predecessor and the result is unexplainable.
- **R-14.1b** Model versions are immutable. A retrained model gets a new version directory; it never
  overwrites an existing one. Overwriting weights in place is how a benchmark becomes irreproducible.

### 14.2 Provenance instead of distribution

There is no download path and no update channel. Models arrive by being trained locally
(`training/train_detector.py`, `training/train_restorer.py`), which writes the model directory and its
`metadata.json` in one step. The training run's commit SHA, dataset version, and configuration are
recorded in `metadata.json` so that any output can be traced back to the run that produced it.

Third-party initialization weights (D-04) are stored outside `models/`, under a path that is
git-ignored and clearly marked non-distributable, and referenced by hash from `metadata.json`.

### 14.3 Missing or stale model handling before a job starts

Still required, for a different reason than in v2.0: the failure mode is now "the model has not been
trained yet" or "settings reference a version that no longer exists", not "the user has not downloaded
it". The cost of getting it wrong is identical — minutes of decoding followed by a failure.

- The host validates that every model the resolved settings require exists locally, at the expected
  version, with a matching hash, **before** the job starts, and reports `E4001`/`E3001` immediately if
  not.
- **The resolution result must reach execution.** If `auto` resolves to `rest-quality` and the host
  then sends `auto` to the worker, the worker applies its own default and the user gets a different
  model than the one that was validated. The **resolved snapshot** is what is sent, and it is a copy —
  resolving models never mutates the user's saved settings.
- If hardware detection or model-status lookup fails, proceed without blocking; the worker still
  reports the problem, later.

**AC-14.3** — `T-MODEL-RESOLVE-SNAPSHOT-01` asserts the worker receives a concrete model id, never
`auto`, whenever resolution succeeded. `T-MODEL-MISSING-01` asserts a missing or hash-mismatched model
fails before any decoding begins.

### 14.4 Python runtime

A local virtual environment (`.venv/`) created by `scripts/setup-worker.ps1` from
`worker/requirements.lock`, plus a GPL FFmpeg build (D-12) under `tools/ffmpeg/`. Both are
git-ignored; the script is the source of truth for their contents.

No embedded distribution, no packaging, no clean-room install test. The one requirement that remains:
`scripts/check-environment.ps1` (§4.5) shall detect a missing or mismatched environment and print the
exact command to fix it, because a stale `.venv` after a dependency change produces failures that look
like code bugs.

**AC-14.4** — `T-ENV-CHECK-01` covers a missing venv, a lockfile/venv version mismatch, and a missing
FFmpeg, each with its remediation command.

---

## 15. Quality Modes

| | Fast | Balanced (default) | Quality |
| --- | --- | --- | --- |
| Temporal window `K` | 3 | 5 | 7-9 where useful |
| Backend | `FastRestorationBackend` | `Balanced` | `Quality` |
| ROI processing resolution | native | native | native or 2x where the model benefits |
| Alignment | cheap flow | full flow | full flow + refinement |
| Temporal consistency | post-stabilization only | recurrent + post | recurrent + previous-output guidance + post |
| Target VRAM | <= 4 GB | <= 8 GB | <= 12 GB |

Quality mode shall not be implemented as "upscale the ROI more". It selects the whole strategy:
window, backend, alignment, and consistency mechanisms.

**AC-15** — Each preset is a named, versioned settings bundle in `Domain`, and a test asserts the
three presets differ in at least window, backend, and consistency mechanism. `T-PRESET-DISTINCT-01`.

---

## 16. Milestones, Gates, and Kill Criteria (closes G-19)

Every phase has an **entry** condition, an **exit** condition that is machine-checkable, and where
applicable a **kill** criterion. A phase does not start until the previous phase's exit condition is
recorded in `docs/`.

### Phase 0 — Foundations and the feasibility gate (1-2 weeks)

**Entry:** none.
**Work:** `git init`; repo skeleton (§4.3); `scripts/check-environment.ps1`; Python venv + **GPL FFmpeg build with x265** (D-12);
decode → passthrough → encode PoC with PTS correctness (§5.1.7); the degradation generator (§11.3);
`eval-v1` v0 (§11.5); the **multi-frame vs. single-frame experiment** using a third-party VSR baseline
internally (§4.2) and a classical multi-frame solver.
**Exit:** `docs/phase0-report.md` contains the §1.4.3 verdict with measured deltas **and the measured
noise floor**; a video round-trips with identical frame count and PTS.
**KILL:** verdict `KILL` → stop and re-scope per §1.4.3.

### Phase 1 — Detection and tracking (2-3 weeks)

**Entry:** Phase 0 PASS or MARGINAL.
**Work:** detector training (D-03), the hard-negative corpus, tracking (§5.3), profile estimation
including **grid anchoring** (§5.4.4), the `analyze` protocol command.
**Exit:** §12.1 targets met on `eval-v1` **including** the false-positive rates; §5.4.4 anchoring
classification AC met; `analyze` produces the region summary end to end.
**KILL:** FP rates cannot be met without recall collapsing below 0.75 → the detector is not viable as
an unsupervised feature; re-scope to user-assisted region selection.

### Phase 2 — Restoration (3-5 weeks; the long pole)

**Entry:** Phase 1 exit.
**Work:** train the degradation-conditioned model (D-04); alignment (§5.7); router (§5.8); temporal
consistency (§5.10); confidence (§5.9.4); the conditioning ablation (§5.9.3); **and the v3.0 addition
— the initialization comparison.**
**Exit:** on `eval-v1`, multi-frame beats single-frame per band per §12.3, temporal AC of §5.10 met,
confidence monotonicity AC met.

**Initialization comparison (v3.0, D-04, D-11).** Train two runs to convergence — one initialized from
third-party VSR pretrained weights, one from scratch — and compare under §13.5. Record the result in
`docs/phase2-init-report.md`.

- Initialized wins beyond the noise floor → keep it, and record in `metadata.json` that the model
  carries an inherited license (§14.1, §18 R-05).
- Difference inside the noise floor → **drop the initialization.** Carrying a contaminating license
  for an unmeasured benefit is the worst available outcome.

v1.0's 2-4 week estimate assumed adopting a pretrained model wholesale. D-04 keeps this the longest
phase even with initialization, because the degradation prior still has to be learned.

### Phase 3 — Engine pipeline (2-3 weeks)

**Entry:** Phase 2 exit.
**Work:** the full worker pipeline (§3.1), scheduler (§5.13), VRAM ladder (§5.14), checkpoint/resume
(§9), protocol v1.0 complete (§8), error codes (§10).
**Exit:** `T-SCHED-*`, `T-VRAM-*`, `T-RESUME-*`, `T-PROTO-*` green; a 1-hour file completes
end-to-end on the dev machine.

### Phase 4 — Desktop MVP (2 weeks)

**Entry:** Phase 3 exit.
**Work:** WPF host, job control, preview, presets, settings, diagnostics, model download (§14.3).
**Exit:** the §17 checklist passes end to end, driven from the UI only.

### Phase 5 — Optimization (1-2 weeks; **halved in v3.0**)

**Entry:** Phase 4 exit.
**Work:** tiling, adaptive window tuning, batching, GPU-resident decode/encode, `torch.compile` /
CUDA graphs. **Dropped under D-09/D-11:** ONNX export, TensorRT engines, the §13.6 export parity
gate, the DirectML path. The native engine (§3.4) remains available and remains unjustified until
measurement says otherwise.
**Exit:** §6.1 Optimized targets met on the reference config, per encoder profile, with ROI coverage
reported.

Note that this phase is now **optional**. If the MVP tier of §6.1 is fast enough in practice for the
one person using it, the correct decision is to skip Phase 5 and spend the time on restoration
quality instead. Record the choice either way.

### Phase 6 — Hardening (0.5-1 week; **mostly dissolved in v3.0**)

**Entry:** Phase 5 exit, or Phase 4 exit if Phase 5 was skipped.
**Work:** crash recovery, long-run memory tests (§6.2), `THIRD_PARTY_NOTICES.md` completion, the §22
matrix on the one supported machine, `README.md` with the §2.3 C-1 statement.
**Dropped under D-11:** installer, code signing, clean-room install test, cross-GPU acceptance matrix.
**Exit:** every AC in this document has a passing test, an explicit waiver, or an "inactive under
D-11" marker.

---

## 17. MVP Definition (checkable)

The MVP is complete when, **driven from the UI only**, on the dev machine:

| # | Criterion | Verified by |
| --- | --- | --- |
| 1 | Load a supported video by drag and drop; metadata displayed | `T-UI-DND-01` |
| 2 | Automatically detect mosaicked regions with FP rates within §5.2.5a | `T-DET-FPRATE-01` |
| 3 | Track moving/resizing regions with stable IDs | §12.2 targets |
| 4 | Estimate block geometry and **grid anchoring** | `T-PROFILE-BLOCK-EST-01`, `T-PROFILE-ANCHOR-01` |
| 5 | Build a valid temporal window with scene-cut truncation | `T-WINDOW-POLICY-01..06` |
| 6 | Restore using at least one temporal backend | §12.3 |
| 7 | Fall back to single-frame and to pass-through with recorded reasons | `T-ROUTER-REASON-01` |
| 8 | Blend without rectangular seams | `T-BLEND-NO-RECT-01` |
| 9 | Preserve audio bit-identically and keep A/V sync | `T-IO-AUDIO-COPY-01`, `T-IO-PTS-*` |
| 10 | Export H.264/H.265; untouched regions visually transparent in **both** encoder profiles | `T-QUALITY-NULLRUN-01`, `T-IO-ENCODE-SELECT-01` |
| 11 | Use NVIDIA GPU acceleration, honestly reported | `T-PROBE-CUDA-HONEST-01` |
| 12 | Process a 1-hour file within the §6.2 memory bounds | `T-LONGRUN-MEMORY-01` |
| 13 | Recover from cancel, from worker crash, and from GPU OOM | `T-SCHED-CANCEL-01`, `T-RESUME-KILL-01`, `T-VRAM-LADDER-01` |
| 14 | Before/after preview including split view | `T-UI-PREVIEW-COALESCE-01` |
| 15 | Report performance and per-job confidence; export metadata marks output synthetic | `T-BENCH-SCHEMA-01`, `T-EXPORT-META-01` |

---

## 18. Risk Register (closes G-20)

| ID | Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- | --- |
| R-01 | **Multi-frame gives no real benefit on representative material** (§1.4) | Existential — the product's differentiator vanishes | Medium | Phase 0 gate before any Phase 2 spend; re-scope path defined | Tech lead |
| R-02 | **In-house restoration model underperforms** (D-04) | High — Phase 2 is the long pole | Medium-High | Keep the internal third-party baseline as the comparison floor; ship `Fast`/`Balanced` first; the backend interface allows a later swap | ML |
| R-03 | **Detector domain gap**: works on synthetics, fails on real mosaics | High | High | Recompression mandatory in training and eval (§11.3); `real-mosaic` recall check every phase; hard negatives (§11.4) | ML |
| R-04 | **False positives alter clean footage** | High — worst user-visible failure | Medium | §5.2.5 requirements, temporal confirmation, `analyze` preview before processing | ML + UX |
| R-05 | **The one-way door closes.** Under D-11 the GPL encoder and the S-Lab-derived restoration weights are legal, but a later decision to distribute — even one copy to one person — forces open-sourcing in full, buying licenses, or retraining from clean-room components | High, and it grows with every phase | Low, but the cost is asymmetric and the decision is easy to make casually | §2.4's reversal trigger; contaminants isolated behind interfaces (R-4.2a, `T-CONTAMINANT-ISOLATION-01`); `THIRD_PARTY_NOTICES.md` maintained anyway; Phase 2's initialization comparison so the dependency is only taken if it is measurably earning its place | Owner |
| R-06 | **Python engine too slow to be pleasant** | Medium | Medium | Two-tier targets (§6.1); Phase 5 optional and native path reserved; boundary designed for swap (§3.4) | Eng |
| R-07 | ~~Backend fallback (DirectML) cannot run the model~~ | — | — | **Closed by D-11.** DirectML dropped; CUDA + CPU only (§5.17) | — |
| R-08 | ~~TensorRT engine portability / first-run build time~~ | — | — | **Closed by D-09.** Export dropped entirely | — |
| R-12 | **x265 `slow` makes the Quality profile CPU-bound and slower than expected** (new in v3.0) | Low — it is a speed/quality trade the user picks per job | Medium | Two encoder profiles (§5.1.4); per-profile targets in §6.1; measure the actual transparency-vs-bitrate gap rather than assuming x265 wins by a specific margin | Eng |
| R-09 | **VFR/odd-container A/V drift** | Medium | Medium | Output timing rule (§5.1.7) with frame-count and PTS assertions | Eng |
| R-10 | **Full re-encode degrades untouched picture** | Medium — reads as "the tool ruined my video" | High if unaddressed | §5.1.8 transparency threshold and null-run test; pass-through stream copy; smart-cut later | Eng |
| R-11 | **Scope creep from "make it better" quality work with no measurement** | Medium | High | §13.5 noise-floor protocol; §12.7 harness-only numbers | Tech lead |

---

## 19. Key Engineering Principles

1. **Segmentation first** — restoration boundaries are mask-based, not rectangle-based.
2. **Analyze before restoring** — estimate degradation parameters, including grid anchoring, before
   choosing a strategy.
3. **Observable information first** — exploit neighbouring-frame evidence before model hallucination,
   and say which one produced the pixels.
4. **Temporal quality is first-class** — single-frame visual quality alone is insufficient.
5. **Model abstraction** — never couple the product permanently to one detector, one VSR, or one op.
6. **GPU-resident where it pays** — minimize transfers, but measure before rewriting for it.
7. **Adaptive execution** — window, tile, model, and memory respond to source and hardware.
8. **PTS correctness** — synchronization beats frame-index convenience.
9. **Recoverability** — long jobs checkpoint, resume, and fail gracefully.
10. **Measurable quality** — detection, tracking, restoration, temporal consistency, boundary quality,
    performance, and memory are evaluated independently, by a checked-in harness.
11. **Measure the noise floor before claiming an improvement** (§13.5). *(New in v2.0; it is the
    difference between engineering and storytelling.)*
12. **Honest capability reporting** — "available" means tested on this machine, not "a driver says a
    device exists" (§8.3, §5.17).

---

## 20. Open Questions for the Product Owner

Q2 and Q3 were answered in v3.0 and are retained as closed for the record. The rest are open; Phase 0
can start without any of them.

| # | Question | Default if unanswered |
| --- | --- | --- |
| Q1 | **Do you accept D-01 (Python worker for MVP, native engine deferred)?** To override, replace D-01 with the v1.0 native-first plan and add ~3-4 weeks of Phase 0 toolchain and ABI work | **Open.** Proceeding with D-01 |
| Q2 | Is distribution intended? | **ANSWERED (v3.0): no. Personal use only** → D-11, §2.4. Closed |
| Q3 | Which clean-video sources may be used for training? | **ANSWERED (v3.0):** unblocked by D-11 — personal non-commercial training. Origins still recorded (§11.2), §2.3 C-2 still binds. Closed |
| Q4 | Is batch/folder processing needed for v1, or is single-job acceptable? (§2.2) | Single job |
| Q5 | Is 4K a v1 requirement? It changes tiling and VRAM work materially | v1 handles 4K at reduced throughput; not a performance target |
| Q6 | Localization scope beyond English? | English only, strings externalized. Low value under D-11 — kept because it costs nothing now and costs a refactor later |
| Q7 | Is a "detect and report only" mode worth building as a fallback if R-01 fires? | Yes — it is the re-scope target in §1.4.3 |
| Q8 | **New in v3.0:** is Phase 5 (optimization) wanted at all, or is the MVP throughput tier good enough for one user? | Decide after Phase 4 with measured numbers in hand, not before (§16 Phase 5) |

---

## 21. Edge Cases

Each maps to a test. This is v1.0's list, retained, with owners assigned.

| # | Edge case | Handling | Test |
| --- | --- | --- | --- |
| 1 | Mosaic partially outside the frame | Clamp + reflect/replicate padding (§5.5.2) | `T-ROI-EDGE-01..08` |
| 2 | Mosaic appears/disappears suddenly | Temporal confirmation (§5.2.5b), track TENTATIVE→ACTIVE | `T-DET-CONFIRM-FRAMES-01` |
| 3 | Mosaic changes size | Kalman `w,h` velocity; adaptive padding | `T-TRACK-LAG-01` |
| 4 | Block size changes mid-track | Profile change requires 3-frame evidence (§5.4.3) | `T-PROFILE-STABILITY-01` |
| 5 | Overlapping regions | Mask union, single ROI when IoU high | `T-DET-OVERLAP-01` |
| 6 | Fast motion | K→3, motion-adaptive smoothing | `T-WINDOW-POLICY-03` |
| 7 | Camera cut | Window truncation, track reset (§5.12) | `T-SCENE-CUT-01` |
| 8 | Camera flash | Explicitly **not** a cut | `T-SCENE-FLASH-01` |
| 9 | Severe motion blur | Alignment confidence excludes bad neighbours | `T-ALIGN-OCCLUSION-EXCLUDE-01` |
| 10 | Heavy compression artifacts | Trained-in via §11.3 recompression | §12.3 per-CRF stratum |
| 11 | Detector miss for a few frames | `max_missing_frames`, mask propagation | `T-TRACK-MISSING-01` |
| 12 | Complete occlusion | OCCLUDED state, restoration suspended | `T-TRACK-OCCLUDE-01` |
| 13 | ROI below model minimum | `E4004` → pass-through | `T-ROI-TOO-SMALL-01` |
| 14 | ROI beyond VRAM budget | Tiling ladder (§5.14) | `T-VRAM-LADDER-03` |
| 15 | VFR input | Output timing rule (§5.1.7) | `T-IO-PTS-VFR-01` |
| 16 | Corrupt/missing frames | `E2002` skip + log, or `E2003` | `T-IO-CORRUPT-01` |
| 17 | Unsupported codec/profile | `E1003` at probe, before any work | `T-IO-CODEC-REJECT-01` |
| 18 | GPU OOM | Ladder, no partial composite | `T-VRAM-NO-PARTIAL-01` |
| 19 | Encoder failure | `E5001` → software fallback | `T-IO-ENCODE-FALLBACK-01` |
| 20 | Disk exhaustion | `E6001`, checkpoint preserved | `T-DISK-FULL-01` |
| 21 | User cancellation | Terminal result <= 500 ms, valid checkpoint | `T-SCHED-CANCEL-01` |
| 22 | Restart then resume | Fingerprint-checked resume (§9.3) | `T-RESUME-KILL-01` |
| 23 | **Object-anchored mosaic** (new) | Router forces single-frame, confidence Low | `T-PROFILE-ANCHOR-02` |
| 24 | **Zero detections in the whole file** (new) | Stream-copy pass-through, no re-encode | `T-IO-PASSTHROUGH-COPY-01` |
| 25 | **Source is already clean but user runs anyway** (new) | Same as 24, plus an explicit "nothing detected" summary | `T-UI-NOTHING-FOUND-01` |

---

## 22. Acceptance Test Matrix

| Axis | Values |
| --- | --- |
| Hardware | **RTX 3080 Ti 12 GB — the only supported machine (§4.5, D-11)**, plus CPU-only as a correctness path. Other GPUs are out of scope in v3.0 |
| Encoder profile | x265 slow (Quality, default), NVENC (Speed) |
| Resolution | 720p, 1080p, 1440p, 4K |
| Mosaic coverage | <5%, 5-15%, 15-30%, >30% |
| Block size band | <=4, 5-12, 13-24, >24 px (§1.4.2) |
| Grid anchoring | screen, object, mixed (§1.4.1) |
| Motion | static, slow, medium, fast |
| Duration | short clip, 10 min, 1 h, 2 h+ |
| Media | H.264 CFR, H.264 VFR, H.265, AV1 input, multi-audio, subtitle-bearing, rotation metadata, corrupt file |

Reporting requirement: results are reported **per cell**, and cells that were not run are marked
"not run" rather than omitted. An unstated gap reads as coverage.

---

## 23. Target Processing Flow

```text
Input Video
    |
FFmpeg demux ──> audio / subtitles / chapters ──────────────┐
    |                                            (stream copy)
Hardware decode (NVDEC / D3D11VA / software)                |
    |                                                       |
PTS-aware presentation-order scheduler (§5.13)              |
    |                                                       |
    +----------------------------+                          |
    |                            |                          |
Scene-cut / flash (§5.12)   Mosaic segmentation (§5.2)      |
                                 |                          |
                        ByteTrack + Kalman (§5.3)           |
                                 |                          |
              Mosaic profile + GRID ANCHORING (§5.4)        |
                                 |                          |
                     Temporal ROI stabilizer (§5.5)         |
                                 |                          |
                     Adaptive window builder (§5.6)         |
                                 |                          |
                     Temporal alignment (§5.7)              |
                                 |                          |
                  Restoration strategy router (§5.8)        |
                    /            |            \             |
           Multi-frame     Single-frame     Pass-through    |
                    \            |            /             |
                     Temporal consistency (§5.10)           |
                                 |                          |
                     Mask-aware blending (§5.11)            |
                                 |                          |
                     Confidence accumulation (§5.9.4)       |
                                 |                          |
              Encode: NVENC / software (§5.1.4, §5.1.8)     |
                                 |                          |
                              Mux <---------------------────┘
                                 |
                          Output Video
                    (+ synthetic / confidence metadata)
```

---

## 24. Final Product Principle

DeMosaic Studio shall not be implemented as:

`Mosaic ROI -> VSR -> Paste Back`

The intended architecture is:

`Detect -> Segment -> Track -> Analyze Degradation (incl. grid anchoring) -> Gather Temporal Evidence
-> Align -> Reconstruct -> Estimate Confidence -> Enforce Temporal Consistency -> Mask-Aware Blend
-> Encode`

And one addition v1.0 did not make, which this revision treats as equally fundamental:

**Measure whether the temporal evidence exists before building an architecture that assumes it, and
report honestly which pixels came from evidence and which came from a model.**
