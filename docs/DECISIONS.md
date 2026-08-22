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
