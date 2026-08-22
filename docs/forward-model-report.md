# The forward model, the window, and what the evaluation clip could never show

**2026-08-23.** The plan was to make the forward model mask-aware, because a neighbour that saw
content the target has lost is the strongest evidence there could be and the model was throwing it
away. The model was made mask-aware. It made the pipeline **worse**, and finding out why turned up
something larger than the change itself.

Raw material: `scripts/eval_gate.py`, `artifacts/screen_*.mp4`, and the probes recorded below.

---

## 1. The change

The forward operator used to declare every pixel of every frame block-averaged:

```python
simulated = block_average(warped, spec, phase)
```

A mosaic covers **part** of a frame. Where a frame is mosaicked it observes block averages; where it
is not, it observes the scene directly, at full resolution. So:

```
A(x) = block_average(x)   where the frame is mosaicked
     = x                  where it is not
```

Measured on synthetic content with oracle alignment — a screen-fixed mosaic, content shifted through
it:

| block | shift | input | old model | mask-aware | gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 34.38 | 34.38 | **36.73** | +2.35 |
| 10 | 16 | 32.64 | 33.25 | **37.04** | +4.40 |
| 14 | 16 | 29.96 | 29.77 | **33.57** | +3.61 |

The gain grows with the shift, exactly as the physics says it must.

## 2. In the pipeline it lost

| arm | inside the region | vs input |
| --- | ---: | ---: |
| observation fix (yesterday) | 32.469 | −0.385 |
| + mask-aware model | 32.226 | **−0.628** |

Four candidate explanations, each measured rather than argued.

**Mask error?** No. ±2 px of dilation or erosion still gives +3.7 to +4.5 dB; only a 6 px dilation
turns it negative. And on real frames the detector's mask scores IoU 0.81–0.95 against the exact
ellipse.

**The aligner locking onto the mosaic?** No. The flow inside the mosaic reads 15.45 px against a
true 16, not zero.

**Sub-pixel alignment error?** No. A deliberate 1 px error still gives +4.04 dB, and fractional
oracle shifts score the same as integer ones.

**The solver diverging?** **Yes, partly.** With the real aligner, by iteration count:

| iterations | 2 | 5 | 10 | 20 | 40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vs input | +0.44 | **+0.58** | +0.53 | −0.18 | **−2.79** |

The pipeline ran 20. With a dense flow the to-target and to-neighbour fields are estimated
*separately*, so the forward warp and the back warp are not exact inverses and the iteration is not
a descent on a consistent objective. The old stopping rule watched the **change** in the residual,
which divergence never produces. Both solvers now keep the iterate with the lowest residual and
stop when it stops being the latest one: 40 iterations went from −2.79 dB to **+0.57**.

That fixed part of it. The rest was not a defect at all.

## 3. What the evaluation clip actually contains

Mask-aware modelling can only use content a neighbour saw **unmosaicked**. Measured on the ladder
input, with the exact ellipse and motion from aligning the clean frames:

| | |
| --- | ---: |
| content motion | 0.1 – 0.6 px/frame |
| mask motion | 3 px/frame |
| lost content a neighbour saw clean, t−1 | **1.6%** |
| t−2 | **3.5%** |

**The mosaic slides over near-static content.** The same picture is covered in every frame. That is
the object-anchored regime §1.4.1 predicts to be unrecoverable and the Phase 0 gate measured at
−0.79 to −1.50 dB — and **every end-to-end number this project has produced was measured on it.**

An earlier reading of 40% / 55% came from deriving the mask as `|clean − degraded| > 12`, which
marks only the pixels the block average moved far. That is a holey mask, not the region, and it
inflated the diversity by more than an order of magnitude. Deriving ground truth from a threshold
on a difference is not ground truth.

## 4. The arithmetic, and the window

A neighbour `d` pixels away exposes a crescent of roughly `2·d·ry` out of an ellipse of `π·rx·ry` —
about **`2·d / (π·rx)`** of what the target lost. For a 300 px mosaic at 4 px/frame that is 1.7% per
neighbour. Measured on a purpose-built screen-anchored clip: 1.8% at t−1, 4.0% cumulative at t−2.
The arithmetic holds to a tenth of a percent.

So the evidence accumulates **with the window**, and a window of 3 sees about 6% of the region no
matter how good the solver is. Measured on that clip, real aligner throughout:

| neighbours | coverage | old model | mask-aware | vs input |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 3.6% | 23.22 | 23.24 | −0.88 |
| 4 | 7.3% | 23.24 | 23.22 | −0.90 |
| 8 | 14.6% | 22.69 | 22.62 | −1.50 |
| 16 | 28.1% | 22.16 | **24.68** | **+0.56** |
| 24 | 49.4% | 21.65 | **26.04** | **+1.92** |

**Note the directions.** More evidence makes the old model *worse* and the mask-aware model
*better*. That is the signature of a forward model that has started describing the data. At 24
neighbours the two differ by **4.4 dB**.

### This invalidates the window table

§5.6's `K ∈ {3,5,7,9}` and D-16's "K=3 ≥ K=5 everywhere" were measured with the old forward model —
the one under which extra frames genuinely do hurt, as the middle column above shows. The window
policy is calibrated against a model the code no longer uses.

`WINDOW_BY_MOTION`, `PRESET_MAX_WINDOW` and the cap of 9 all need re-measuring. That is a Phase 2
scale experiment, not an edit, and it has a real cost: 24 alignments per region per frame against
the 2 the pipeline does now, on a pipeline already running at 0.63 fps.

## 5. What shipped

| arm | inside the region | vs input | SSIM | frames improved |
| --- | ---: | ---: | ---: | ---: |
| the mosaicked input — the bar | 32.854 | — | 0.9144 | — |
| start of the session | 32.095 | −0.759 | 0.9041 | 4% |
| observation fix | 32.469 | −0.385 | 0.9094 | 42% |
| **+ non-diverging solver — shipped** | **32.543** | **−0.311** | 0.9117 | **42%** |
| + mask-aware model as well | 32.425 | −0.429 | 0.9127 | 45% |

The mask model was **implemented, tested, and switched off** at this point — it needed about 16
neighbours to pay and the window policy capped K at 9. On the only content the project could
evaluate end to end it cost 0.12 dB.

> **Superseded the same day.** §8 below replaces the window with an accumulator, which reaches that
> depth at O(1) cost per frame. The mask model is on, and the numbers in this section are the last
> ones the batch form produced.

## 6. What this changes

1. **The evaluation clip has to be replaced.** `artifacts/screen_input.mp4` and `screen_clean.mp4`
   exist now — content panning 4 px/frame past a mosaic that does not move. Every end-to-end
   comparison in this repository is against a clip in the regime the design says is hopeless.
2. **The window table is invalid.** Re-measure `WINDOW_BY_MOTION` and the cap against the corrected
   forward model before trusting any of D-16.
3. **Long windows are expensive.** 16+ neighbours is 8× the alignment work. Whether the answer is a
   longer window, evidence accumulated across frames instead of re-aligned per frame, or a learned
   restorer, is now a design question with numbers attached to it. §8 answers it: the accumulator.

## 7. Limitations

- **The window sweep is one target frame on one clip.** It reproduces an arithmetic prediction,
  which is worth more than a single number, but it is still n=1.
- **The screen-anchored clip is synthetic in a second way**: the pan is a crop of a real frame, so
  the motion is a pure translation with no parallax, occlusion or object motion.
- **16 neighbours is the crossover on that one clip.** It will move with mosaic size and motion —
  the arithmetic says the governing quantity is coverage, not neighbour count, and coverage is what
  a future version should measure at runtime. The accumulator sidesteps the threshold by reaching
  any depth cheaply, but it does not make the quantity go away.
- **Nobody has looked at the screen-anchored output** at the time of writing. §8 does.


---

## 8. The cost, and what replaced the window

**2026-08-23, later.** The window had to grow to about 17 for the corrected model to pay. Measured
before changing anything:

| neighbours | solve | align | per frame | fps |
| ---: | ---: | ---: | ---: | ---: |
| 2 (as shipped) | 294 ms | 219 ms | 634 ms | 1.58 |
| 8 | 1104 ms | 874 ms | 2100 ms | 0.48 |
| 16 | 2217 ms | 1748 ms | 4086 ms | **0.24** |
| 24 | 3444 ms | 2622 ms | 6188 ms | **0.16** |

17× to 25× short of §6.1's ≥4 fps, and **the solver costs more than the alignment** — so "8× the
alignment work" understated it. Memory was never the constraint: 183 MB for K=17.

### Carrying the evidence instead

One estimate per track, warped by a single frame-to-frame flow each frame, with the new observation
folded in. One alignment and one warp regardless of how far back the evidence goes.

| | ms/frame | fps |
| --- | ---: | ---: |
| pipeline as shipped (K=3) | 634 | 1.58 |
| **accumulator, unbounded history** | **231** | **4.33** |
| batch at 16 neighbours | 4287 | 0.23 |

It is also *better*, for a reason already in `docs/phase2-alignment-report.md` §3 — shorter
baselines align better. The batch form reaches across the whole window; this chains one-frame
alignments. On 24 frames of history: batch **26.04 dB**, accumulator **28.96 dB**.

### End to end

| | inside the region | vs input | SSIM | frames improved |
| --- | ---: | ---: | ---: | ---: |
| **screen-anchored clip** | | | | |
| mosaicked input | 25.306 | — | 0.7756 | — |
| batch, K=3 | 25.126 | −0.180 | 0.7675 | 51% |
| **accumulator** | **27.110** | **+1.804** | **0.8178** | **76%** |
| **object-anchored ladder clip** | | | | |
| mosaicked input | 32.854 | — | 0.9144 | — |
| batch, K=3 | 32.543 | −0.311 | 0.9117 | 42% |
| **accumulator** | **34.358** | **+1.504** | **0.9248** | **97%** |

It wins on the object-anchored clip too, which no window of 3 could: 1.6% of the region per frame
is nothing to a window and a third of the region to a chain of 24.

### Looking at it

`artifacts/look_screen.png`. The blocks are **gone** and the structure is back — the vertical bar,
the diagonal edge, the bright region all return. The replacement carries **directional streaking**
along the motion, which is what repeated warping leaves behind.

That streak is invisible to PSNR: quality still improves monotonically from 8 to 24 frames of
history while the streaking gets worse. It is the next thing to attack and it needs an eye or a
perceptual metric, not this one.

### Two defects the wiring found

- **`target_index` indexes the rolling history buffer**, which stops advancing once the buffer is
  full. Used as a frame number it restarted the evidence chain on every single frame, while the
  routing counts looked plausible and no test failed.
- **`Roi.bounds` is not the rectangle `Roi.crop` returns** — reflect padding is part of the crop.
  Comparing the wrong rectangles mismatched shapes by exactly the padding. `Roi.crop_bounds` now
  names the one that means "what the crop covers".

### Limitations, again

- **Two clips, both synthetic.** The screen-anchored one is a crop panning across a real frame, so
  the motion is a pure translation with no parallax or occlusion.
- **The streaking has not been quantified**, only seen.
- **4.33 fps is the measured ROI**; the ladder clip's larger ellipse at 1920×800 runs at 1.0 fps.
  Detection (119 ms) and one alignment (107 ms) are the whole cost — the accumulator itself is 6 ms.


---

## 9. Chasing the streaking, and what was actually wrong

**The characterisation in §8 was an over-reading.** The error's high-frequency horizontal energy is
*lower* in the output (0.352) than in the mosaicked input (0.391), and the offline prototype — no
detector, no ROI, no blending, no encoder — produces the same texture. Most of what looked like
streaking is residual block structure and partially recovered detail: an incomplete restoration
rather than an added artefact.

Two ablations pinned the loop down, because it only does two things:

| | PSNR | ripple |
| --- | ---: | ---: |
| the mosaicked input | 24.12 | 0.391 |
| **warping alone, from a clean frame** | **13.19** | 0.146 |
| folding alone, nothing moving | 24.11 | 0.394 |
| both, as shipped | 28.96 | 0.352 |

Chaining 24 warps of a clean frame costs 11 dB; folding with nothing moving costs nothing. The warp
is the damage and **the fold is what keeps the chain honest** — it re-anchors the estimate to a real
observation every frame.

### The real defect: quality peaks with depth

| depth | 4 | 8 | 12 | 16 | 24 | 32 | 48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| screen-anchored | +3.28 | +4.07 | +4.07 | **+4.10** | +3.78 | +3.59 | +3.27 |
| object-anchored | +1.71 | +2.49 | +2.95 | +3.23 | +3.31 | **+3.35** | +2.72 |

Averaged over four target frames each. The oldest evidence in a chain of N has been warped N times
and carries N frames of the flow's error.

### The fix: forget on a horizon

A depth **cap** is a reset — it discards the chain and starts at zero, which oscillates. Decaying
the carried estimate towards the current observation by `1/32` per frame bounds the horizon and
keeps the chain. Measured at depth 48:

| horizon | off | 64 | **32** | 16 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen | +3.27 | +4.49 | **+5.28** | +5.97 | +6.01 |
| ladder | +2.72 | +2.85 | **+2.81** | +2.54 | +1.98 |

End to end:

| | before | after |
| --- | ---: | ---: |
| screen-anchored | +1.804 dB, 76% of frames | **+2.296 dB, 84%** |
| object-anchored ladder | +2.087 dB, 99% | **+2.217 dB, 100%** |

And the pipeline's gain rises with the chain's depth exactly as designed: **+0.84 dB** at depth 0–4,
**+1.56** at 5–19, **+2.15** at 20 or more.

### Four fixes measured and rejected

- **Anchor the estimate, compose the flows** so the estimate is never cumulatively resampled — the
  batch solver's arrangement. **Much worse**: 11.51 dB at depth 24 against 28.96. Composed flows
  drift, and a drifting flow misplaces everything at once rather than blurring it a little.
- **Skip the warp when nothing moves.** The static tail went from +1.64 dB to −1.11. Folding keeps
  paying without motion because it re-anchors; freezing removes that.
- **Decay by forward–backward flow confidence** instead of uniformly — costs 2.24 dB.
- **Damp the fold** (step 0.5, 0.25) — costs 0.9 and 3.2 dB.

Only the uniform horizon helped, and it is the only one of the four that changes **how long
evidence lives** rather than how strongly each frame speaks.

### What is left

The block structure inside the region survives even at depths where coverage is complete. That is
not a coverage problem: the fold restores each block's **mean** and has nothing to say about the
arrangement *within* a block. Closing that is what a learned restorer is for.

### Limitations

- **Two clips, both synthetic**, and the horizon's optimum differs between them (8–16 against
  32–64). A frame count is therefore not the governing quantity; accumulated warping error, which
  grows with motion per frame, is the candidate and two clips cannot confirm it.
- **The ripple statistic is a share of horizontal spectral energy above the block fundamental.** It
  separates the arms usefully but has no absolute meaning.
