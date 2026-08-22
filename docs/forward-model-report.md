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

The mask model is **implemented, tested, and switched off** — it engages at
`MASK_MODEL_MIN_NEIGHBOURS = 16`, which the window policy's cap of 9 never reaches. That is
deliberate: on the only content this project can evaluate end to end it costs 0.12 dB, and the
reason is quantified rather than guessed. It turns on by itself when the window policy is
re-measured.

## 6. What this changes

1. **The evaluation clip has to be replaced.** `artifacts/screen_input.mp4` and `screen_clean.mp4`
   exist now — content panning 4 px/frame past a mosaic that does not move. Every end-to-end
   comparison in this repository is against a clip in the regime the design says is hopeless.
2. **The window table is invalid.** Re-measure `WINDOW_BY_MOTION` and the cap against the corrected
   forward model before trusting any of D-16.
3. **Long windows are expensive.** 16+ neighbours is 8× the alignment work. Whether the answer is a
   longer window, evidence accumulated across frames instead of re-aligned per frame, or a learned
   restorer, is now a design question with numbers attached to it.

## 7. Limitations

- **The window sweep is one target frame on one clip.** It reproduces an arithmetic prediction,
  which is worth more than a single number, but it is still n=1.
- **The screen-anchored clip is synthetic in a second way**: the pan is a crop of a real frame, so
  the motion is a pure translation with no parallax, occlusion or object motion.
- **`MASK_MODEL_MIN_NEIGHBOURS = 16` is the crossover on that one clip.** It will move with mosaic
  size and motion — the arithmetic says the governing quantity is coverage, not neighbour count, and
  coverage is what a future version should measure at runtime.
- **Nobody has looked at the screen-anchored output**, because the pipeline cannot yet produce a
  meaningful one for it.
