# Calibrating the gate found a bug in the restorer

**2026-08-22.** The task was to calibrate `minRestorationConfidence`, which had existed since v3.1,
defaulted to 0.0 (off), and had never been measured. Calibrating it required per-region ground
truth, and the ground truth said something the gate could not fix.

Raw material: `scripts/eval_gate.py`, `docs/gate-calibration.json`,
`docs/gate-calibration-fix.json`, `docs/untouched-decomposition-*.json`.

---

## 1. The calibration, before anything was fixed

214 applied restorations on the ladder clip, scored on the pixels each one actually changed,
against the clean original. Weighted by pixels touched, because the output is judged on picture and
not on regions.

| | |
| --- | ---: |
| restorations that helped | **20 of 214** (154,474 px of 4.2 M) |
| ungated, as the pipeline ran | **−1.4808 dB** |
| oracle — apply only where it helped | **+0.0180 dB** |

Split by whether the region was real:

| population | weighted dB | regions | helped |
| --- | ---: | ---: | ---: |
| on the real mosaic | **−1.2045** | 95 | 3% |
| on clean picture (false positives) | −2.9820 | 119 | 14% |

**Every path was negative** — multi-frame −1.41 dB, single-frame −2.12 dB, object-anchored
−3.31 dB. There was no good side of any threshold to select for. A gate cannot choose between
harmful and harmful.

## 2. What that pointed at

The single-frame number is the one that does not fit. Single-frame restoration needs no alignment,
so "alignment is hard" cannot explain it. Probing the solver directly:

| | |
| --- | --- |
| one observation, true grid | **+0.00 dB**, residual 0 |
| one observation, deliberately wrong block size | **+0.00 dB** |
| one observation, deliberately wrong phase | **+0.00 dB** |

Back-projection with a single observation is a **no-op by construction**: the residual has zero
mean inside every block, so its block average — the back-projection — is identically zero. The
solver was innocent, and it was innocent even when lied to about the grid.

So the damage came from around it. It did:

```python
reconstruct([Observation(block_average(crop_target, profile, phase), 0.0, 0.0)], ...)
```

**It applied the forward operator to something that was already an observation.** The frame already
contains the block averages where the mosaic is; `block_average` is what produced them. Applying it
again did two things:

1. **Re-quantised the mosaic onto the *estimated* grid**, shifting every block whenever the phase
   estimate was off — which is exactly what the output looked like.
2. **Destroyed the clean picture inside the ROI rectangle but outside the mask.** The ROI is a
   rectangle around an irregular region; everything in the corners is picture the pipeline had no
   business touching, and the dilation-and-feather blend then composited it back.

The multi-frame path did the same to every neighbour crop.

## 3. Undoing it

One variable, same clip, same settings.

| | before | after |
| --- | ---: | ---: |
| pipeline's share of the untouched-region loss | 1.63 dB | **0.26 dB** |
| damage on the pixels it altered outside the region | −4.21 dB | **−0.52 dB** |
| restorations that helped | 20 of 214 | **68 of 208** |
| ungated weighted | −1.4808 dB | **−0.8249 dB** |
| oracle | +0.0180 dB | **+0.0921 dB** |

And for the first time a signal separated: thresholding `confidence` at 0.88 scored **+0.0761 dB**,
98.3% of the oracle.

## 4. Then the gate could not express it

Setting `minRestorationConfidence` to the calibrated 0.88 restored **nothing at all**. Two defects,
both invisible until the gate was measured rather than reasoned about:

**The threshold was unreachable.** Release required `confidence > threshold + margin`, and the
confidence formula tops out at `0.25 + 0.35 + 0.4 · blockPenalty` — 0.90 for a 10 px mosaic. No
threshold above 0.85 could ever open the gate, and one above 0.90 silently meant "never restore".
The margin now guards the **closing** side, so a threshold means what it says.

**A track started open.** It took three consecutive low-confidence frames to engage, so every new
track was restored for two frames regardless of its confidence — which is how this was found: a
gate set above every reachable confidence still let two frames per track through. A track now
starts gated. Restoration is an intervention; it takes evidence to begin, not evidence to stop.

**And the gate was being fed a raw signal.** Its parameter is named `smoothed_confidence`; the
pipeline passed the per-frame value. The gate is per track and sticky in both directions, so one
long track (95 of the 208 regions here) opened on a run of good frames and then coasted through the
bad ones. Measured:

| fed to the gate | best threshold | weighted dB | kept |
| --- | ---: | ---: | ---: |
| raw per-frame confidence | 0.91 | **0.0000** (withholds everything) | 0 |
| smoothed, α = 1/2 | 0.89 | +0.0489 | 14 |
| **smoothed, α = 1/3 (the gate's own window)** | **0.88** | **+0.0511** | 18 |
| per-region ideal, no hysteresis — flickers | 0.88 | +0.0751 | 47 |

`ConfidenceSmoother` now damps over the same window the gate reasons about, in both languages, with
parity cases.

## 5. End to end

| arm | inside the region | vs input | SSIM | frames improved |
| --- | ---: | ---: | ---: | ---: |
| the mosaicked input — the bar | 32.854 | — | 0.9144 | — |
| before the fix | 32.095 | −0.759 | 0.9041 | 4% |
| after the fix, ungated | 32.469 | −0.385 | 0.9094 | 42% |
| **after the fix, gated at 0.88** | **32.872** | **+0.018** | 0.9140 | **51%** |

Outside the region, the pipeline's own share of the loss fell from **1.63 dB to 0.02 dB**.

**+0.018 dB is not a restoration.** It is break-even. Looking at the frames says the same thing: the
mosaic is still plainly there, its block edges slightly softer than the input's. What changed is
that the pipeline stopped *damaging* the picture — inside the region, outside it, and in the halo
around it.

## 6. What this changes

1. **The restorer's job is still entirely ahead.** Four earlier measurements said multi-frame gains
   little; this one says the pipeline was also spending a dB and a half on a bug. Those are
   different problems and only the second is now fixed.
2. **The gate is worth having, and its calibration is n=1.** 0.87 scores −0.60 dB and 0.88 scores
   +0.05 dB — a knife-edge on a single synthetic clip. **The default stays 0.0.** Shipping 0.88
   would be fitting one ellipse on one film.
3. **Calibrate against the thing you ship.** The idealised per-region sweep picked an operating
   point the real gate could not reach and would have been shipped as "calibrated". The
   shipped-gate arm in `eval_gate.py` exists so that cannot happen again.

## 7. Limitations

- **One clip, one synthetic mosaic, one block size.** Everything above.
- **PSNR and SSIM only, and PSNR is a weak proxy for a restoration.** A restorer that invents
  plausible detail should be expected to score *worse* than the blocky input on PSNR while looking
  better. That is an argument for a perceptual metric, not against this measurement — here the
  frames and the numbers agree.
- **The ground-truth split into true and false positives is only possible because the degradation
  was synthetic.** On real footage neither this calibration nor the oracle arm can be computed.
- **The smoothing constant is tied to the hysteresis window by argument, not by measurement.**
  α = 1/2 and α = 1/3 scored within 0.003 dB of each other, so this clip cannot distinguish them.
