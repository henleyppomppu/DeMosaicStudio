# Does dense flow close the alignment gap?

**Run 2026-08-22. Answer: it closes 19% of it — and the measurement says the rest is not an
alignment problem.**

Raw data: `docs/phase2-alignment.json` (64 measurements). Script: `scripts/eval_alignment.py`.

---

## 1. The question

The Phase 0 gate measured, in the decision band:

```
oracle    (perfect alignment)      +3.298 dB
estimated (global translation)     -0.861 dB
```

and named the difference Phase 2's critical path (§18 R-13). This experiment adds **dense optical
flow with per-pixel confidence** — RAFT-small, forward-backward consistency — and measures where it
lands between them.

Every arm runs the same solver on the same degraded, recompressed frames. Only the alignment
differs.

## 2. Result

| Arm | Gain over single-frame |
| --- | ---: |
| global translation (the gate's number) | −2.110 dB |
| **dense flow, K=5** | **−0.791 dB** |
| **dense flow, K=3** | **−0.443 dB** |
| oracle (perfect alignment) | +3.494 dB |

Gap closed: **18.8%**. Flow usable on 90.5% of pixels.

**Alignment itself improved a lot.** Mean |neighbour − target| on the clean frames:

| | unaligned | global translation | **dense flow** |
| --- | ---: | ---: | ---: |
| residual | 12.54 | 5.77 | **3.91** |

So dense flow is doing its job. The reconstruction still loses.

## 3. Why — the motion bands say it

| Motion | oracle | global | dense | dense K=3 | residual after dense | flow usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | +3.698 | −3.733 | −1.933 | −1.476 | 7.01 | 75.6% |
| medium | +4.396 | −3.415 | −1.306 | −0.638 | 3.95 | 90.0% |
| **slow** | +3.219 | +0.035 | **+0.363** | **+0.387** | 3.43 | 98.9% |
| static | +2.662 | −1.329 | −0.286 | −0.044 | **1.27** | 97.4% |

Two rows carry the finding.

**`static` has the best alignment of all — residual 1.27, flow usable on 97% of pixels — and still
gains nothing.** Nothing is misaligned; there is simply no new information. A static subject under a
screen-anchored grid lands on the same block every frame, which is §1.4.1's phase-diversity
condition failing for a different reason than object anchoring.

That also exposes something about the oracle arm: it scores **+2.662** on those same static clips.
It can only do that because its neighbours are the target *deliberately shifted*, which manufactures
phase diversity the real footage does not have. **The oracle is not merely an optimistic ceiling; on
static content it measures something that cannot exist.**

**`slow` is the only band that gains, in both this experiment and the Phase 0 gate.** Two
independent measurements agree. It is the one regime where both conditions hold at once: enough
motion to move the subject across the grid, little enough that the neighbours still contain the same
content.

**K=3 beats K=5 in every band.** Shorter baseline, better correspondence. If the limit were flow
accuracy, more frames would still help; it does not.

## 4. What this changes

### 4.1 The binding constraint is content correspondence, not flow accuracy

Residual fell by 69% (12.54 → 3.91) and the gain moved by 1.3 dB out of a 5.6 dB gap. Improving
alignment further has a poor exchange rate, because what remains after alignment is not
misalignment — it is that a frame 2/24 s away genuinely contains different content: motion blur,
changing occlusion, independent object motion, lighting.

**§18 R-13 was half right.** Alignment was worth fixing and fixing it helped. But it is not the
whole gap, and the remainder is not addressable by a better flow estimator.

### 4.2 Multi-frame has a narrow operating window

Screen-anchored grid **and** slow motion. Outside that, a classical multi-frame solver is worse than
single-frame. The router (§5.8) must treat motion band as a gate, not merely as an input to window
size.

### 4.3 §5.6's window policy is contradicted by measurement

The PRD says low motion → K of 7–9. The measurement says low motion is the only place multi-frame
works at all, **and that K=3 is as good as K=5 there**. Larger windows buy nothing and cost VRAM and
time. Recorded as D-16; §5.6's table is revised in `prd.md` v3.3.

## 5. Limitations — read before quoting any number above

- **This is a classical solver.** Iterative back-projection treats every aligned pixel as evidence.
  A learned restorer can be trained to *discount* poorly-corresponding evidence, and might tolerate
  what IBP cannot. This experiment bounds what alignment plus a linear inverse can do; it does not
  bound what a trained model can do. That distinction matters for Phase 2's plan and is the main
  reason this is not a KILL for multi-frame.
- **Flow is computed on the clean frames.** The real pipeline only has the mosaicked video, so this
  arm is already optimistic about flow quality.
- **RAFT-small, off the shelf.** No fine-tuning, no mosaic-specific training, 990 K parameters. A
  larger or adapted estimator would do better — though §4.1 argues the headroom there is small.
- **One film, 8 clips, 64 measurements, blocks 8 and 12, CRF 18 and 26.** No noise floor was measured
  for *this* script; the Phase 0 gate's 0.000 dB applies to the encoder, and the flow network is
  deterministic in eval mode, so the arms are comparable — but that reasoning is not a measurement.
- **PSNR only.** No LPIPS, no warping error, no perceptual assessment. A reconstruction that scores
  worse on PSNR may still look better, and nobody has looked.

## 6. What to do next

1. **Do not spend more on the flow estimator.** §4.1 gives the exchange rate.
2. **Make the router gate on motion band** (§5.8), and default the window to K=3 where multi-frame
   runs at all (D-16).
3. **Test whether a learned restorer tolerates imperfect correspondence** — that is the open question
   §5 names, and it is the thing that decides whether multi-frame survives outside the slow band.
4. Re-run the Phase 0 gate's `estimated` arm with dense flow folded in, so the gate's headline
   number reflects the current best alignment rather than global translation.
