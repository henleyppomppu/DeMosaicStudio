# Phase 0 Report — Foundations and the Feasibility Gate

**Gate run 2026-08-22. Verdict: `PASS_ALIGNMENT_BLOCKED`.**

> The information multi-frame restoration needs **is** present in neighbouring frames — but only
> under a screen-anchored grid, and the global-translation alignment model cannot reach it.
> Alignment (`prd.md` §5.7), not the reconstruction model, is Phase 2's critical path.

Produced by `scripts/eval_multiframe_gate.py`; raw data in `docs/phase0-gate.json` (576
measurements).

---

## 1. Task status

| # | Task | Status |
| --- | --- | --- |
| 0.1 | `git init`, repo skeleton, build props, solution | **Done.** 0 warnings, 0 errors |
| 0.2 | `scripts/check-environment.ps1` | **Done.** 7/7 required checks pass |
| 0.3 | Python venv + GPL FFmpeg (x265) | **Done.** `scripts/setup-worker.ps1`, verified idempotent |
| 0.4 | Decode → passthrough → encode PoC, PTS invariance | **Done.** 15 media tests |
| 0.5 | Synthetic degradation generator | **Partial.** Core operator, grid anchoring, and motion estimation done; the §11.3 randomisation matrix is not |
| 0.6 | Clean corpus + manifest | **Done.** 24 clips from Tears of Steel (CC BY 3.0), motion-stratified |
| 0.7 | **Multi-frame vs single-frame experiment** | **Done.** This report |

---

## 2. The gate

### 2.1 What was measured

For each clip × block size × grid anchoring × CRF × target frame:

1. A fixed 256×256 ROI in **frame** coordinates, so a screen-anchored grid stays fixed while
   content moves through it — the phase diversity of §1.4.1.
2. Mosaic applied, phase either fixed to the frame (SCREEN) or riding the content (OBJECT).
3. **Re-encoded through H.264** at CRF 18 and 26 (§11.3 makes this mandatory).
4. Reconstructed by iterative back-projection in three arms that differ **only in what the
   neighbours are** (D-14):

   | Arm | Neighbours | Answers |
   | --- | --- | --- |
   | `single` | none (K=1) | the floor |
   | `oracle` | synthesised from the target at known shifts | **is the information recoverable?** |
   | `estimated` | the real neighbours, aligned by global translation | can our alignment reach it? |

5. Scored against the untouched clip.

No learned model is involved anywhere. That is deliberate: a learned restorer would make a poor
result ambiguous between "no information survived" and "this model is bad", and the gate exists to
answer the first question only.

**`single` and `passthrough` score identically (24.710 dB).** By construction: with one observation
IBP converges to the block means, which *is* pass-through. So the oracle gain below measures
temporal evidence alone, with no prior and no model quality mixed in.

### 2.2 Configuration

| | |
| --- | --- |
| Corpus | 12 clips from `clean-tos`, spread across all four motion bands |
| Blocks | 4, 8, 12, 20 px |
| Anchoring | SCREEN, OBJECT |
| CRF | 18, 26 |
| Window | K = 5 |
| Targets per clip | 3 |
| Measurements | 576 |
| **Noise floor** | **+0.000 dB** |

The noise floor is exactly zero because x264 is deterministic at these settings, so **every
difference reported below is real**. That also means the noise floor is a weak constraint in this
particular gate and the 1.0 dB threshold is doing the work — which is the honest reading, not a
stronger one.

---

## 3. Results

### 3.1 Decision band — SCREEN, blocks 6–12 (`prd.md` §1.4.3)

| Arm | PSNR | Gain over single-frame |
| --- | ---: | ---: |
| pass-through | 24.710 dB | — |
| single frame | 24.710 dB | — |
| **multi, oracle** | **28.008 dB** | **+3.298 dB** |
| multi, estimated | 23.849 dB | −0.861 dB |

n = 144. Neighbours passing the alignment filter: 65.3%.

`+3.298 dB` clears the 1.0 dB threshold by a wide margin. **§1.4's premise holds.**

### 3.2 By block size — the §1.4.2 bands, measured

| Block | §1.4.2 predicted | SCREEN oracle | SCREEN estimated | **OBJECT oracle** |
| ---: | --- | ---: | ---: | ---: |
| 4 | deblocking; mostly evidence-backed | +2.516 | −2.680 | **−1.499** |
| 8 | target band | **+3.312** | −1.022 | **−1.191** |
| 12 | target band | **+3.285** | −0.699 | **−0.895** |
| 20 | prior-dominated | +2.427 | −0.184 | **−0.790** |

Two things worth stating plainly.

**The predicted band is where the gain peaks.** §1.4.2 guessed that 5–12 px would be the band where
multi-frame genuinely reconstructs, and that is exactly where the oracle gain is largest. At 4 px
single-frame already retains more, so there is less headroom; at 20 px the destruction outruns what
five frames can restore. The hypothesis was written before any of this was measured.

**Object-anchored is negative at every block size.** Not "no better" — *worse*, by 0.79 to 1.50 dB.
Extra frames under an object-anchored grid contribute no new information and do contribute codec
noise, so fusing them actively damages the result. This turns §5.4.4's grid-anchoring estimator and
§5.8's fallback from a "should" into a measured requirement: **a pipeline that fails to detect
object anchoring will produce worse output than one that never attempted multi-frame at all.**

### 3.3 By motion band — SCREEN, all blocks

| Motion | n | oracle | estimated | neighbours usable |
| --- | ---: | ---: | ---: | ---: |
| static | 72 | +2.665 | −0.182 | 52.8% |
| slow | 72 | +2.923 | **+0.319** | 61.1% |
| medium | 72 | +3.159 | −1.762 | 86.1% |
| fast | 72 | +2.794 | −2.961 | 61.1% |

The estimated arm is positive **only** in the slow band — the one regime where a global translation
is approximately the right motion model. It degrades steadily as motion grows, and is worst where
motion is fastest. That is the signature of an alignment failure, not an information failure.

Note the medium band: 86% of neighbours passed the alignment filter and the result was still
−1.76 dB. **A per-frame photometric residual ratio is not an adequate alignment-confidence measure**
— it accepts frames whose global alignment is plausible but whose local motion is not. §5.9.4's
confidence has to be per-pixel, derived from dense flow, not per-frame.

### 3.4 By CRF — recompression cost, quantified

| CRF | oracle gain | estimated gain |
| ---: | ---: | ---: |
| 18 | +3.253 dB | −1.058 dB |
| 26 | +2.518 dB | −1.234 dB |

Recompression erases **roughly 23% of the recoverable gain** between CRF 18 and 26. §11.3 required
recompression in every evaluation sample on the grounds that it destroys the inter-block residue
multi-frame solving depends on; that is now a number rather than an assertion, and it is a good
reason to keep the CRF ladder in the Phase 1/2 training data.

---

## 4. What this changes

1. **Phase 2 is not killed.** The premise §19 rests on holds in the target band.
2. **Alignment is the critical path, not the restoration model.** The gap between +3.30 dB (perfect
   alignment) and −0.86 dB (global translation) is the entire prize. §5.7's optical-flow /
   deformable alignment moves from "one of several permitted implementations" to the thing Phase 2
   should attack first.
3. **Grid-anchoring detection (§5.4.4) is load-bearing, not diagnostic.** Object-anchored
   multi-frame is measurably harmful. §5.8's fallback must be reliable before multi-frame ships.
4. **Alignment confidence must be per-pixel.** §3.3's medium-motion row shows a per-frame measure
   accepting frames it should reject.
5. **Recompression stays mandatory in the dataset** (§11.3), now with a measured cost.

---

## 5. Limitations — read before quoting any number above

- **One source film.** Tears of Steel (live action + VFX), 1920×800, already H.264-compressed. The
  "clean" reference therefore carries codec artifacts of its own. Relative comparisons are sound;
  absolute PSNR is against a compressed reference.
- **The oracle arm is a generous upper bound.** Its neighbours are the target frame shifted, so they
  contain *exactly* the same content. Real neighbours never do — they carry occlusion, independent
  object motion and lighting change that no alignment can convert into evidence about the target.
  Real flow-based alignment will land somewhere between −0.86 and +3.30 dB, and closer to the
  middle than to the top.
- **One ROI position** (centre crop), one window size (K=5), one restoration algorithm.
- **LPIPS not measured.** It needs a pretrained network and arrives with Phase 2. `prd.md` §1.4.3
  decides on PSNR *and* LPIPS *and* warping error together, so **this gate satisfies one of the
  three criteria and reports the other two as not measured.** Re-run it when LPIPS is available.
- **Warping error not measured.** The gate reconstructs isolated target frames, not sequences.
- **Global translation only.** No affine, no dense flow. That is the finding, not a defect of the
  gate — but it means the estimated arm is a lower bound on what alignment can do, just as the
  oracle arm is an upper bound.

---

## 6. Verification that has run

```
dotnet build DeMosaicStudio.slnx -c Release   0 warnings, 0 errors
dotnet test  DeMosaicStudio.slnx -c Release   88 passed
.venv\Scripts\python.exe -m pytest            passing (see CLAUDE.md for the current count)
scripts\check-environment.ps1                 7/7, exit 0
```

Cross-language parity is verified in both directions: the fingerprint canonical text and digests,
the error-code table, and `PROTOCOL_VERSION`.

## 7. Verification that has NOT run

- Any learned model, on any path.
- The GPU compute path. Everything above is CPU numpy and x264.
- Real mosaicked footage. All mosaics here are synthetic, which is what makes ground truth exist.
