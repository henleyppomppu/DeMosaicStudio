# What one film was hiding

**2026-08-23.** The corpus was 24 clips of one film. Two more films — Sintel and Big Buck Bunny,
both Blender Foundation, both CC BY 3.0 — arrived today, and the first thing they did was invert a
decision.

Raw material: `training/datasets/clean-sintel.manifest.json`, `clean-bbb.manifest.json`,
`training/datasets/SOURCES.md`.

---

## 1. The detector on footage it has never seen

Fraction of **clean** frames producing at least one region, at `min_area = 1024`. §5.2.5a asks for
**≤ 0.5%**.

| model | source | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.995 | 0.999 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.1.0 | tos (trained on) | 80.6% | 73.6% | 48.6% | 36.8% | 17.4% | 10.4% | **4.2%** |
| v0.1.0 | sintel | 77.8% | 71.5% | 41.7% | 27.8% | 11.1% | 8.3% | **1.4%** |
| v0.1.0 | bbb | 89.6% | 87.5% | 73.6% | 66.0% | 41.7% | 34.0% | **10.4%** |
| **v0.2.0** | tos | 56.2% | 47.9% | 30.6% | 20.8% | 7.6% | 6.2% | **3.5%** |
| **v0.2.0** | sintel | 61.8% | 41.7% | 26.4% | 18.1% | 13.2% | 11.8% | **6.2%** |
| **v0.2.0** | bbb | 84.0% | 72.2% | 47.2% | 37.5% | 26.4% | 20.1% | **16.7%** |

**No threshold on either model meets §5.2.5a on any source.** The best figure anywhere is 1.4%, at a
threshold of 0.999 that would leave nothing detectable.

## 2. v0.2.0 overfits, and one film could not say so

v0.2.0 was retrained on widened negatives and the retraining was justified by measurement: video-level
firing fell 18.8% → 9.4% for 0.032 of IoU. That measurement was on Tears of Steel.

On Tears of Steel it is better at **every** threshold. On the two unseen films it is better at low
thresholds and **worse at high ones**:

| threshold 0.999 | v0.1.0 | v0.2.0 | |
| --- | ---: | ---: | --- |
| tos (trained on) | 4.2% | **3.5%** | better |
| sintel | **1.4%** | 6.2% | **4× worse** |
| bbb | **10.4%** | 16.7% | **60% worse** |

That is what overfitting to a negatives corpus looks like from the outside: the model learned which
*textures in this film* are not mosaics, rather than what a mosaic is.

Since the pipeline ships threshold 0.5, where v0.2.0 is better on all three, the shipped choice
stands. What does not stand is the claim that the retraining generalised.

## 3. Flat cartoon shading is the worst case, by a lot

Big Buck Bunny fires on **84%** of clean frames at the shipped threshold and **16.7%** even at
0.999 — against 56% and 3.5% for the film the detector was trained on.

That was the reason for choosing it. Large regions of near-constant colour with hard edges and no
grain is what a mosaic looks like when nothing is wrong, and the detector had never seen any.

## 4. The restorer is content-dependent too

The accumulator, at the shipped forgetting horizon, over three target frames per clip:

| clip | motion px/frame | gain |
| --- | ---: | ---: |
| screen-anchored (from tos) | 2.59 | **+7.05 dB** |
| object-anchored ladder (from tos) | 0.84 | +2.88 |
| **sintel** | 10.95 | **+0.39** |
| **bbb** | 3.84 | **−0.07** |

Every headline this project has quoted came from the top two rows. **The method gains almost
nothing on fast content and nothing at all on flat cartoon content**, and two clips of one film
could not show it.

The forgetting horizon's optimum spans a factor of eight across the four — 32 on the
screen-anchored clip, 4 on Sintel and Big Buck Bunny. The shipped 32 is within 0.04 dB of the
four-clip mean optimum, so it stays; but on Sintel it costs about 1.1 dB against the best fixed
choice for that content.

## 5. What this changes

1. **§5.2.5a is unmet by two orders of magnitude and cannot be met by tuning.** No threshold on
   either model reaches 0.5% on any source. This is a training-data problem, which is what
   `scripts/fetch_negatives.py` is for.
2. **A retraining measured on one film is not evidence that it generalised.** v0.2.0's improvement
   was real and did not survive contact with two other films at the thresholds where it mattered.
3. **The restoration's headline figure belongs to a clip, not to the method.** Any future number
   should be quoted per content class, or as a range, and never as one figure.

## 6. Limitations

- **Three films, all Blender open movies**, all CG or CG-heavy. None is camera-original footage,
  none is broadcast or phone video, and the two new ones are animation.
- **Clean frames only.** This measures false positives; it says nothing about whether recall
  survived, and a threshold of 0.999 certainly destroys it.
- **Sampled**: 6 frames from each of 24 clips per source, 144 per source per model.
- **The detector was trained on Tears of Steel crops**, so the "trained on" row is not a held-out
  measurement even for that film.
