# The four hard negatives, measured

**2026-08-23.** §11.4 names four classes of content a mosaic detector is expected to confuse with
block averaging — real optical defocus, LED video walls, mesh fabric, and pixel art — and says they
cannot be manufactured and must be collected. The corpus never contained any of them, so the claim
had never been tested.

48 files are now collected from Wikimedia Commons, every one with its licence and author recorded
(`training/datasets/negatives.manifest.json`). This is what the detector does with them.

Raw material: `scripts/fetch_negatives.py`, `scripts/eval_negatives.py`, `docs/negatives-report.json`.

---

## 1. The measurement

Fraction of images producing at least one region, `min_area = 1024`, detector v0.2.0. Every
collected image is put through the same H.264 encode the clean corpus went through, so the
comparison is about content and not image quality.

| class | n | 0.5 | 0.9 | 0.99 | 0.999 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bokeh | 12 | 25.0% | **0.0%** | 0.0% | 0.0% |
| **clean film (control)** | 24 | **54.2%** | **29.2%** | **8.3%** | **4.2%** |
| led | 12 | 58.3% | 33.3% | 8.3% | 0.0% |
| mesh (wire netting) | 12 | **8.3%** | 8.3% | 0.0% | 0.0% |
| pixel-art | 12 | 66.7% | 16.7% | 0.0% | 0.0% |

§5.2.5a asks for **≤ 0.5%**.

## 2. The hypothesis is mostly wrong

**Two of the four are easier than ordinary footage.** Real optical defocus fires at 25% against the
control's 54%, and disappears entirely by 0.9. Wire netting fires at **8.3%**, the lowest of
anything measured including the film the detector was trained on.

**Two are harder, and only at the lowest threshold.** LED displays (58.3%) and pixel art (66.7%)
exceed the control at 0.5 — and both fall *below* it at 0.9 and above.

So the answer to "are these four classes special?" is: pixel art and LED walls, slightly, at one
operating point. Defocus and mesh, no.

## 3. What is actually wrong is general

The control fires on **54% of frames at the shipped threshold and 4.2% even at 0.999** — and that
control is 24 first frames of Tears of Steel, which is the film the detector was trained on crops
of. It is the most favourable possible test set and it fails the requirement by two orders of
magnitude.

**The detector does not have a hard-negative problem. It has a detector problem.** Collecting four
content classes addresses a hypothesis the measurement does not support; what it needs is to be
better at the thing it does, on everything.

That is worth knowing *before* retraining on the collected data, which is the whole reason to
measure first. The last retraining (v0.2.0) was justified by a measurement on one film and
`docs/detector-generalisation-report.md` records what happened to it on two others.

## 4. What went wrong collecting it

**`Category:Tulle` is the French commune of Tulle, not the fabric.** The first mesh collection was
municipal photography — a telephone box, a local history poster — and the first measurement taken
from it said 33% where wire netting says 8.3%. A category name that reads right in English is not a
check that it contains what you think.

**Pixel art is under-collected and cannot be fixed by synthesis.** Commons has almost none, most
pixel art being copyrighted; the files here come from `Category:Video game screenshots` and are
genuine retro game screens, but twelve of them is not a class. Manufacturing more would be worse
than having none: a photograph downscaled with nearest-neighbour and scaled back up **is** a mosaic,
so labelling one as a negative teaches the detector the exact opposite of the thing.

## 5. Limitations

- **Twelve images per class.** At that size a difference of ten points is not a difference.
- **Photographs, not video.** Compression is matched, but framing, motion blur and sensor noise are
  not, and none of these images is a frame from a moving shot.
- **The classes are not pure.** "LED displays" includes street scenes that happen to contain one;
  "Video game screenshots" includes 3D-rendered games alongside pixel art.
- **The control is training-adjacent**, which flatters the detector. It still fires 54%.
