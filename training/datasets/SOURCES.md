# Training and Evaluation Sources

Every source file records its origin and known license here. prd.md §11.2.

Under D-11 the corpus is not gated on redistribution-friendly licensing — models and datasets are
never distributed, and training is personal, non-commercial use. This file is kept anyway for two
reasons that outlive the licensing question:

1. It is the only artifact that makes prd.md §2.4's reversal question answerable later.
2. prd.md §2.3 C-2 is unaffected by D-11: no model is trained on, or fine-tuned toward
   reconstructing, identifiable real individuals. The model learns to invert a *degradation*, and
   the corpus is chosen accordingly.

## Splits

Splits are by **source file and by scene**, never by frame (§11.6). Near-duplicate frames from the
same clip in different splits inflate every metric.

| Split | Purpose | Target size | Have |
| --- | --- | --- | --- |
| `clean-train` | Training | >= 40 h | — |
| `clean-val` | Validation, disjoint sources | >= 4 h | — |
| `clean-test` | Frozen test, disjoint sources | >= 4 h | 96 s (below) |
| `negatives` | Hard negatives (§11.4) | >= 6 h | — |
| `real-mosaic` | Unpaired, detector recall and domain-gap checking only | >= 2 h | — |

**The Phase 0 gate needs far less than training does.** The corpus below is sized for the gate
(§1.4.3), not for Phase 1 or 2. Those still need the hours in the table.

## Origins

### `clean-tos` — Tears of Steel (2012)

| Field | Value |
| --- | --- |
| Attribution | (CC) Blender Foundation \| mango.blender.org |
| License | CC BY 3.0 |
| Source file | `tears_of_steel_1080p.mov`, 1920x800, 24 fps, 734 s |
| Download | https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip |
| Archive SHA-256 | `d87a41de040d3814dbde143e9ab85ef122caf22265f660b0bebf476cd8b357a5` |
| Retrieved | 2026-08-22 |
| Built by | `scripts/build_corpus.py --clips 24 --seconds 4` |
| Manifest | `training/datasets/clean-tos.manifest.json` |

**Why this film.** Of the Blender open movies it is the only one that is live action plus VFX
rather than pure CG, so it carries real camera grain, real skin and fabric texture, and real
lens behaviour. Those are exactly the fine details block averaging destroys and multi-frame
reconstruction has to recover.

**Known limitation.** The distributed file is already H.264-compressed, so the "clean" ground truth
carries codec artifacts of its own. That is acceptable for the gate, which compares single-frame
against multi-frame on the *same* reference — but the absolute PSNR figures are against a
compressed reference and should not be quoted as if against a camera original.

**Motion distribution** (measured, not assumed — `training/degradation/motion.py`):
fast=4 · medium=7 · slow=8 · static=5

| Clip | Split | Origin | Motion band | Median px/frame | Frames | SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `tos_000.mp4` | clean-tos | Tears of Steel @ 45s | slow | 0.59 | 96 | `40fe36f9427f…` |
| `tos_001.mp4` | clean-tos | Tears of Steel @ 69s | static | 0.02 | 96 | `cc1b7826e64b…` |
| `tos_002.mp4` | clean-tos | Tears of Steel @ 92s | slow | 0.97 | 96 | `67f90ddac05a…` |
| `tos_003.mp4` | clean-tos | Tears of Steel @ 116s | slow | 0.27 | 96 | `f9c0e1e9ae1c…` |
| `tos_004.mp4` | clean-tos | Tears of Steel @ 140s | slow | 0.28 | 96 | `fe1982624f7c…` |
| `tos_005.mp4` | clean-tos | Tears of Steel @ 164s | fast | 7.40 | 96 | `e1f56713555a…` |
| `tos_006.mp4` | clean-tos | Tears of Steel @ 187s | static | 0.06 | 96 | `173574cb5f02…` |
| `tos_007.mp4` | clean-tos | Tears of Steel @ 211s | slow | 0.43 | 96 | `8ccbd28eb592…` |
| `tos_008.mp4` | clean-tos | Tears of Steel @ 235s | static | 0.19 | 96 | `e98f542c606b…` |
| `tos_009.mp4` | clean-tos | Tears of Steel @ 258s | medium | 4.19 | 96 | `549b8c1908f8…` |
| `tos_010.mp4` | clean-tos | Tears of Steel @ 282s | medium | 4.38 | 96 | `b84ed76668c8…` |
| `tos_011.mp4` | clean-tos | Tears of Steel @ 306s | medium | 3.08 | 96 | `b3eb295900ca…` |
| `tos_012.mp4` | clean-tos | Tears of Steel @ 330s | fast | 9.08 | 96 | `c8c1c3ffe9e0…` |
| `tos_013.mp4` | clean-tos | Tears of Steel @ 353s | slow | 0.38 | 96 | `533a9212d4a6…` |
| `tos_014.mp4` | clean-tos | Tears of Steel @ 377s | medium | 4.94 | 96 | `2f1fc940372c…` |
| `tos_015.mp4` | clean-tos | Tears of Steel @ 401s | slow | 0.67 | 96 | `3235aaede2fc…` |
| `tos_016.mp4` | clean-tos | Tears of Steel @ 424s | medium | 4.71 | 96 | `1837ed082276…` |
| `tos_017.mp4` | clean-tos | Tears of Steel @ 448s | medium | 5.55 | 96 | `2b72c1316435…` |
| `tos_018.mp4` | clean-tos | Tears of Steel @ 472s | fast | 40.03 | 96 | `903bec35cc6b…` |
| `tos_019.mp4` | clean-tos | Tears of Steel @ 496s | fast | 36.48 | 96 | `1a3a104c3fdb…` |
| `tos_020.mp4` | clean-tos | Tears of Steel @ 519s | medium | 1.12 | 96 | `ba9d5b3c9b2c…` |
| `tos_021.mp4` | clean-tos | Tears of Steel @ 543s | static | 0.12 | 96 | `27df0dfc2a71…` |
| `tos_022.mp4` | clean-tos | Tears of Steel @ 567s | slow | 0.88 | 96 | `806992661baf…` |
| `tos_023.mp4` | clean-tos | Tears of Steel @ 590s | static | 0.20 | 96 | `5909be049d55…` |

### `clean-sintel` — Sintel (2010)

| Field | Value |
| --- | --- |
| Attribution | (CC) Blender Foundation \| durian.blender.org |
| License | CC BY 3.0 |
| Source file | `Sintel.2010.1080p.mkv`, 1920x818, 24 fps, 888 s |
| Download | https://download.blender.org/durian/movies/Sintel.2010.1080p.mkv |
| SHA-256 | `97f1dbc66231df42ad49bd8c29aa174b8f48933058e47e7157d4ba63d93a8efa` |
| Retrieved | 2026-08-23 |
| Built by | `scripts/build_corpus.py --clips 24 --seconds 4 --prefix sintel --skip-head 60 --skip-tail 120` |
| Manifest | `training/datasets/clean-sintel.manifest.json` |

**Why this film.** The corpus was one film, so nothing could distinguish "the pipeline works" from
"the pipeline works on Tears of Steel". Sintel is pure CG rather than live action plus VFX, and its
motion distribution is the opposite end: **fast=9 · medium=5 · slow=3 · static=7** against Tears of
Steel's fast=4 · medium=7 · slow=8 · static=5.

**What it immediately showed.** The accumulator gains **+0.39 dB** here at the shipped forgetting
horizon against +7.05 on the screen-anchored clip. Fast content is where the method is weakest, and
one film could not say so.

First six clips (the manifest has all 24):

| Clip | Split | Origin | Motion band | Median px/frame | Frames | SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `sintel_000.mp4` | clean-sintel | 60s | fast | 12.74 | 96 | `77838dbc8a9b…` |
| `sintel_001.mp4` | clean-sintel | 90s | static | 0.00 | 95 | `61122a88daa1…` |
| `sintel_002.mp4` | clean-sintel | 119s | slow | 0.77 | 95 | `328e82bab4b9…` |
| `sintel_003.mp4` | clean-sintel | 149s | static | 0.00 | 95 | `05eb042cdb47…` |
| `sintel_004.mp4` | clean-sintel | 178s | fast | 12.68 | 95 | `2367affcd448…` |
| `sintel_005.mp4` | clean-sintel | 208s | static | 0.06 | 95 | `690dfb9b0461…` |

### `clean-bbb` — Big Buck Bunny (2008)

| Field | Value |
| --- | --- |
| Attribution | (CC) Blender Foundation \| peach.blender.org |
| License | CC BY 3.0 |
| Source file | `bbb_sunflower_1080p_30fps_normal.mp4`, 1920x1080, 30 fps, 635 s |
| Download | https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip |
| Archive SHA-256 | `e320fef389ec749117d0c1583945039266a40f25483881c2ff0d33207e62b362` |
| Retrieved | 2026-08-23 |
| Built by | `scripts/build_corpus.py --clips 24 --seconds 4 --prefix bbb --skip-head 20 --skip-tail 40` |
| Manifest | `training/datasets/clean-bbb.manifest.json` |

**Why this film, and it is not for the clean corpus.** Flat cartoon shading is the shape a mosaic
detector confuses with block averaging: large regions of near-constant colour, hard edges, no grain.
It is the closest thing to a **hard negative** (section 11.4) that can be downloaded rather than
filmed, and it is nothing like the live action plus VFX the detector was trained on.

**What it immediately showed.** At the shipped threshold, **81.8%** of its clean frames produce a
region, against 55.2% on Tears of Steel — and the mean false-positive area is three times as large.
section 5.2.5a asks for 0.5%.

**Motion:** medium=6 · slow=1 · **static=17**. Held cartoon shots, so it is a poor source for the
temporal questions and a good one for the detector.

First six clips (the manifest has all 24):

| Clip | Split | Origin | Motion band | Median px/frame | Frames | SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `bbb_000.mp4` | clean-bbb | 20s | static | 0.18 | 120 | `66e4c41e8c0e…` |
| `bbb_001.mp4` | clean-bbb | 44s | static | 0.05 | 119 | `38ae625f561e…` |
| `bbb_002.mp4` | clean-bbb | 68s | static | 0.01 | 119 | `b69567759535…` |
| `bbb_003.mp4` | clean-bbb | 92s | static | 0.02 | 120 | `f7a79373341d…` |
| `bbb_004.mp4` | clean-bbb | 116s | medium | 2.35 | 119 | `208e7d9297fd…` |
| `bbb_005.mp4` | clean-bbb | 140s | static | 0.15 | 119 | `f456af3ca006…` |
