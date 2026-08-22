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
