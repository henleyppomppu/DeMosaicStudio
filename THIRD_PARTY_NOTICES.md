# Third-Party Notices

Maintained even though D-11 means nothing is distributed. It costs minutes, and it is the only
artifact that makes `prd.md` §2.4's reversal question answerable later. See `prd.md` §4.2 for the
audit and the "contaminating?" column.

A dependency that appears in a lockfile without an entry here fails
`T-LICENSE-NOTICE-COVERAGE-01`.

## Current dependencies

| Component | Version | License | Contaminating if distributed? | Source |
| --- | --- | --- | --- | --- |
| xunit | 2.9.3 | Apache-2.0 | No | https://github.com/xunit/xunit |
| xunit.runner.visualstudio | 3.1.4 | Apache-2.0 | No | https://github.com/xunit/visualstudio.xunit |
| Xunit.SkippableFact | 1.5.23 | MS-PL | No | https://github.com/AArnott/Xunit.SkippableFact |
| Microsoft.NET.Test.Sdk | 17.14.1 | MIT | No | https://github.com/microsoft/vstest |
| coverlet.collector | 6.0.4 | MIT | No | https://github.com/coverlet-coverage/coverlet |
| pytest | 8.3.4 | MIT | No | https://github.com/pytest-dev/pytest |
| lpips | 0.1.4 | BSD-2-Clause | No | https://github.com/richzhang/PerceptualSimilarity |
| scipy | 1.18.1 | BSD-3-Clause | No | https://github.com/scipy/scipy |
| tqdm | 4.70.0 | MPL-2.0 / MIT | No | https://github.com/tqdm/tqdm |

### Model weights, not code

| Weights | License | Note |
| --- | --- | --- |
| AlexNet (torchvision, `alexnet-owt-7be5be79.pth`, 233 MB) | BSD-3-Clause | LPIPS trunk. **Evaluation only** - `scripts/perceptual.py`, never the shipped worker |
| LPIPS v0.1 linear layers (bundled in the `lpips` package, ~6 KB) | BSD-2-Clause | Same |

## Planned dependencies with license consequences

Recorded ahead of adoption so the cost is visible before, not after.

| Component | License | Note |
| --- | --- | --- |
| FFmpeg, GPL build with x264/x265 | GPL-2.0+ | D-12. **Would make the application GPL if ever distributed.** Cheap to undo: rebuild LGPL and lose x265 |
| BasicVSR++ / BasicSR model-zoo weights | S-Lab License 1.0, non-commercial | D-04, initialization only, and only if Phase 2 measures a benefit. **The fine-tuned derivative inherits the license.** This is the expensive door (§18 R-05) |
| PyTorch | BSD-3 | No |
| `timm` encoders | Apache-2.0 | Each pretrained checkpoint carries its own license — check per checkpoint |
| numpy | BSD-3 | No |

## Not used

| Component | License | Why not |
| --- | --- | --- |
| Ultralytics YOLOv8/v11-seg | AGPL-3.0 | Legally available under D-11 and still rejected on engineering grounds — see D-03 |
