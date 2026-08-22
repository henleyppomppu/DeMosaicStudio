"""Where the untouched-region loss comes from. prd.md FR-1.8, section 5.1.8.

The ladder left the output 4.57 dB below its input on picture that was never mosaicked. Two things
could be spending that, and they need different fixes:

* **the encoder** - every frame is decoded and re-encoded, including frames nothing touched.
  R-1.8c's smart-cut is the answer to this one.
* **the pipeline** - dilation and feather bleed past the mask, so "outside" is not entirely
  outside. Mask discipline is the answer to that one, and smart-cut would not help at all.

A null run separates them: same media layer, same encoder, same settings, restoration disabled.
Whatever the null run loses is the encoder; whatever the pipeline loses beyond it is the pipeline.
That is T-QUALITY-NULLRUN-01's definition, run here against the ladder's own input so the numbers
are comparable to docs/phase3-endtoend-report.md section 8.3.

It then asks the question that decides whether smart-cut is worth building **for this content**:
how many frames does the detector actually fire on? Smart-cut re-encodes any GOP containing a
restored frame. If detections are spread across every GOP there is nothing left to stream-copy, and
the ceiling on smart-cut is zero no matter how well it is implemented.

Usage:

    .venv/Scripts/python.exe scripts/eval_untouched.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO / "training"))

from demosaic_worker.detect.regions import extract_regions  # noqa: E402
from demosaic_worker.detect.segmenter import Segmenter  # noqa: E402
from demosaic_worker.media.passthrough import run_passthrough  # noqa: E402
from demosaic_worker.metrics import psnr  # noqa: E402

import evalclips  # noqa: E402  (scripts/ is on the path as this file's own directory)

CORPUS = REPO / "training" / "datasets" / "clean"
ARTIFACTS = REPO / "artifacts"
MODELS = REPO / "models" / "detector"


@dataclass(frozen=True, slots=True)
class Arm:
    """One video scored on picture that was never mosaicked."""

    name: str
    what: str
    outside_vs_clean: float
    outside_vs_input: float


def _luma(path: Path, limit: int = 200) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            out.append(frame.to_ndarray(format="gray").astype(np.float64))
            if len(out) >= limit:
                break
    return out


def _keyframes(path: Path) -> list[int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [i for i, f in enumerate(container.decode(stream)) if f.key_frame]


def score_outside(clean: list[np.ndarray], degraded: list[np.ndarray],
                  candidate: list[np.ndarray]) -> tuple[float, float]:
    """PSNR on pixels the mosaic never touched, against the clean original and against the input.

    The two views answer different questions. Against clean says how good the picture is; against
    the input says what *this stage* did to it, which is the number a fix has to move.
    """
    count = min(len(clean), len(degraded), len(candidate))
    versus_clean: list[float] = []
    versus_input: list[float] = []

    for index in range(count):
        untouched = np.abs(clean[index] - degraded[index]) <= 1
        if untouched.sum() <= 10000:
            continue
        versus_clean.append(psnr(clean[index][untouched], candidate[index][untouched]))
        versus_input.append(psnr(degraded[index][untouched], candidate[index][untouched]))

    return (
        float(np.mean(versus_clean)) if versus_clean else 0.0,
        float(np.mean(versus_input)) if versus_input else 0.0,
    )


def bleed_analysis(clean: list[np.ndarray], degraded: list[np.ndarray],
                   null_run: list[np.ndarray], pipeline: list[np.ndarray]) -> dict:
    """Isolates the pixels the restoration altered *outside* the mosaicked region.

    Comparing the pipeline against the null run rather than against the input removes the encoder
    from the picture entirely: both went through the same encoder at the same settings, so whatever
    differs is the restoration.

    The area and the error have to be reported separately. A wide halo with a small error and a
    narrow halo with a large one produce the same average PSNR and need opposite fixes - the first
    is the dilation being too generous, the second is the restoration being wrong where it fires.
    """
    count = min(len(clean), len(degraded), len(null_run), len(pipeline))

    untouched_area: list[float] = []
    bleed_area: list[float] = []
    bleed_before: list[float] = []
    bleed_after: list[float] = []

    for index in range(count):
        untouched = np.abs(clean[index] - degraded[index]) <= 1
        altered = np.abs(pipeline[index] - null_run[index]) > 1
        bleed = untouched & altered

        untouched_area.append(float(untouched.mean()))
        bleed_area.append(float(bleed.mean()))

        if bleed.sum() > 1000:
            bleed_before.append(psnr(clean[index][bleed], null_run[index][bleed]))
            bleed_after.append(psnr(clean[index][bleed], pipeline[index][bleed]))

    fraction_of_untouched = (
        float(np.mean(bleed_area)) / float(np.mean(untouched_area))
        if untouched_area and np.mean(untouched_area) else 0.0
    )

    return {
        "untouchedAreaMean": round(float(np.mean(untouched_area)), 4),
        "bleedAreaMean": round(float(np.mean(bleed_area)), 5),
        "bleedShareOfUntouched": round(fraction_of_untouched, 4),
        "framesMeasured": len(bleed_after),
        "bleedPsnrNullRun": round(float(np.mean(bleed_before)), 3) if bleed_before else 0.0,
        "bleedPsnrPipeline": round(float(np.mean(bleed_after)), 3) if bleed_after else 0.0,
    }


def detector_coverage(source: Path, model: str, threshold: float, min_area: int) -> dict:
    """How many frames, and how many GOPs, contain at least one region.

    A frame with no region needs no re-encode. A GOP with no such frame can be stream-copied whole.
    Both numbers are ceilings on what smart-cut could ever save on this file.
    """
    segmenter = Segmenter(MODELS / model)
    keyframes = _keyframes(source)

    firing: list[bool] = []
    with av.open(str(source)) as container:
        for frame in container.decode(container.streams.video[0]):
            luma = frame.to_ndarray(format="gray").astype(np.float64)
            probability = segmenter.probability(luma)
            regions, _ = extract_regions(
                probability, threshold=threshold, min_area=min_area, max_regions=64
            )
            firing.append(bool(regions))

    total = len(firing)
    gop_bounds = keyframes + [total]
    gops = []
    for start, end in zip(gop_bounds, gop_bounds[1:]):
        gops.append({"start": start, "end": end, "fires": any(firing[start:end])})

    copyable_frames = sum(1 for f in firing if not f)
    copyable_gop_frames = sum(g["end"] - g["start"] for g in gops if not g["fires"])

    return {
        "frames": total,
        "framesFiring": total - copyable_frames,
        "framesCopyable": copyable_frames,
        "gops": len(gops),
        "gopsFiring": sum(1 for g in gops if g["fires"]),
        "framesInCopyableGops": copyable_gop_frames,
        "keyframes": keyframes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    evalclips.add_argument(parser)
    parser.add_argument("--input", type=Path,
                        help="the mosaicked input; defaults to the clip's own")
    parser.add_argument("--pipeline", type=Path, required=True,
                        help="the pipeline's output for that input")
    parser.add_argument("--model", default="det-unet-0.2.0")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--min-area", type=int, default=1024)
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "untouched-decomposition.json")
    args = parser.parse_args(argv)

    clip = evalclips.resolve(args.clip)
    source = args.input or clip.degraded
    print(f"clip: {clip.name} - {clip.what}", flush=True)

    for path in (source, args.pipeline):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    null_output = ARTIFACTS / "nullrun.mp4"
    print(f"null run: re-encoding {source.name} with restoration disabled "
          f"(x265 {args.preset} CRF {args.crf})", flush=True)
    run_passthrough(
        source, null_output, transform=None,
        encoder="libx265", crf=args.crf, preset=args.preset,
    )

    clean = _luma(clip.clean)
    degraded = _luma(source)

    arms = [
        Arm("input", "the mosaicked video itself", *score_outside(clean, degraded, degraded)),
        Arm("null run", "re-encode only, restoration disabled",
            *score_outside(clean, degraded, _luma(null_output))),
        Arm("pipeline", "re-encode + restoration",
            *score_outside(clean, degraded, _luma(args.pipeline))),
    ]

    print(flush=True)
    print(f"{'arm':10} {'what':38} {'vs clean':>9} {'vs input':>9}")
    for arm in arms:
        print(f"{arm.name:10} {arm.what:38} {arm.outside_vs_clean:9.2f} "
              f"{arm.outside_vs_input:9.2f}")

    encoder_cost = arms[0].outside_vs_clean - arms[1].outside_vs_clean
    pipeline_cost = arms[1].outside_vs_clean - arms[2].outside_vs_clean

    print()
    print(f"  encoder spends   {encoder_cost:5.2f} dB   (input -> null run)")
    print(f"  pipeline spends  {pipeline_cost:5.2f} dB   (null run -> pipeline)")
    print(f"  total            {encoder_cost + pipeline_cost:5.2f} dB")

    print()
    print("bleed - where the pipeline wrote outside the mosaicked region", flush=True)
    bleed = bleed_analysis(clean, degraded, _luma(null_output), _luma(args.pipeline))
    print(f"  untouched picture     {bleed['untouchedAreaMean']:.1%} of the frame")
    print(f"  of which altered      {bleed['bleedShareOfUntouched']:.2%}  "
          f"({bleed['bleedAreaMean']:.3%} of the whole frame)")
    print(f"  on those pixels       {bleed['bleedPsnrNullRun']:.2f} dB -> "
          f"{bleed['bleedPsnrPipeline']:.2f} dB  "
          f"({bleed['bleedPsnrPipeline'] - bleed['bleedPsnrNullRun']:+.2f} dB)")

    print()
    print("detector coverage - the ceiling on what smart-cut could save here", flush=True)
    coverage = detector_coverage(source, args.model, args.threshold, args.min_area)

    frames = coverage["frames"]
    print(f"  frames firing         {coverage['framesFiring']:4} / {frames}")
    print(f"  frames copyable       {coverage['framesCopyable']:4} "
          f"({coverage['framesCopyable'] / frames:.0%} of the file)")
    print(f"  GOPs firing           {coverage['gopsFiring']:4} / {coverage['gops']}  "
          f"(keyframes at {coverage['keyframes']})")
    print(f"  frames in copyable GOPs {coverage['framesInCopyableGops']:4} "
          f"({coverage['framesInCopyableGops'] / frames:.0%} of the file)")

    print()
    ceiling = coverage["framesInCopyableGops"] / frames
    if ceiling == 0:
        print("  Smart-cut saves NOTHING on this file: every GOP contains a detection.")
        print("  It is still the right answer for content where the mosaic covers part of the")
        print("  runtime - but this clip cannot measure that, and neither can this ladder.")
    else:
        print(f"  Smart-cut could stream-copy {ceiling:.0%} of the file, recovering at most")
        print(f"  {encoder_cost:.2f} dB on that portion.")

    args.out.write_text(
        json.dumps(
            {
                "clip": clip.name,
                "arms": [asdict(a) for a in arms],
                "encoderCostDb": round(encoder_cost, 3),
                "pipelineCostDb": round(pipeline_cost, 3),
                "bleed": bleed,
                "coverage": coverage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
