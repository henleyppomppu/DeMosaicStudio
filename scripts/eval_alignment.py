"""Does dense flow close the alignment gap the Phase 0 gate found? prd.md section 5.7, section 18 R-13.

The gate measured, in the decision band:

    oracle (perfect alignment)          +3.298 dB
    estimated (global translation)      -0.861 dB

and named the difference Phase 2's critical path. This script adds a third estimate - **dense
optical flow with per-pixel confidence** - and measures where it lands between the two.

The comparison is controlled the same way the gate's was: every arm runs the *same* solver on the
*same* degraded, recompressed frames, and only the alignment differs. So a difference is
attributable to alignment and to nothing else.

Usage::

    .venv/Scripts/python.exe scripts/eval_alignment.py --clips 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))
sys.path.insert(0, str(REPO / "training"))

from demosaic_worker.analyze.profile import MosaicProfile  # noqa: E402
from demosaic_worker.analyze.motion import cumulative_content_shifts  # noqa: E402
from demosaic_worker.metrics import psnr, shift_bilinear, ssim  # noqa: E402
from demosaic_worker.restore.flow import DenseAligner, warp_by_flow  # noqa: E402
from demosaic_worker.restore.ibp import (  # noqa: E402
    FlowObservation,
    Observation,
    block_average,
    reconstruct,
    reconstruct_flow,
    upsample_baseline,
)

FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
CORPUS = REPO / "training" / "datasets" / "clean"
MANIFEST = REPO / "training" / "datasets" / "clean-tos.manifest.json"

WINDOW = 5
ROI = 256


@dataclass(frozen=True, slots=True)
class Row:
    """One (clip, block, CRF, target) comparison."""

    clip: str
    motion_band: str
    block: int
    crf: int
    target: int
    psnr_passthrough: float
    psnr_single: float
    psnr_oracle: float
    psnr_global: float
    psnr_dense: float
    psnr_dense_k3: float
    residual_none: float
    residual_global: float
    residual_dense: float
    ssim_single: float
    ssim_dense: float
    flow_usable_fraction: float

    @property
    def gain_oracle(self) -> float:
        """The ceiling: what perfect alignment delivers."""
        return self.psnr_oracle - self.psnr_single

    @property
    def gain_global(self) -> float:
        """What the Phase 0 gate measured."""
        return self.psnr_global - self.psnr_single

    @property
    def gain_dense(self) -> float:
        """What dense flow delivers."""
        return self.psnr_dense - self.psnr_single

    @property
    def gain_dense_k3(self) -> float:
        """Dense flow restricted to the two immediately adjacent frames.

        Shorter baseline, better content correspondence. Tests whether the residual that survives
        alignment is a flow-accuracy problem or an irreducible content difference.
        """
        return self.psnr_dense_k3 - self.psnr_single

    @property
    def gap_closed(self) -> float:
        """Fraction of the global-to-oracle gap that dense flow recovers."""
        span = self.gain_oracle - self.gain_global
        return (self.gain_dense - self.gain_global) / span if abs(span) > 1e-9 else 0.0


def _load_luma(path: Path, roi: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            plane = frame.to_ndarray(format="gray").astype(np.float64)
            height, width = plane.shape
            top, left = (height - roi) // 2, (width - roi) // 2
            frames.append(plane[top : top + roi, left : left + roi])
    return frames


def _recompress(frames: list[np.ndarray], crf: int, tmp: Path) -> list[np.ndarray]:
    height, width = frames[0].shape
    raw, out = tmp / f"a_{crf}.y4m", tmp / f"a_{crf}.mp4"

    with raw.open("wb") as handle:
        handle.write(f"YUV4MPEG2 W{width} H{height} F24:1 Ip A1:1 Cmono\n".encode("ascii"))
        for frame in frames:
            handle.write(b"FRAME\n")
            handle.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())

    subprocess.run(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(raw), "-c:v", "libx264", "-preset", "medium",
            "-crf", str(crf), "-pix_fmt", "yuv420p", str(out),
        ],
        check=True,
        capture_output=True,
    )

    decoded = _load_luma(out, min(height, width))
    raw.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return decoded


def _oracle_shifts(block: int, window: int) -> list[tuple[float, float]]:
    step = block / window
    return [(0.0, 0.0)] + [
        (round(step * k, 3), round(step * (window - k) / 2.0, 3)) for k in range(1, window)
    ]


def _measure_clip(
    path: Path,
    motion_band: str,
    aligner: DenseAligner,
    blocks: tuple[int, ...],
    crfs: tuple[int, ...],
    targets: int,
    tmp: Path,
) -> list[Row]:
    clean = _load_luma(path, ROI)
    if len(clean) < WINDOW + 2:
        return []

    half = WINDOW // 2
    positions = np.linspace(half, len(clean) - half - 1, targets).astype(int).tolist()
    rows: list[Row] = []

    for target in positions:
        window = list(range(target - half, target + half + 1))
        frames = [clean[i] for i in window]
        truth = frames[half]

        shifts = cumulative_content_shifts(frames, half)

        # Dense alignment is computed once on the clean frames and reused for every block size:
        # the flow describes the scene's motion, which does not depend on how it was mosaicked.
        alignments = [
            None if i == half else aligner.align(truth, frames[i]) for i in range(len(frames))
        ]
        usable = statistics.fmean(
            a.usable_fraction for a in alignments if a is not None
        )

        for block in blocks:
            spec = MosaicProfile(block_width=block, block_height=block)
            phase = (0, 0)

            degraded = [block_average(f, spec, phase) for f in frames]

            oracle_shifts = _oracle_shifts(block, WINDOW)
            oracle_degraded = [
                block_average(
                    shift_bilinear(truth, dx, dy) if (dx or dy) else truth, spec, phase
                )
                for dx, dy in oracle_shifts
            ]

            for crf in crfs:
                coded = _recompress(degraded, crf, tmp)
                coded_oracle = _recompress(oracle_degraded, crf, tmp)
                if len(coded) != len(degraded) or len(coded_oracle) != len(oracle_degraded):
                    continue

                single = reconstruct(
                    [Observation(coded[half], 0.0, 0.0)], spec, phase, iterations=30
                ).image

                oracle = reconstruct(
                    [
                        Observation(coded_oracle[i], oracle_shifts[i][0], oracle_shifts[i][1])
                        for i in range(len(coded_oracle))
                    ],
                    spec,
                    phase,
                    iterations=30,
                ).image

                global_obs = [Observation(coded[half], 0.0, 0.0)]
                global_obs.extend(
                    Observation(coded[i], shifts[i][0], shifts[i][1])
                    for i in range(len(coded))
                    if i != half
                )
                global_arm = reconstruct(global_obs, spec, phase, iterations=30).image

                flow_obs = [FlowObservation.target(coded[half])]
                flow_obs.extend(
                    FlowObservation(
                        coded[i],
                        alignments[i].target_to_neighbour,
                        alignments[i].neighbour_to_target,
                        alignments[i].confidence,
                    )
                    for i in range(len(coded))
                    if i != half
                )
                dense = reconstruct_flow(flow_obs, spec, phase, iterations=30).image

                # K=3: the target and its two immediate neighbours only.
                near = [half - 1, half + 1]
                flow_obs_k3 = [FlowObservation.target(coded[half])]
                flow_obs_k3.extend(
                    FlowObservation(
                        coded[i],
                        alignments[i].target_to_neighbour,
                        alignments[i].neighbour_to_target,
                        alignments[i].confidence,
                    )
                    for i in near
                )
                dense_k3 = reconstruct_flow(flow_obs_k3, spec, phase, iterations=30).image

                # How much of each neighbour survives alignment, on the clean frames.
                residual_none = statistics.fmean(
                    float(np.mean(np.abs(truth - frames[i]))) for i in range(len(frames)) if i != half
                )
                residual_global = statistics.fmean(
                    float(np.mean(np.abs(shift_bilinear(truth, *shifts[i]) - frames[i])))
                    for i in range(len(frames)) if i != half
                )
                residual_dense = statistics.fmean(
                    float(np.mean(np.abs(
                        warp_by_flow(truth, alignments[i].neighbour_to_target) - frames[i]
                    )))
                    for i in range(len(frames)) if i != half
                )

                rows.append(
                    Row(
                        clip=path.name,
                        motion_band=motion_band,
                        block=block,
                        crf=crf,
                        target=target,
                        psnr_passthrough=psnr(truth, upsample_baseline(coded[half])),
                        psnr_single=psnr(truth, single),
                        psnr_oracle=psnr(truth, oracle),
                        psnr_global=psnr(truth, global_arm),
                        psnr_dense=psnr(truth, dense),
                        psnr_dense_k3=psnr(truth, dense_k3),
                        residual_none=round(residual_none, 3),
                        residual_global=round(residual_global, 3),
                        residual_dense=round(residual_dense, 3),
                        ssim_single=ssim(truth, single),
                        ssim_dense=ssim(truth, dense),
                        flow_usable_fraction=round(usable, 4),
                    )
                )

    return rows


def _aggregate(rows: list[Row], predicate) -> dict[str, float]:
    picked = [r for r in rows if predicate(r)]
    if not picked:
        return {"n": 0}

    return {
        "n": len(picked),
        "psnr_single": round(statistics.fmean(r.psnr_single for r in picked), 3),
        "gain_oracle": round(statistics.fmean(r.gain_oracle for r in picked), 3),
        "gain_global": round(statistics.fmean(r.gain_global for r in picked), 3),
        "gain_dense": round(statistics.fmean(r.gain_dense for r in picked), 3),
        "gain_dense_k3": round(statistics.fmean(r.gain_dense_k3 for r in picked), 3),
        "gap_closed": round(statistics.fmean(r.gap_closed for r in picked), 3),
        "flow_usable": round(statistics.fmean(r.flow_usable_fraction for r in picked), 3),
        "residual_none": round(statistics.fmean(r.residual_none for r in picked), 2),
        "residual_global": round(statistics.fmean(r.residual_global for r in picked), 2),
        "residual_dense": round(statistics.fmean(r.residual_dense for r in picked), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--crfs", type=int, nargs="+", default=[18, 26])
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "phase2-alignment.json")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        raise SystemExit(f"corpus manifest missing: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    by_band: dict[str, list[dict]] = {}
    for clip in manifest["clips"]:
        by_band.setdefault(clip["motion_band"], []).append(clip)

    selected: list[dict] = []
    while len(selected) < args.clips and any(by_band.values()):
        for band in sorted(by_band):
            if by_band[band] and len(selected) < args.clips:
                selected.append(by_band[band].pop(0))

    aligner = DenseAligner()
    print(f"flow: RAFT-small on {aligner.device}", flush=True)

    rows: list[Row] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for index, clip in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {clip['name']} ({clip['motion_band']})", flush=True)
            rows.extend(
                _measure_clip(
                    CORPUS / clip["name"],
                    clip["motion_band"],
                    aligner,
                    tuple(args.blocks),
                    tuple(args.crfs),
                    args.targets,
                    tmp,
                )
            )

    overall = _aggregate(rows, lambda r: True)

    report = {
        "question": "does dense optical flow close the alignment gap the Phase 0 gate found?",
        "decision_band": overall,
        "by_motion_band": {
            band: _aggregate(rows, lambda r, band=band: r.motion_band == band)
            for band in sorted({r.motion_band for r in rows})
        },
        "by_block": {
            str(block): _aggregate(rows, lambda r, block=block: r.block == block)
            for block in sorted({r.block for r in rows})
        },
        "by_crf": {
            str(crf): _aggregate(rows, lambda r, crf=crf: r.crf == crf)
            for crf in sorted({r.crf for r in rows})
        },
        "samples": len(rows),
        "measurements": [asdict(r) for r in rows],
    }

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"n = {overall.get('n', 0)}   flow usable {overall.get('flow_usable', 0):.1%} of pixels")
    print(f"  single frame        {overall.get('psnr_single', 0):7.3f} dB")
    print(f"  gain, global shift  {overall.get('gain_global', 0):+7.3f} dB   (the gate's number)")
    print(f"  gain, dense flow    {overall.get('gain_dense', 0):+7.3f} dB")
    print(f"  gain, dense K=3     {overall.get('gain_dense_k3', 0):+7.3f} dB")
    print(f"  gain, oracle        {overall.get('gain_oracle', 0):+7.3f} dB   (the ceiling)")
    print(f"  gap closed          {overall.get('gap_closed', 0):7.1%}")
    print()
    print("mean |neighbour - target| on clean frames:")
    print(f"  unaligned {overall.get('residual_none', 0):6.2f}   "
          f"global {overall.get('residual_global', 0):6.2f}   "
          f"dense {overall.get('residual_dense', 0):6.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
