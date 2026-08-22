"""The Phase 0 feasibility gate. prd.md §1.4.3, AC-1.4.

One question: **does multi-frame restoration beat single-frame on representative material?**

The whole product architecture (§19) assumes the answer is yes. This script measures it, and the
answer is allowed to be no.

Method
------
For each clean clip, for each block size and grid anchoring:

1. Crop a fixed ROI in *frame* coordinates. A screen-anchored grid is fixed there too, so content
   moving through the ROI is exactly the phase diversity §1.4.1 describes.
2. Apply the mosaic, with the phase either fixed to the frame (SCREEN) or riding the content
   (OBJECT).
3. **Re-encode through H.264 at a CRF ladder.** This is mandatory (§11.3): codec quantisation
   erases the tiny inter-block variations that multi-frame solving depends on, and a gate run on
   clean synthetic mosaics would report a number the real pipeline can never reproduce.
4. Reconstruct with the *same* solver in three arms, differing only in what the neighbours are:

   * **single** — K=1. The floor for "temporal evidence adds nothing".
   * **oracle** — K=5 neighbours synthesised from the target frame at known shifts. This is the
     information-theoretic upper bound: perfect alignment, real content, real codec.
   * **estimated** — K=5 *actual* neighbouring frames, aligned by the global-translation model,
     with badly aligned neighbours excluded exactly as §5.7/§5.8 require.

   The two multi-frame arms exist because a single one cannot distinguish "the information is not
   there" from "we could not align well enough to use it", and those two lead to opposite decisions.
   A gate that conflated them would report KILL for a solvable alignment problem.

5. Score every arm against the untouched clip, plus pass-through as the do-nothing floor (§12.3).

Noise floor
-----------
Before comparing anything, the identical configuration is run twice and the spread recorded
(§13.5). A difference smaller than that spread is not a result. Here the nondeterminism comes from
the encoder rather than from a GPU, but the discipline is the same and the reason is identical:
without it, the first plausible number gets believed.

Usage::

    .venv/Scripts/python.exe scripts/eval_multiframe_gate.py --clips 12
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

from demosaic_worker.analyze.profile import GridAnchor, MosaicProfile, band_for  # noqa: E402
from demosaic_worker.analyze.motion import cumulative_content_shifts, summarize  # noqa: E402
from demosaic_worker.metrics import psnr, shift_bilinear, ssim  # noqa: E402
from demosaic_worker.restore.ibp import Observation, block_average, reconstruct, upsample_baseline  # noqa: E402

FFMPEG = REPO / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
CORPUS = REPO / "training" / "datasets" / "clean"
MANIFEST = REPO / "training" / "datasets" / "clean-tos.manifest.json"

#: prd.md §1.4.3 decides on the 6-12 band; the others are measured for context.
BLOCK_SIZES = (4, 8, 12, 20)

#: prd.md §1.4.3 thresholds.
PASS_PSNR_DB = 1.0
PASS_SSIM = 0.0  # SSIM is reported; the PRD's LPIPS threshold cannot be evaluated yet (see below)

WINDOW = 5
ROI = 256


#: A neighbour is used only if alignment reduced its residual by at least this much. Mirrors
#: `align_conf_min` (§5.7): neighbours below the bar are *excluded*, not down-weighted, because
#: fusing a misaligned frame is worse than not fusing it at all.
ALIGN_IMPROVEMENT_MIN = 0.80


@dataclass(frozen=True, slots=True)
class Measurement:
    """One (clip, block size, anchoring, CRF, target frame) result."""

    clip: str
    motion_band: str
    block: int
    anchor: str
    crf: int
    target: int
    psnr_passthrough: float
    psnr_single: float
    psnr_multi_oracle: float
    psnr_multi_estimated: float
    ssim_single: float
    ssim_multi_oracle: float
    neighbours_offered: int
    neighbours_aligned: int
    align_ratio_median: float

    @property
    def gain_oracle(self) -> float:
        """Upper bound: what perfect alignment would deliver. This is what §1.4 is really asking."""
        return self.psnr_multi_oracle - self.psnr_single

    @property
    def gain_estimated(self) -> float:
        """What the global-translation alignment model actually delivered."""
        return self.psnr_multi_estimated - self.psnr_single


def _load_luma(path: Path, roi: int) -> list[np.ndarray]:
    """Decodes a clip's luma and crops a fixed centre ROI in frame coordinates."""
    frames: list[np.ndarray] = []

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            plane = frame.to_ndarray(format="gray").astype(np.float64)
            height, width = plane.shape
            top = (height - roi) // 2
            left = (width - roi) // 2
            frames.append(plane[top : top + roi, left : left + roi])

    return frames


def _phase_for(spec: MosaicSpec, anchor: GridAnchor, shift: tuple[float, float]) -> tuple[int, int]:
    """Grid phase for one frame under the given anchoring.

    SCREEN keeps the phase fixed in frame coordinates, so moving content crosses block boundaries.
    OBJECT slides the grid with the content, so the same pixels land in the same block every time
    and no new information ever arrives — §1.4.1's dead case, measured rather than assumed.
    """
    if anchor is GridAnchor.SCREEN:
        return spec.grid_offset_x, spec.grid_offset_y

    dx, dy = shift
    return (
        int(round(spec.grid_offset_x + dx)) % spec.block_width,
        int(round(spec.grid_offset_y + dy)) % spec.block_height,
    )


def _recompress(frames: list[np.ndarray], crf: int, tmp: Path) -> list[np.ndarray]:
    """Round-trips a luma sequence through H.264 at the given CRF."""
    height, width = frames[0].shape
    raw = tmp / f"deg_{crf}.y4m"
    out = tmp / f"deg_{crf}.mp4"

    with raw.open("wb") as handle:
        handle.write(f"YUV4MPEG2 W{width} H{height} F24:1 Ip A1:1 Cmono\n".encode("ascii"))
        for frame in frames:
            handle.write(b"FRAME\n")
            handle.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())

    subprocess.run(
        [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(raw),
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    decoded = _load_luma(out, min(height, width)) if height == width else _decode_all(out)

    raw.unlink(missing_ok=True)
    out.unlink(missing_ok=True)

    return decoded


def _decode_all(path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="gray").astype(np.float64))
    return frames


def _oracle_shifts(block: int, window: int) -> list[tuple[float, float]]:
    """Known shifts for the oracle arm, spread across the block so phases are distinct.

    Fixed rather than random: the gate must be reproducible, and a random draw that happened to
    land several neighbours on the same phase would understate the upper bound.
    """
    step = block / window
    return [(0.0, 0.0)] + [
        (round(step * k, 3), round(step * (window - k) / 2.0, 3)) for k in range(1, window)
    ]


def _alignment_quality(
    frames: list[np.ndarray],
    shifts: list[tuple[float, float]],
    target_index: int,
) -> tuple[int, list[int], list[float]]:
    """Scores how well a global translation explains each neighbour.

    Returns ``(offered, kept_indices, ratios)`` where a ratio is the post-alignment residual over
    the un-aligned residual. A ratio near or above 1 means the motion model did not explain the
    frame, and fusing it would back-project other people's pixels into the estimate.
    """
    target = frames[target_index]
    kept: list[int] = []
    ratios: list[float] = []

    for index, frame in enumerate(frames):
        if index == target_index:
            continue

        dx, dy = shifts[index]
        aligned = shift_bilinear(target, dx, dy)

        residual_after = float(np.mean(np.abs(aligned - frame)))
        residual_before = float(np.mean(np.abs(target - frame)))

        ratio = residual_after / residual_before if residual_before > 1e-9 else 1.0
        ratios.append(ratio)

        if ratio <= ALIGN_IMPROVEMENT_MIN:
            kept.append(index)

    return len(frames) - 1, kept, ratios


def _measure_clip(
    clip_path: Path,
    motion_band: str,
    blocks: tuple[int, ...],
    crfs: tuple[int, ...],
    targets: int,
    tmp: Path,
) -> list[Measurement]:
    clean = _load_luma(clip_path, ROI)
    if len(clean) < WINDOW + 2:
        return []

    half = WINDOW // 2
    positions = np.linspace(half, len(clean) - half - 1, targets).astype(int).tolist()

    results: list[Measurement] = []

    for block in blocks:
        spec = MosaicProfile(block_width=block, block_height=block, grid_offset_x=0, grid_offset_y=0)

        for anchor in (GridAnchor.SCREEN, GridAnchor.OBJECT):
            for target in positions:
                window = list(range(target - half, target + half + 1))
                window_frames = [clean[i] for i in window]

                shifts = cumulative_content_shifts(window_frames, half)

                degraded = [
                    block_average(frame, spec, _phase_for(spec, anchor, shift))
                    for frame, shift in zip(window_frames, shifts, strict=True)
                ]

                truth = clean[target]
                phase = _phase_for(spec, anchor, (0.0, 0.0))

                # Alignment quality, measured on the *clean* frames so it reports the motion model's
                # fitness rather than the codec's noise.
                offered, kept, ratios = _alignment_quality(window_frames, shifts, half)

                # Oracle arm: neighbours synthesised from the target at known shifts. Same content,
                # same block size, same codec — only the alignment error is removed.
                oracle_shifts = _oracle_shifts(block, WINDOW)
                oracle_degraded = [
                    block_average(
                        shift_bilinear(truth, dx, dy) if (dx or dy) else truth,
                        spec,
                        _phase_for(spec, anchor, (dx, dy) if anchor is GridAnchor.OBJECT else (0.0, 0.0)),
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

                    oracle_obs = [
                        Observation(coded_oracle[i], oracle_shifts[i][0], oracle_shifts[i][1])
                        for i in range(len(coded_oracle))
                    ]
                    multi_oracle = reconstruct(oracle_obs, spec, phase, iterations=30).image

                    estimated_obs = [Observation(coded[half], 0.0, 0.0)]
                    estimated_obs.extend(
                        Observation(coded[i], shifts[i][0], shifts[i][1])
                        for i in kept
                    )
                    multi_estimated = reconstruct(estimated_obs, spec, phase, iterations=30).image

                    results.append(
                        Measurement(
                            clip=clip_path.name,
                            motion_band=motion_band,
                            block=block,
                            anchor=anchor.value,
                            crf=crf,
                            target=target,
                            psnr_passthrough=psnr(truth, upsample_baseline(coded[half])),
                            psnr_single=psnr(truth, single),
                            psnr_multi_oracle=psnr(truth, multi_oracle),
                            psnr_multi_estimated=psnr(truth, multi_estimated),
                            ssim_single=ssim(truth, single),
                            ssim_multi_oracle=ssim(truth, multi_oracle),
                            neighbours_offered=offered,
                            neighbours_aligned=len(kept),
                            align_ratio_median=round(
                                float(np.median(ratios)) if ratios else float("nan"), 3
                            ),
                        )
                    )

    return results


def _aggregate(rows: list[Measurement], predicate) -> dict[str, float]:
    selected = [r for r in rows if predicate(r)]
    if not selected:
        return {"n": 0}

    return {
        "n": len(selected),
        "psnr_passthrough": round(statistics.fmean(r.psnr_passthrough for r in selected), 3),
        "psnr_single": round(statistics.fmean(r.psnr_single for r in selected), 3),
        "psnr_multi_oracle": round(statistics.fmean(r.psnr_multi_oracle for r in selected), 3),
        "psnr_multi_estimated": round(statistics.fmean(r.psnr_multi_estimated for r in selected), 3),
        "gain_oracle": round(statistics.fmean(r.gain_oracle for r in selected), 3),
        "gain_estimated": round(statistics.fmean(r.gain_estimated for r in selected), 3),
        "ssim_single": round(statistics.fmean(r.ssim_single for r in selected), 4),
        "ssim_multi_oracle": round(statistics.fmean(r.ssim_multi_oracle for r in selected), 4),
        "neighbours_offered": sum(r.neighbours_offered for r in selected),
        "neighbours_aligned": sum(r.neighbours_aligned for r in selected),
        "align_usable_fraction": round(
            sum(r.neighbours_aligned for r in selected)
            / max(1, sum(r.neighbours_offered for r in selected)),
            3,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=12)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--crfs", type=int, nargs="+", default=[18, 26])
    parser.add_argument("--blocks", type=int, nargs="+", default=list(BLOCK_SIZES))
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "phase0-gate.json")
    args = parser.parse_args(argv)

    if not FFMPEG.exists():
        raise SystemExit(f"FFmpeg not found at {FFMPEG}. Run scripts/setup-worker.ps1 first.")
    if not MANIFEST.exists():
        raise SystemExit(f"corpus manifest missing: {MANIFEST}. Run scripts/build_corpus.py first.")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # Spread the selection across motion bands rather than taking the first N, so a gate run with
    # few clips does not accidentally become an all-static or all-fast measurement.
    by_band: dict[str, list[dict]] = {}
    for clip in manifest["clips"]:
        by_band.setdefault(clip["motion_band"], []).append(clip)

    selected: list[dict] = []
    while len(selected) < args.clips and any(by_band.values()):
        for band in sorted(by_band):
            if by_band[band] and len(selected) < args.clips:
                selected.append(by_band[band].pop(0))

    rows: list[Measurement] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for index, clip in enumerate(selected, start=1):
            path = CORPUS / clip["name"]
            print(f"[{index}/{len(selected)}] {clip['name']}  ({clip['motion_band']})", flush=True)
            rows.extend(
                _measure_clip(
                    path,
                    clip["motion_band"],
                    tuple(args.blocks),
                    tuple(args.crfs),
                    args.targets,
                    tmp,
                )
            )

        # --- noise floor (§13.5): the identical configuration, run again -------------------------
        print("measuring the noise floor (same configuration, second run)", flush=True)
        repeat: list[Measurement] = []
        for clip in selected[: max(1, len(selected) // 3)]:
            repeat.extend(
                _measure_clip(
                    CORPUS / clip["name"],
                    clip["motion_band"],
                    tuple(args.blocks),
                    tuple(args.crfs),
                    args.targets,
                    tmp,
                )
            )

    by_key = {(r.clip, r.block, r.anchor, r.crf, r.target): r for r in rows}
    deltas = [
        abs(r.gain_oracle - by_key[(r.clip, r.block, r.anchor, r.crf, r.target)].gain_oracle)
        for r in repeat
        if (r.clip, r.block, r.anchor, r.crf, r.target) in by_key
    ]
    noise_floor = round(max(deltas), 4) if deltas else 0.0

    # --- the decision band: screen-anchored, blocks 6-12 (§1.4.3) --------------------------------
    decision = _aggregate(rows, lambda r: r.anchor == "SCREEN" and 6 <= r.block <= 12)
    gain_oracle = decision.get("gain_oracle", 0.0)
    gain_estimated = decision.get("gain_estimated", 0.0)

    # The oracle arm decides whether the *information* is recoverable; the estimated arm decides
    # whether our current alignment can reach it. Only the first can kill the product (§1.4.3).
    if decision.get("n", 0) == 0:
        verdict = "INCONCLUSIVE"
        rationale = "no measurements in the decision band"
    elif gain_oracle <= noise_floor:
        verdict = "KILL"
        rationale = (
            "even with perfect alignment, multi-frame does not beat single-frame beyond the "
            "noise floor: the information is not in the neighbouring frames"
        )
    elif gain_oracle < PASS_PSNR_DB:
        verdict = "MARGINAL"
        rationale = "multi-frame helps, but by less than the 1.0 dB threshold even with perfect alignment"
    elif gain_estimated <= noise_floor:
        verdict = "PASS_ALIGNMENT_BLOCKED"
        rationale = (
            f"the information is recoverable ({gain_oracle:+.2f} dB with perfect alignment), but the "
            "global-translation model cannot reach it. Alignment (prd.md §5.7), not the reconstruction "
            "model, is Phase 2's critical path"
        )
    else:
        verdict = "PASS"
        rationale = "multi-frame beats single-frame with realistic alignment"

    report = {
        "verdict": verdict,
        "rationale": rationale,
        "noise_floor_psnr_db": noise_floor,
        "thresholds": {
            "psnr_gain_db": PASS_PSNR_DB,
            "align_improvement_min": ALIGN_IMPROVEMENT_MIN,
        },
        "decision_band": {"anchor": "SCREEN", "blocks": "6-12", **decision},
        "by_block": {
            str(b): {
                "band": band_for(b),
                "screen": _aggregate(rows, lambda r, b=b: r.anchor == "SCREEN" and r.block == b),
                "object": _aggregate(rows, lambda r, b=b: r.anchor == "OBJECT" and r.block == b),
            }
            for b in sorted({r.block for r in rows})
        },
        "by_motion_band": {
            band: _aggregate(rows, lambda r, band=band: r.anchor == "SCREEN" and r.motion_band == band)
            for band in sorted({r.motion_band for r in rows})
        },
        "by_crf": {
            str(crf): _aggregate(rows, lambda r, crf=crf: r.anchor == "SCREEN" and r.crf == crf)
            for crf in sorted({r.crf for r in rows})
        },
        "not_measured": [
            "LPIPS — needs a pretrained network; arrives with Phase 2 (prd.md §12.3)",
            "warping error on reconstructed sequences — the gate reconstructs isolated target frames",
        ],
        "samples": len(rows),
        "measurements": [asdict(r) for r in rows],
    }

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"noise floor        {noise_floor:+.3f} dB")
    print(f"decision band      n={decision.get('n', 0)}")
    print(f"  passthrough      {decision.get('psnr_passthrough', 0.0):7.3f} dB")
    print(f"  single frame     {decision.get('psnr_single', 0.0):7.3f} dB")
    print(f"  multi, oracle    {decision.get('psnr_multi_oracle', 0.0):7.3f} dB   gain {gain_oracle:+.3f}")
    print(f"  multi, estimated {decision.get('psnr_multi_estimated', 0.0):7.3f} dB   gain {gain_estimated:+.3f}")
    print(f"  neighbours usable {decision.get('align_usable_fraction', 0.0):.1%}")
    print()
    print(f"VERDICT            {verdict}")
    print(f"                   {rationale}")
    try:
        shown = args.out.relative_to(REPO)
    except ValueError:
        shown = args.out
    print(f"report             {shown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
