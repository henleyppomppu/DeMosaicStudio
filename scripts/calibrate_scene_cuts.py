"""Calibrates the scene-cut thresholds from the corpus. prd.md section 5.12, section 12.

The thresholds in `demosaic_worker/scene/cuts.py` started as guesses and a guess is not allowed to
stay (section 12.7). This measures the two distances on:

* **within-shot pairs** - consecutive frames of one clip. These must not be cuts.
* **across-shot pairs** - the first frame of one clip against the first frame of another. Each clip
  comes from a different point in the film, so these are genuine cuts.
* **flash pairs** - a real frame against itself scaled in luminance. These must not be cuts.

and reports the separation, so the thresholds are set where the distributions actually divide.

Usage::

    .venv/Scripts/python.exe scripts/calibrate_scene_cuts.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import av
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "worker"))

from demosaic_worker.scene.cuts import histogram_distance, structure_distance  # noqa: E402

MANIFEST = REPO / "training" / "datasets" / "clean-tos.manifest.json"
CORPUS = REPO / "training" / "datasets" / "clean"


def _frames(path: Path, count: int, stride: int = 4) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            out.append(frame.to_ndarray(format="gray").astype(np.float64)[::stride, ::stride])
            if len(out) >= count:
                break
    return out


def _describe(name: str, values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "name": name,
        "n": len(values),
        "min": round(values[0], 4),
        "p05": round(values[int(0.05 * len(values))], 4),
        "median": round(statistics.median(values), 4),
        "p95": round(values[int(0.95 * len(values))], 4),
        "max": round(values[-1], 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "scene-cut-calibration.json")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        raise SystemExit(f"corpus manifest missing: {MANIFEST}")

    names = [c["name"] for c in json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]]
    clips = {n: _frames(CORPUS / n, args.frames_per_clip) for n in names}

    within_hist: list[float] = []
    within_struct: list[float] = []
    for frames in clips.values():
        for a, b in zip(frames, frames[1:], strict=False):
            within_hist.append(histogram_distance(a, b))
            within_struct.append(structure_distance(a, b))

    across_hist: list[float] = []
    across_struct: list[float] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = clips[first][0], clips[second][0]
            across_hist.append(histogram_distance(a, b))
            across_struct.append(structure_distance(a, b))

    flash_hist: list[float] = []
    flash_struct: list[float] = []
    for frames in clips.values():
        base = frames[0]
        for gain, offset in ((2.6, 60.0), (0.35, 0.0), (1.8, 30.0)):
            flashed = np.clip(base * gain + offset, 0, 255)
            flash_hist.append(histogram_distance(base, flashed))
            flash_struct.append(structure_distance(base, flashed))

    report = {
        "frames_per_clip": args.frames_per_clip,
        "clips": len(names),
        "histogram": [
            _describe("within-shot", within_hist),
            _describe("across-shot", across_hist),
            _describe("flash", flash_hist),
        ],
        "structure": [
            _describe("within-shot", within_struct),
            _describe("across-shot", across_struct),
            _describe("flash", flash_struct),
        ],
    }

    # Each threshold is set against the population it can actually be confused with, which is not
    # the same population for the two signals:
    #
    #   histogram  — a flash scores *high*, so it cannot constrain this threshold. What must stay
    #                below it is an ordinary continuation.
    #   structure  — a continuation scores *low* and so does a flash, but the flash is the one that
    #                comes closest to a cut, so it is what sets the floor.
    #
    # Note that "within-shot" here is really "within-clip": some 4 s corpus clips contain a cut, and
    # those pairs correctly score high. That contamination inflates the within-shot p95 and is the
    # reason the structure floor is taken from the flash population instead.
    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    hist_floor = pct(within_hist, 0.95)
    hist_ceiling = pct(across_hist, 0.05)

    struct_floor = pct(flash_struct, 0.95)
    struct_ceiling = pct(across_struct, 0.05)

    report["suggested"] = {
        "histogram_threshold": round((hist_floor + hist_ceiling) / 2, 3),
        "histogram_window": [round(hist_floor, 4), round(hist_ceiling, 4)],
        "histogram_separable": hist_floor < hist_ceiling,
        "structure_threshold": round((struct_floor + struct_ceiling) / 2, 3),
        "structure_window": [round(struct_floor, 4), round(struct_ceiling, 4)],
        "structure_separable": struct_floor < struct_ceiling,
    }

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for metric in ("histogram", "structure"):
        print(f"=== {metric} ===")
        print(f"{'population':14} {'n':>6} {'min':>8} {'p05':>8} {'median':>8} {'p95':>8} {'max':>8}")
        for row in report[metric]:
            print(
                f"{row['name']:14} {row['n']:6d} {row['min']:8.4f} {row['p05']:8.4f} "
                f"{row['median']:8.4f} {row['p95']:8.4f} {row['max']:8.4f}"
            )
        print()

    s = report["suggested"]
    print(f"histogram: {s['histogram_window']} -> {s['histogram_threshold']}  separable={s['histogram_separable']}")
    print(f"structure: {s['structure_window']} -> {s['structure_threshold']}  separable={s['structure_separable']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
