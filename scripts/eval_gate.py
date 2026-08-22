"""Does any confidence signal predict whether a restoration helped? prd.md section 5.8.1, section 13.5.

`minRestorationConfidence` exists, defaults to 0.0, and has never been calibrated. Meanwhile the
decomposition (docs/untouched-region-report.md) measured that the restoration makes things worse
everywhere it touches: -0.69 dB inside the mosaicked region, -4.21 dB on clean picture it should
not have touched at all. A gate is the cheapest possible answer to that - but only if some signal
available at runtime actually separates the restorations that help from the ones that hurt.

So this measures the separation before anything is gated, with three arms, the same way the Phase 0
feasibility gate was run:

* **ungated**   - what the pipeline does today, every restoration applied;
* **oracle**    - apply only where it actually helped. Not implementable; it is the ceiling, and a
                  signal that recovers little of it is not worth wiring in;
* **by signal** - threshold each candidate signal, take the best operating point per signal, and
                  see how much of the oracle it recovers.

The unit is one applied restoration - one track on one frame - scored on the pixels the pipeline
actually changed, against the clean original.

Usage:

    .venv/Scripts/python.exe scripts/eval_gate.py
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

from demosaic_worker.metrics import psnr  # noqa: E402
from demosaic_worker.policies import ConfidenceGate  # noqa: E402

import evalclips  # noqa: E402

CORPUS = REPO / "training" / "datasets" / "clean"
ARTIFACTS = REPO / "artifacts"


@dataclass(frozen=True, slots=True)
class Applied:
    """One restoration, with the signals that were available when it was decided."""

    frame: int
    track: int
    area: int
    pixels: int
    before: float
    after: float
    delta: float
    confidence: float
    grid_confidence: float
    mean_alignment: float
    evidence_depth: int
    block_size: int
    kind: str
    reason: str
    #: Whether this region overlaps the real mosaic at all.
    true_positive: bool = False


def _luma(path: Path, limit: int = 400) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            out.append(frame.to_ndarray(format="gray").astype(np.float64))
            if len(out) >= limit:
                break
    return out


def score_regions(records: list[dict], clean: list[np.ndarray],
                  degraded: list[np.ndarray], restored: list[np.ndarray],
                  mask_of) -> list[Applied]:
    """Scores each logged restoration on the pixels it actually changed.

    Using the pixels that changed rather than the whole box matters: the box is the ROI, the blend
    covers the mask plus a dilation, and averaging over the untouched remainder would dilute every
    effect towards zero.
    """
    applied: list[Applied] = []
    untouched_already = 0

    for record in records:
        index = record["frame"]
        if index >= min(len(clean), len(degraded), len(restored)):
            continue

        left, top, right, bottom = record["box"]
        crop = lambda a: a[index][top:bottom, left:right]  # noqa: E731

        changed = np.abs(crop(restored) - crop(degraded)) > 1
        if changed.sum() < 200:
            continue

        # The truth mask comes from the clip's own definition, not from thresholding a difference.
        # A threshold marks only the pixels the block average moved far, which is a holey mask
        # rather than the region - and reading diversity off one of those is how this project
        # overstated a measurement by an order of magnitude (D-27).
        truth = mask_of(index, clean[index].shape)[top:bottom, left:right]
        overlap = float((changed & truth).sum()) / float(changed.sum())

        before = psnr(crop(clean)[changed], crop(degraded)[changed])
        after = psnr(crop(clean)[changed], crop(restored)[changed])

        if not np.isfinite(before):
            # The input already matched the original exactly on these pixels, so there was nothing
            # to restore and "how much did the restoration help" has no answer. It happens on false
            # positives over untouched picture. Counting them as an infinite loss would make every
            # aggregate infinite, which is how this was found.
            untouched_already += 1
            continue

        applied.append(Applied(
            frame=index,
            track=record["track"],
            area=record["area"],
            pixels=int(changed.sum()),
            before=round(before, 3),
            after=round(after, 3),
            delta=round(after - before, 3),
            confidence=record["confidence"],
            grid_confidence=record["gridConfidence"],
            mean_alignment=record["meanAlignment"],
            evidence_depth=record["evidenceDepth"],
            block_size=record["blockSize"],
            kind=record["kind"],
            reason=record["reason"],
            true_positive=overlap > 0.10,
        ))

    if untouched_already:
        print(f"  {untouched_already} regions skipped: the input was already exact there")

    return applied


def weighted_delta(rows: list[Applied]) -> float:
    """Total dB moved, weighted by how many pixels each restoration touched.

    An unweighted mean lets a 300-pixel region count as much as a 50,000-pixel one, which is the
    wrong unit: the output is judged on picture, not on regions.
    """
    if not rows:
        return 0.0
    weights = np.array([r.pixels for r in rows], dtype=np.float64)
    deltas = np.array([r.delta for r in rows], dtype=np.float64)
    return float((weights * deltas).sum() / weights.sum())


def sweep(rows: list[Applied], name: str, values: list[float], higher_keeps: bool) -> dict:
    """Finds the threshold on one signal that leaves the best weighted outcome.

    `higher_keeps` says which side of the threshold gets restored. Gating is only ever an
    improvement if withholding is *better* than restoring, so the score of a gated-out region is 0
    rather than its (negative) delta.
    """
    candidates = sorted(set(values))
    best: dict = {"signal": name, "threshold": None, "weighted": weighted_delta(rows),
                  "kept": len(rows), "keptPixels": sum(r.pixels for r in rows)}

    total_pixels = sum(r.pixels for r in rows) or 1

    for threshold in candidates:
        kept = [r for r in rows
                if (getattr(r, name) >= threshold if higher_keeps
                    else getattr(r, name) <= threshold)]
        # Withheld regions contribute 0 dB: the original pixels are kept, so nothing moves.
        moved = sum(r.pixels * r.delta for r in kept) / total_pixels
        if moved > best["weighted"]:
            best = {"signal": name, "threshold": float(threshold), "weighted": round(moved, 4),
                    "kept": len(kept), "keptPixels": sum(r.pixels for r in kept)}

    return best



def sweep_shipped_gate(rows: list[Applied], thresholds: list[float]) -> dict:
    """Sweeps `minRestorationConfidence` through the gate the product actually ships.

    The plain threshold sweep above is an idealisation: it decides each region independently. The
    real gate is per track, with hysteresis and a release margin, and a track starts closed. Those
    differences are not cosmetic - a threshold of 0.88 looked best under the idealisation and turns
    out to be **unreachable**, because release needs `threshold + margin` and the confidence
    formula tops out below that. Calibrating against a model of the gate rather than the gate is
    how you ship an operating point that withholds everything.
    """
    total_pixels = sum(r.pixels for r in rows) or 1
    ordered = sorted(rows, key=lambda r: (r.frame, r.track))

    best = {"threshold": 0.0, "weighted": weighted_delta(rows), "kept": len(rows)}

    for threshold in sorted(set(thresholds)):
        gate = ConfidenceGate(threshold)
        kept = [r for r in ordered if not gate.should_withhold(r.track, r.confidence)]
        moved = sum(r.pixels * r.delta for r in kept) / total_pixels
        if moved > best["weighted"]:
            best = {"threshold": round(float(threshold), 4), "weighted": round(moved, 4),
                    "kept": len(kept)}

    return best

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    evalclips.add_argument(parser)
    parser.add_argument("--input", type=Path, help="defaults to the clip's own mosaicked input")
    parser.add_argument("--output", type=Path, required=True, help="the pipeline's output")
    parser.add_argument("--log", type=Path, required=True,
                        help="the region log the same run wrote (settings.diagnostics.regionLog)")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "gate-calibration.json")
    args = parser.parse_args(argv)

    clip = evalclips.resolve(args.clip)
    source = args.input or clip.degraded
    print(f"clip: {clip.name} - {clip.what}")

    for path in (source, args.output, args.log):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = score_regions(
        records, _luma(clip.clean), _luma(source), _luma(args.output), clip.mask
    )

    if not rows:
        print("no scorable restorations", file=sys.stderr)
        return 1

    helped = [r for r in rows if r.delta > 0]
    hurt = [r for r in rows if r.delta <= 0]

    print(f"{len(rows)} applied restorations, "
          f"{sum(r.pixels for r in rows):,} pixels changed")
    print(f"  helped  {len(helped):4}  ({sum(r.pixels for r in helped):,} px)")
    print(f"  hurt    {len(hurt):4}  ({sum(r.pixels for r in hurt):,} px)")
    print()

    # The disaggregation that decides what to fix. If the restorations on the *real* mosaic are
    # also negative, no gate can save this - the solver is the problem. If they are positive, the
    # gate has a real target and it is the false positives.
    true_positive = [r for r in rows if r.true_positive]
    false_positive = [r for r in rows if not r.true_positive]

    print(f"{'population':26} {'weighted dB':>12} {'regions':>8} {'pixels':>12} {'helped':>7}")
    for label, group in (("on the real mosaic", true_positive),
                         ("on clean picture", false_positive)):
        if not group:
            continue
        share = sum(1 for r in group if r.delta > 0) / len(group)
        print(f"{label:26} {weighted_delta(group):12.4f} {len(group):8} "
              f"{sum(r.pixels for r in group):12,} {share:7.0%}")
    print()

    ungated = weighted_delta(rows)
    oracle = sum(r.pixels * r.delta for r in helped) / (sum(r.pixels for r in rows) or 1)

    print(f"{'arm':28} {'weighted dB':>12}  {'kept':>6}")
    print(f"{'ungated (today)':28} {ungated:12.4f}  {len(rows):6}")
    print(f"{'oracle (apply only if it helped)':28} {oracle:12.4f}  {len(helped):6}")
    print()

    signals = [
        ("confidence", True),
        ("grid_confidence", True),
        ("mean_alignment", True),
        ("evidence_depth", True),
        ("block_size", False),
    ]

    results = []
    for name, higher_keeps in signals:
        values = [float(getattr(r, name)) for r in rows]
        best = sweep(rows, name, values, higher_keeps)
        best["recoveredOfOracle"] = (
            round((best["weighted"] - ungated) / (oracle - ungated), 3)
            if oracle > ungated else 0.0
        )
        results.append(best)

    # The arm that matters for shipping: the gate as it actually behaves.
    grid = [round(x / 100, 2) for x in range(0, 101)]
    shipped = sweep_shipped_gate(rows, grid)
    shipped["recoveredOfOracle"] = (
        round((shipped["weighted"] - ungated) / (oracle - ungated), 3) if oracle > ungated else 0.0
    )
    print(f"{'shipped gate (hysteresis, per track, starts closed)':52} "
          f"threshold {shipped['threshold']:.2f} -> {shipped['weighted']:+.4f} dB, "
          f"{shipped['kept']} kept, {shipped['recoveredOfOracle']:.0%} of oracle")
    print()

    print(f"{'signal':20} {'threshold':>10} {'weighted dB':>12} {'kept':>6} {'of oracle':>10}")
    for best in sorted(results, key=lambda b: -b["weighted"]):
        threshold = "-" if best["threshold"] is None else f"{best['threshold']:.4g}"
        print(f"{best['signal']:20} {threshold:>10} {best['weighted']:12.4f} "
              f"{best['kept']:6} {best['recoveredOfOracle']:10.1%}")

    # Grouped views: a categorical signal cannot be swept, but it can still separate.
    print()
    for field in ("reason", "kind"):
        print(f"by {field}:")
        groups: dict[str, list[Applied]] = {}
        for row in rows:
            groups.setdefault(getattr(row, field), []).append(row)
        for key, group in sorted(groups.items(), key=lambda kv: -sum(r.pixels for r in kv[1])):
            print(f"  {key:32} {weighted_delta(group):+7.3f} dB   "
                  f"{len(group):4} regions  {sum(r.pixels for r in group):>9,} px")
        print()

    args.out.write_text(json.dumps({
        "clip": clip.name,
        "applied": len(rows),
        "helped": len(helped),
        "ungatedWeightedDb": round(ungated, 4),
        "oracleWeightedDb": round(oracle, 4),
        "truePositive": {"regions": len(true_positive),
                         "weightedDb": round(weighted_delta(true_positive), 4)},
        "falsePositive": {"regions": len(false_positive),
                          "weightedDb": round(weighted_delta(false_positive), 4)},
        "shippedGate": shipped,
        "signals": results,
        "rows": [asdict(r) for r in rows],
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
