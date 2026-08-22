"""Generates fixtures/parity/policies.json from the *Python* implementation.

The fixture is then asserted from both sides. Generating it from one side and checking it from both
is deliberate: a fixture hand-written from the spec would drift from both implementations at once
and nobody would notice, whereas this way any disagreement is a red test on the C# side.
"""

import json
import sys
from pathlib import Path

REPO = Path("D:/Workspace/DeMosaicStudio")
sys.path.insert(0, str(REPO / "worker"))

from demosaic_worker.analyze.profile import GridAnchor
from demosaic_worker.policies import (
    ConfidenceGate,
    ConfidenceSmoother,
    QualityPreset,
    RouteInputs,
    WindowDecision,
    WindowReductionReason,
    decide_window,
    route,
)

window_cases = []
for motion in (0.0, 0.1, 0.24, 0.25, 0.5, 0.99, 1.0, 3.0, 6.0, 6.01, 12.0, 30.0):
    for preset in QualityPreset:
        for setting in (None, 3, 5, 9):
            for anchor in GridAnchor:
                for scene, stream, vram in ((9, 9, 9), (3, 9, 9), (9, 1, 9), (9, 9, 5), (6, 9, 9)):
                    d = decide_window(
                        setting=setting,
                        preset=preset,
                        motion_pixels_per_frame=motion,
                        anchor=anchor,
                        same_scene_frames=scene,
                        stream_frames=stream,
                        vram_max_window=vram,
                    )
                    window_cases.append(
                        {
                            "motion": motion,
                            "preset": preset.value,
                            "setting": setting,
                            "anchor": anchor.value,
                            "sameSceneFrames": scene,
                            "streamFrames": stream,
                            "vramMaxWindow": vram,
                            "effective": d.effective,
                            "requested": d.requested,
                            "reason": d.reason.value,
                        }
                    )

route_cases = []
for has_region in (True, False):
    for confirmed in (True, False):
        for disabled in (True, False):
            for gated in (True, False):
                for exhausted in (True, False):
                    for occluded in (True, False):
                        for anchor in GridAnchor:
                            for motion in (0.1, 0.5, 3.0, 12.0):
                                for window in (1, 3, 5):
                                    for reduction in WindowReductionReason:
                                        for neighbours in (0, 1, 4):
                                            for alignment in (0.1, 0.5, 0.8):
                                                for area in (100, 4096):
                                                    inputs = RouteInputs(
                                                        has_region=has_region,
                                                        region_area=area,
                                                        min_region_area=256,
                                                        is_confirmed=confirmed,
                                                        user_disabled=disabled,
                                                        withheld_by_confidence_gate=gated,
                                                        degradation_chain_exhausted=exhausted,
                                                        anchor=anchor,
                                                        motion_pixels_per_frame=motion,
                                                        window=WindowDecision(window, 5, reduction),
                                                        valid_aligned_neighbours=neighbours,
                                                        mean_alignment_confidence=alignment,
                                                        align_conf_min=0.35,
                                                        occlusion_invalidated_neighbours=occluded,
                                                    )
                                                    d = route(inputs)
                                                    route_cases.append(
                                                        {
                                                            "hasRegion": has_region,
                                                            "regionArea": area,
                                                            "isConfirmed": confirmed,
                                                            "userDisabled": disabled,
                                                            "withheldByConfidenceGate": gated,
                                                            "degradationChainExhausted": exhausted,
                                                            "anchor": anchor.value,
                                                            "motion": motion,
                                                            "window": window,
                                                            "windowReason": reduction.value,
                                                            "validAlignedNeighbours": neighbours,
                                                            "meanAlignmentConfidence": alignment,
                                                            "occlusionInvalidatedNeighbours": occluded,
                                                            "path": d.path.value,
                                                            "reason": d.reason.value,
                                                        }
                                                    )

# The full sweep is ~600k rows; keep a deterministic stratified sample so the fixture stays reviewable
# while still covering every (path, reason) pair and every categorical input at least once.
seen: dict[tuple[str, str], int] = {}
sampled = []
for case in route_cases:
    key = (case["path"], case["reason"])
    if seen.get(key, 0) < 12:
        seen[key] = seen.get(key, 0) + 1
        sampled.append(case)

# --- the confidence gate and smoother --------------------------------------------------------
#
# These were added to the fixture by hand once, and the next regeneration silently dropped them:
# a generator that writes only part of a file is a generator that deletes the rest. They live here
# now, so the fixture is whole whenever this runs.

GATE_SEQUENCES = [
    ("a fresh track is withheld before anything is known", 0.40, [0.1]),
    ("sustained low confidence stays withheld", 0.40, [0.1] * 6),
    ("a confident track opens after the hysteresis window", 0.40, [0.9] * 6),
    ("exactly at the threshold opens the gate", 0.40, [0.40] * 6),
    ("chattering never gets three consecutive frames, so it never opens", 0.40, [0.39, 0.41] * 8),
    ("open then a sustained fall re-closes it", 0.40, [0.9] * 4 + [0.1] * 5),
    ("a disabled gate never withholds", 0.0, [0.0] * 4),
    ("a negative threshold is also disabled", -1.0, [0.0] * 3),
    ("a gate above every confidence withholds forever", 1.01, [1.0] * 6),
    ("a dip that stays within the margin does not re-close it", 0.40, [0.9, 0.42, 0.9, 0.9, 0.9]),
]

gate_cases = []
for description, threshold, confidences in GATE_SEQUENCES:
    gate = ConfidenceGate(threshold)
    gate_cases.append({
        "description": description,
        "threshold": threshold,
        "confidences": confidences,
        "withheld": [gate.should_withhold(1, c) for c in confidences],
        "gatedTrackCount": gate.gated_track_count,
    })

gate = ConfidenceGate(0.40)
gate_interleaved = [[gate.should_withhold(1, 0.05), gate.should_withhold(2, 0.95)]
                    for _ in range(5)]

SMOOTHER_SEQUENCES = [
    ("the first frame passes through unchanged", 3, [0.9]),
    ("a steady signal converges towards it", 3, [0.9] * 6),
    ("a single dip is damped, not followed", 3, [0.9, 0.3, 0.9, 0.9]),
    ("a sustained fall is followed, just slowly", 3, [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("a window of one is no smoothing at all", 1, [0.9, 0.1, 0.5]),
    ("a longer window damps harder", 6, [0.9, 0.1, 0.1, 0.1]),
]

smoother_cases = []
for description, window, confidences in SMOOTHER_SEQUENCES:
    smoother = ConfidenceSmoother(window)
    smoother_cases.append({
        "description": description,
        "window": window,
        "confidences": confidences,
        "smoothed": [round(smoother.update(1, c), 9) for c in confidences],
    })

smoother = ConfidenceSmoother(3)
smoother_interleaved = [[round(smoother.update(1, 0.9), 9), round(smoother.update(2, 0.1), 9)]
                        for _ in range(4)]

payload = {
    "_comment": (
        "prd.md §13.4. Generated from worker/demosaic_worker/policies.py and asserted from both "
        "sides. Covers the measured window table (D-16) and the router's closed reason enum. "
        "Regenerate with scripts/make_policy_fixture.py after any deliberate policy change, and "
        "expect the C# test to fail until the mirror is updated too."
    ),
    "windowCases": window_cases,
    "routeCases": sampled,
    "routeReasonCoverage": sorted({f"{p}/{r}" for p, r in seen}),
    "confidenceGateCases": gate_cases,
    "confidenceGateInterleaved": {
        "description": "two tracks fed alternately; gate state is per track",
        "threshold": 0.40,
        "confidences": [0.05, 0.95],
        "withheld": gate_interleaved,
        "gatedTrackCount": gate.gated_track_count,
    },
    "confidenceSmootherCases": smoother_cases,
    "confidenceSmootherInterleaved": {
        "description": "two tracks fed alternately; smoother state is per track",
        "window": 3,
        "confidences": [0.9, 0.1],
        "smoothed": smoother_interleaved,
    },
}

out = REPO / "fixtures" / "parity" / "policies.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"window cases:    {len(window_cases)}")
print(f"gate cases:      {len(gate_cases)}")
print(f"smoother cases:  {len(smoother_cases)}")
print(f"route cases:  {len(sampled)} sampled from {len(route_cases)}")
print(f"coverage:     {len(seen)} (path, reason) pairs")
for pair in sorted(seen):
    print(f"  {pair[0]:12} {pair[1]}")
