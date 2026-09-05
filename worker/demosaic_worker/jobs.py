"""Job execution — the pipeline, assembled. prd.md §3.1, §5.13.

This is where the stages become a product. The order below is §18's processing flow, and the two
invariants it exists to preserve are worth stating before the code:

* **A frame is either fully restored or bit-identical to its source** (§5.8.2). Every failure path
  degrades to a lower-effort restoration and finally to the untouched frame. There is no partial
  composite.
* **The output timeline is the source timeline** (§5.1.7). The restoration stages never see a
  timestamp; they receive pixels and return pixels, and the media layer carries the PTS through.

The pipeline runs a **sliding window** in presentation order: a frame cannot be restored until its
neighbours have been decoded, so the writer trails the reader by ``K // 2`` frames. That lag is
latency, not a stall (§5.6).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .analyze.estimator import AnchorObservation, estimate_anchor, estimate_geometry
from .analyze.motion import classify
from .analyze.profile import GridAnchor, MosaicProfile, band_for
from .detect.regions import Region, extract_regions
from .errors import (
    E1001,
    E3001,
    E3002,
    E4001,
    E4002,
    E7006,
    W3101,
    W4102,
    W4103,
    W5102,
    W6101,
    WorkerError,
)
from .media.probe import probe as probe_media
from .messages import Emitter
from .policies import (
    ConfidenceGate,
    ConfidenceSmoother,
    QualityPreset,
    RestorationPath,
    RouteInputs,
    decide_window,
    route,
)
from .post.blend import TemporalAlpha, blend_region
from .protocol import Stage
from .roi import build_roi
from .restore.accumulator import EvidenceAccumulator
from .restore.refine import DiffusionRefiner, RefineSettings, RefineStats, chroma_for
from .restore.temporal import DEFAULT_ALPHA, TemporalSmoother
from .restore.upscale import RESTORER_ID, Upscaler, bicubic_restore
from .scene.cuts import SceneChange, classify_pair, same_scene_span
from .track.tracker import Track, Tracker

REPO = Path(__file__).resolve().parent.parent.parent
MODELS = REPO / "models"


@dataclass(frozen=True, slots=True)
class TrackOutcome:
    """What restoring one track on one frame produced.

    Carries the ROI and the profile as well as the pixels: the caller needs the ROI to know where to
    composite and the profile's block size to set the blender's dilation allowance (§5.11), and
    threading them back out beats recomputing either.
    """

    decision: Any
    restored: np.ndarray | None
    confidence: float
    roi: Any
    profile: MosaicProfile
    #: Mean per-pixel alignment quality over the usable neighbours. Diagnostic only.
    mean_alignment: float = 0.0
    #: How many neighbours survived alignment. Diagnostic only.
    usable_neighbours: int = 0


@dataclass
class JobContext:
    """One job's request and mutable state. Owned by the dispatch loop (§8.5)."""

    job_id: str
    source_path: str
    output_path: str
    settings: dict[str, Any] = field(default_factory=dict)
    resume: bool = False
    comparison_pts: list[int] = field(default_factory=list)
    analyze_only: bool = False
    sample_every: int = 1

    cancelled: bool = False
    paused: bool = False
    finished: bool = False

    frames_seen: int = 0
    frames_restored: int = 0
    frames_passed_through: int = 0
    frames_with_regions: int = 0
    regions_detected: int = 0
    regions_gated: int = 0
    route_reasons: dict[str, int] = field(default_factory=dict)
    confidences: list[float] = field(default_factory=list)
    passthrough: bool = False

    def note_route(self, reason: str) -> None:
        """Counts a routing reason for the job summary. A router that cannot explain itself…"""
        self.route_reasons[reason] = self.route_reasons.get(reason, 0) + 1

    def summary(self) -> dict[str, Any]:
        """What the host reports to the user."""
        mean_confidence = float(np.mean(self.confidences)) if self.confidences else 0.0
        return {
            "framesSeen": self.frames_seen,
            "framesRestored": self.frames_restored,
            "framesPassedThrough": self.frames_passed_through,
            "regionsDetected": self.regions_detected,
            "regionsGated": self.regions_gated,
            "routeReasons": dict(sorted(self.route_reasons.items())),
            "confidenceMean": round(mean_confidence, 4),
            "framesWithRegions": self.frames_with_regions,
            # Set by the runner once it knows what it actually wrote. It used to be
            # `regions_detected == 0`, which was a claim about the *decision* rather than about the
            # bytes: the video was fully re-encoded either way, and the summary said it had not
            # been. R-1.8a is a stream copy or it is nothing.
            "passthrough": self.passthrough,
            "synthetic": self.frames_restored > 0,
        }


def _setting(settings: dict[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = settings
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


class JobRunner:
    """Runs jobs. One instance per worker process; models are loaded lazily and cached."""

    def __init__(self, models_root: Path | None = None) -> None:
        self.models_root = models_root or MODELS
        self._segmenter: Any = None
        self._aligner: Any = None

    # --- capability reporting (§8.3) -------------------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """What this build can do.

        Availability means "we loaded it and ran something", never "a driver reported a device".
        A device count from a driver query is not availability: models load and *then* fail.
        """
        report: dict[str, Any] = {"cudaAvailable": False, "models": self._available_models()}

        try:
            import torch

            if torch.cuda.is_available():
                probe = torch.randn(64, 64, device="cuda")
                _ = float((probe @ probe).sum().item())
                torch.cuda.synchronize()
                report["cudaAvailable"] = True
                report["device"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                report["vramFreeBytes"] = int(free)
                report["vramTotalBytes"] = int(total)
        except Exception as exc:  # noqa: BLE001 - an unavailable GPU is a fact, not a failure
            report["cudaError"] = str(exc)

        return report

    def _available_models(self) -> list[dict[str, str]]:
        index = self.models_root / "index.json"
        if not index.exists():
            return []
        return json.loads(index.read_text(encoding="utf-8")).get("models", [])

    def _model_directory(self, task: str, version: str | None = None) -> Path | None:
        """Resolves a task to a model directory, honouring an explicit version.

        Without ``version`` the highest available version wins. Leaving the choice to store order
        would make the model that ran depend on how `index.json` happened to be written, which is
        exactly the kind of thing that makes a benchmark irreproducible.
        """
        candidates = [e for e in self._available_models() if e.get("task") == task]
        if not candidates:
            return None

        if version is not None:
            for entry in candidates:
                if entry.get("version") == version:
                    return self.models_root / entry["path"]
            raise WorkerError(
                E3001,
                f"no {task} model at version {version} in the store",
                available=[e.get("version") for e in candidates],
            )

        best = max(candidates, key=lambda e: tuple(int(p) for p in e["version"].split(".")))
        return self.models_root / best["path"]

    # --- probe (§8.3) ----------------------------------------------------------------------------

    def probe(self, source_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Media and hardware facts, with no processing."""
        info = probe_media(Path(source_path))

        media = {
            "path": info.path,
            "container": info.container,
            "sizeBytes": info.size_bytes,
            "durationSeconds": info.duration_seconds,
            "width": info.width,
            "height": info.height,
            "nominalFps": float(info.nominal_fps) if info.nominal_fps else None,
            "isVfr": info.is_vfr,
            "videoCodec": info.video_codec,
            "pixelFormat": info.pixel_format,
            "rotation": info.rotation,
            "audioStreams": [
                {"index": s.index, "codec": s.codec, "sampleRate": s.sample_rate,
                 "channels": s.channels, "language": s.language}
                for s in info.audio_streams
            ],
            "subtitleStreams": info.subtitle_streams,
        }

        return media, self.capabilities()

    def preview(
        self,
        job_id: str,
        source_path: str,
        pts: int,
        settings: dict[str, Any],
        *,
        overlay: bool = False,
    ) -> dict[str, Any]:
        """Renders one frame. Not implemented yet; reported honestly rather than faked."""
        raise WorkerError(
            E1001,
            "preview is not implemented",
            jobId=job_id,
            pts=pts,
            source=Path(source_path).name,
            overlay=overlay,
        )

    # --- the pipeline ----------------------------------------------------------------------------

    def _segment(self, luma: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
        if self._segmenter is None:
            from .detect.segmenter import Segmenter

            directory = self._model_directory(
                "mosaic-segmentation", _setting(settings, "modelVersion")
            )
            if directory is None:
                raise WorkerError(E3001, "no mosaic-segmentation model in the store")

            self._segmenter = Segmenter(
                directory, device=_setting(settings, "device", default="auto")
            )

        return self._segmenter.probability(luma)

    def _align(self, target: np.ndarray, neighbour: np.ndarray, *, backward: bool = True) -> Any:
        if self._aligner is None:
            from .restore.flow import DenseAligner

            self._aligner = DenseAligner()

        return self._aligner.align(target, neighbour, backward=backward)

    def run(self, context: JobContext, emitter: Emitter) -> dict[str, Any]:
        """Runs one job to completion, cancellation, or a numbered failure."""
        source = Path(context.source_path)
        if not source.exists():
            raise WorkerError(E1001, f"source not found: {source.name}")

        if not context.analyze_only and not context.output_path:
            # `analyze` writes nothing by design, so only `process` needs a destination. Without
            # this the empty string became Path("") and failed much later, inside the muxer.
            raise WorkerError(E7006, "process requires an outputPath")

        settings = context.settings
        # Fast by default since D-43: measured best on the footage in hand (see the ADR).
        preset = QualityPreset(_setting(settings, "restoration", "preset", default="Fast"))
        window_setting = _setting(settings, "restoration", "temporalWindow", default="auto")
        window_setting = None if window_setting in (None, "auto") else int(window_setting)
        min_region_area = int(_setting(settings, "detection", "minRegionArea", default=256))
        # Two different settings, and conflating them was a bug: §5.2.3 lists "detection confidence"
        # (which region survives into a track) and "mask binarization threshold" (which pixel is in
        # the mask) separately. The mask threshold was hard-coded to 0.5 and ignored the settings
        # entirely, so the calibrated operating point could not be applied at all.
        detection_threshold = float(_setting(settings, "detection", "confidence", default=0.45))
        mask_threshold = float(_setting(settings, "detection", "maskThreshold", default=0.5))
        min_confirm = int(_setting(settings, "detection", "minConfirmFrames", default=2))
        max_missing = int(_setting(settings, "detection", "maxMissingFrames", default=3))
        align_conf_min = float(_setting(settings, "restoration", "alignConfMin", default=0.35))
        feather_px = int(_setting(settings, "restoration", "featherWidth", default=3))

        gate = ConfidenceGate(
            float(_setting(settings, "restoration", "minRestorationConfidence", default=0.0))
        )
        smoother = ConfidenceSmoother()
        accumulator = EvidenceAccumulator()

        # An opt-in, per-region diagnostic trace. This is how the gate's operating point gets
        # measured: without it there is no way to ask which confidence signal predicts whether a
        # restoration helped. It is a diagnostic knob, so it stays out of the settings fingerprint
        # (section 9.3) - turning it on must not discard anyone's cached work.
        region_log_path = _setting(settings, "diagnostics", "regionLog")
        region_log = (
            Path(region_log_path).open("w", encoding="utf-8") if region_log_path else None
        )
        tracker = Tracker(
            high_confidence=detection_threshold,
            min_confirm_frames=min_confirm,
            max_missing_frames=max_missing,
        )
        temporal_alpha = TemporalAlpha()

        # D-43: the preset chooses the restorer, and for the first time it chooses something -
        # D-31 measured Fast, Balanced and Quality producing identical output.
        #
        #   Fast      decimate + bicubic. No network. Removes the grid, leaves blur, ~1 ms.
        #   Balanced  decimate + compact SR network, every region in one batch, + temporal blend.
        #             Invents detail. Falls back to Fast with W6101 when no weights are installed.
        #   Quality   the evidence accumulator (D-28): optical flow + fold. The only path that
        #             uses neighbouring frames, and about 20x slower for it.
        backend = self._backend_for(preset, emitter)
        temporal = TemporalSmoother(
            alpha=float(_setting(settings, "restoration", "temporalAlpha", default=DEFAULT_ALPHA))
        )
        # Run the detector on every Nth frame and let the tracker carry regions between. The
        # detector is the largest fixed cost per frame once restoration is cheap (67 ms of a
        # ~90 ms frame at 1080p), and mosaic regions do not appear and vanish between adjacent
        # frames. `minConfirmFrames` already delays the first restoration by two frames, so a
        # region found one frame late costs nothing the pipeline was not already spending.
        detect_every = max(1, int(_setting(settings, "detection", "detectEvery", default=1)))

        # D-44: an optional diffusion pass over each restored region, at low strength, with the
        # user's own model from the store. Off by default; costs nothing when off. The pipeline
        # loads on first use, and a load failure downgrades the job to unrefined with W6101 once
        # rather than failing it - the restoration underneath is complete without it.
        refine_settings = RefineSettings.from_settings(settings)
        refiner = DiffusionRefiner(MODELS, refine_settings) if refine_settings.enabled else None
        refine_stats = RefineStats()

        emitter.progress(context.job_id, Stage.PROBING, 0.0, force=True)
        info = probe_media(source)
        total = int((info.duration_seconds or 0) * float(info.nominal_fps or 24)) or None

        from .media.passthrough import (
            StreamCopyUnavailable,
            run_analysis,
            run_passthrough,
            run_stream_copy,
        )

        # A sliding window in presentation order (§5.6). The transform below sees each frame with
        # its already-decoded neighbours; the media layer carries the timeline.
        radius = 4
        history: list[np.ndarray] = []
        # Where each frame in `history` was found to be mosaicked. The forward model needs it: a
        # neighbour observes the scene *directly* wherever its own mask does not cover, and that is
        # the strongest evidence about content the target has lost. Measured on synthetic content
        # with oracle alignment, modelling it is worth +1.0 to +4.4 dB against the mosaicked input,
        # where modelling every pixel as block-averaged was worth nothing at all.
        mask_history: list[np.ndarray] = []
        # Scene boundaries inside `history`, kept in step with it. One new pair is classified per
        # frame; re-running `detect_cuts` over the whole buffer every frame classified seven pairs
        # at full resolution each time and cost 268 ms a frame - a quarter of the budget, for a
        # result that had not changed since the previous frame.
        changes: list[SceneChange] = []
        # The previous frame's luma as the decoder gave it. The scene-cut histogram wants integers
        # and the detector wants an upload; both were being handed the float64 copy and spending
        # more time converting it back than doing their job.
        previous_plane: list[np.ndarray] = []
        anchor_history: dict[int, list[AnchorObservation]] = {}
        started = time.time()

        def transform(frame: Any, index: int) -> Any:
            if context.cancelled:
                return None

            context.frames_seen = index + 1

            # **Progress is reported here, before anything can return early.**
            #
            # It used to be emitted only on the path that actually restored a frame, which meant a
            # video with nothing to restore - or a stretch of one between regions - reported 0% for
            # its whole run while the decoder worked through it. Every early return below (cancelled,
            # detector failure, no restorable track, analysis) skipped it, and those are the common
            # cases, not the rare ones.
            #
            # The stage is fixed by the mode rather than by what this frame turned out to contain:
            # section 8.4 forbids a stage moving backwards, and choosing it per-frame would do
            # exactly that on the first frame with no region in it.
            #
            # Offered every frame, not every eighth. The emitter's own rate limit (section 8.4, four a
            # second) is what stops a fast job flooding the channel, and it does that job whatever
            # it is offered - whereas a fixed stride is a *floor* on the interval, so on hardware
            # that manages a frame a second it turned into one update every eight seconds. Measured
            # here at 1080p on the CPU: 0%, then 6.7% ten seconds later, then 13.3% twenty after
            # that. That reads as a hung window, which is what it was mistaken for.
            elapsed = max(time.time() - started, 1e-6)
            rate = (index + 1) / elapsed

            # `eta` has been in the protocol from the start and nothing has ever filled it in. A
            # percentage alone cannot answer the question a user actually asks ten minutes in -
            # "is this stuck?" - because at this pipeline's throughput an hour of video sits below
            # 0.5% for the first ten minutes and rounds to zero. Measured: 0.45 frames a second at
            # 1080p, so a one-hour source is a sixty-seven-hour job.
            #
            # Null when the container did not say how long the source is: that is also the case
            # where `fraction` is stuck at zero, and the host tells the two apart by seeing a rate
            # with no estimate.
            remaining = (total - index) / rate if total and rate > 0 else None

            emitter.progress(
                context.job_id,
                Stage.ANALYZING if context.analyze_only else Stage.RESTORING,
                min((index / total) if total else 0.0, 0.99),
                pts=int(frame.pts) if frame.pts is not None else None,
                fps=round(rate, 2),
                eta_seconds=round(remaining, 1) if remaining is not None else None,
            )

            # **The frame is handled as YUV planes, not RGB.**
            #
            # Converting every restored frame out to rgb24 and back cost 45.3 dB of luma on its own,
            # with no processing at all: the round trip destroys and re-creates 4:2:0 chroma and
            # rounds twice through the colour matrix. Measured against 46.5 dB for the encoder alone,
            # it was the single largest source of damage in the first end-to-end run — and it
            # applied to the whole frame, including everything the pipeline never touched.
            #
            # Working on the luma plane and leaving chroma alone round-trips losslessly.
            planes = frame.to_ndarray(format="yuv420p")
            plane_height = frame.height
            plane = planes[:plane_height]
            luma = plane.astype(np.float64)

            history.append(luma)
            if previous_plane:
                changes.append(classify_pair(previous_plane[0], plane, len(history) - 1))
            previous_plane[:] = [plane]
            # The slot is reserved now and filled once detection has run. Appending the mask where
            # it is computed would desynchronise the two lists on every early return - a detector
            # failure, or a frame with no region - and a neighbour would then be paired with some
            # other frame's mask.
            mask_history.append(np.zeros_like(luma, dtype=bool))
            if len(history) > 2 * radius + 1:
                history.pop(0)
                mask_history.pop(0)
                # The boundary before the frame that just left goes with it; the rest slide down.
                changes[:] = [replace(c, index=c.index - 1) for c in changes if c.index > 1]

            if index % detect_every == 0:
                try:
                    probability = self._segment(plane, settings)
                except WorkerError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one frame failing is not the job failing
                    emitter.warn(E3002, f"detector failed on frame {index}: {exc}")
                    context.frames_passed_through += 1
                    context.note_route("DetectorFailed")
                    return None

                regions, clamped = extract_regions(
                    probability, threshold=mask_threshold, min_area=min_region_area
                )
                if clamped:
                    emitter.warn(W3101, "region count clamped", frame=index)

                for found in regions:
                    mask_history[-1] |= found.mask

                context.regions_detected += len(regions)
                tracks = tracker.update(regions)
            else:
                # Nobody looked at this frame. The tracker moves each region where its motion
                # model predicts, and the frame's mask is the union of what it is carrying.
                tracks = tracker.coast()
                for carried in tracks:
                    if carried.region is not None:
                        mask_history[-1] |= carried.region.mask

            restorable = [t for t in tracks if t.is_restorable and t.region is not None]

            if not restorable:
                context.frames_passed_through += 1
                context.note_route("NoRegion")
                return None

            # The protocol defines `analyze` as detection and tracking only. Everything below is
            # restoration, and running it to throw the pixels away made the preview cost more than
            # the job it previews (162 s against 153 s, measured).
            if context.analyze_only:
                context.frames_with_regions += 1
                context.note_route("AnalyzedOnly")
                return None

            target_index = len(history) - 1
            scene_start, scene_end = same_scene_span(
                changes, target_index, radius, total_frames=len(history)
            )
            same_scene = scene_end - scene_start + 1

            output = luma.copy()
            touched = False

            if backend == "evidence":
                outcomes = [
                    (track, self._restore_track(
                        track=track,
                        luma=luma,
                        history=history,
                        mask_history=mask_history,
                        frame_index=index,
                        target_index=target_index,
                        anchor_history=anchor_history,
                        same_scene=same_scene,
                        preset=preset,
                        window_setting=window_setting,
                        min_region_area=min_region_area,
                        align_conf_min=align_conf_min,
                        gate=gate,
                        smoother=smoother,
                        accumulator=accumulator,
                        emitter=emitter,
                        context=context,
                    ))
                    for track in restorable
                ]
            else:
                outcomes = self._restore_single_frame(
                    restorable,
                    luma=luma,
                    planes=planes,
                    plane_height=plane_height,
                    refiner=refiner,
                    refine_stats=refine_stats,
                    backend=backend,
                    temporal=temporal,
                    frame_index=index,
                    history_length=len(history),
                    anchor_history=anchor_history,
                    same_scene=same_scene,
                    preset=preset,
                    window_setting=window_setting,
                    min_region_area=min_region_area,
                    align_conf_min=align_conf_min,
                    gate=gate,
                    smoother=smoother,
                    emitter=emitter,
                    context=context,
                )

            for track, outcome in outcomes:
                context.note_route(outcome.decision.reason.value)

                if outcome.decision.path is RestorationPath.PASS_THROUGH or outcome.restored is None:
                    continue

                context.confidences.append(outcome.confidence)

                # Everything below happens inside the ROI, on luma only. Chroma is left exactly as
                # decoded: a mosaic destroys luma detail, and rewriting chroma to chase it would
                # damage colour the pipeline has no evidence about.
                roi = outcome.roi
                left, top, right, bottom = roi.bounds
                crop_mask = track.region.mask[top:bottom, left:right]

                output[top:bottom, left:right] = blend_region(
                    output[top:bottom, left:right],
                    outcome.restored,
                    crop_mask,
                    block_size=outcome.profile.block_size,
                    feather_px=feather_px,
                    temporal=temporal_alpha,
                    track_id=track.track_id,
                )
                touched = True

                if region_log is not None:
                    left_, top_, right_, bottom_ = roi.bounds
                    region_log.write(json.dumps({
                        "frame": index,
                        "track": track.track_id,
                        "box": [left_, top_, right_, bottom_],
                        "area": int(track.region.area),
                        "confidence": round(outcome.confidence, 4),
                        "gridConfidence": round(outcome.profile.confidence, 4),
                        "kind": str(getattr(outcome.profile.kind, "value", outcome.profile.kind)),
                        "blockSize": int(outcome.profile.block_size),
                        "anchor": str(getattr(outcome.profile.anchor, "value",
                                              outcome.profile.anchor)),
                        "anchorConfidence": round(outcome.profile.anchor_confidence, 4),
                        "meanAlignment": round(outcome.mean_alignment, 4),
                        # With an accumulator this is the depth of the evidence chain, not a count
                        # of simultaneously aligned neighbours. A chain that keeps restarting shows
                        # up here as a number that never grows.
                        "evidenceDepth": outcome.usable_neighbours,
                        "reason": outcome.decision.reason.value,
                    }) + "\n")

            if not touched:
                context.frames_passed_through += 1
                return None

            context.frames_restored += 1

            import av

            # Rebuild the plane stack with the restored luma and the *original* chroma. Rounding
            # rather than truncating: astype() truncates, which biases every pixel down by half a
            # level across the whole frame.
            planes = planes.copy()
            planes[:plane_height] = np.rint(np.clip(output, 0, 255)).astype(np.uint8)

            replacement = av.VideoFrame.from_ndarray(planes, format="yuv420p")
            replacement.time_base = frame.time_base
            return replacement

        if context.analyze_only:
            emitter.progress(context.job_id, Stage.ANALYZING, 0.0, force=True)
            analysis = run_analysis(
                source,
                transform=transform,
                sample_every=max(1, int(context.sample_every or 1)),
            )

            emitter.progress(context.job_id, Stage.FINALIZING, 1.0, force=True)
            summary = context.summary()
            # The transform only runs on sampled frames, so its own counter undercounts the file.
            # The decoder saw all of them.
            summary["framesSeen"] = analysis.frames_seen
            summary["framesExamined"] = analysis.frames_examined
            summary["regionsGated"] = gate.gated_track_count
            # No file was written, so there is no output timeline to compare. Saying "preserved"
            # here would be asserting something about bytes that do not exist.
            summary["timeline"] = f"analysis only, {analysis.frames_examined} frames examined"
            return summary

        emitter.progress(context.job_id, Stage.RESTORING, 0.0, force=True)

        # Encode to a sibling first. R-1.8a can only be decided once the whole file has been seen,
        # and staging keeps that decision cheap: a job that turns out to have restored nothing
        # discards this and stream-copies instead. It also means a crash mid-encode never leaves a
        # truncated file sitting at the destination looking finished.
        destination = Path(context.output_path)
        # The extension has to survive: PyAV picks the muxer from it, and a staging name of
        # "out.mp4.part" fails with "Could not determine output format".
        staging = destination.with_name(f"{destination.stem}.part{destination.suffix}")

        try:
            result = run_passthrough(
                source,
                staging,
                transform=transform,
                encoder=_encoder_for(settings),
                crf=int(_setting(settings, "encode", "constantQuality", default=18)),
                preset=_setting(settings, "encode", "preset", default="medium"),
            )

            if context.frames_restored == 0:
                # R-1.8a. A full re-encode at the transparent operating point still costs about
                # 2.9 dB across the whole frame (docs/untouched-decomposition.json), and a file
                # the pipeline never touched has nothing to show for paying it.
                emitter.progress(context.job_id, Stage.FINALIZING, 0.5, force=True)
                try:
                    result = run_stream_copy(source, destination)
                    context.passthrough = True
                    staging.unlink(missing_ok=True)
                except StreamCopyUnavailable as reason:
                    # Keep the re-encode rather than failing the job - but say so. The defect this
                    # replaced was a summary that reported passthrough while re-encoding; falling
                    # back silently would restore it exactly.
                    emitter.warn(
                        W5102,
                        f"restored nothing, but could not stream-copy: {reason}. "
                        "The output was re-encoded and is slightly softer than the source.",
                    )
                    os.replace(staging, destination)
            else:
                os.replace(staging, destination)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

        if region_log is not None:
            region_log.close()

        emitter.progress(context.job_id, Stage.FINALIZING, 1.0, force=True)

        timeline = result.timeline()
        summary = context.summary()
        summary["frameCountPreserved"] = timeline.frame_count_preserved
        summary["timeline"] = timeline.describe()
        summary["regionsGated"] = gate.gated_track_count
        summary["regionsRefined"] = refine_stats.regions
        if refine_stats.regions:
            emitter.log(
                "info",
                "diffusion refiner: %d regions, %.0f ms each" % (
                    refine_stats.regions, 1000 * refine_stats.seconds / refine_stats.regions),
                model=refine_settings.model, strength=refine_settings.strength,
            )

        if gate.gated_track_count:
            emitter.warn(
                W4102,
                f"{gate.gated_track_count} regions left untouched: evidence below threshold",
            )

        return summary

    def _restore_track(
        self,
        *,
        track: Track,
        luma: np.ndarray,
        history: list[np.ndarray],
        mask_history: list[np.ndarray],
        target_index: int,
        #: The frame's own number. `target_index` indexes the rolling history buffer, which
        #: stops advancing once the buffer is full - using it here restarted the evidence
        #: chain on every frame while every log line said the window was fine.
        frame_index: int,
        anchor_history: dict[int, list[AnchorObservation]],
        same_scene: int,
        preset: QualityPreset,
        window_setting: int | None,
        min_region_area: int,
        align_conf_min: float,
        gate: ConfidenceGate,
        smoother: ConfidenceSmoother,
        accumulator: EvidenceAccumulator,
        emitter: Emitter,
        context: JobContext,
    ) -> "TrackOutcome":
        """Restores one track's region on one frame, or explains why it did not.

        **Everything happens inside a padded ROI** (§5.5). Running dense flow and back-projection at
        full frame size costs more than ten times what the job needs and spends VRAM on pixels that
        are discarded — an earlier version did exactly that and a 96-frame clip did not finish.
        """
        located = self._locate_track(track, luma, anchor_history)
        if located is None:
            return TrackOutcome(route(RouteInputs(has_region=False)), None, 0.0, None, MosaicProfile())
        region, profile, roi = located
        left, top, right, bottom = region.box
        anchor = profile.anchor

        window = decide_window(
            setting=window_setting,
            preset=preset,
            motion_pixels_per_frame=track.speed,
            anchor=anchor,
            same_scene_frames=same_scene,
            stream_frames=len(history),
        )
        if window.was_reduced:
            emitter.warn(
                W4103,
                f"window reduced to {window.effective}",
                requested=window.requested,
                rule=window.reason.value,
                track=track.track_id,
            )

        crop_target = roi.crop(luma)
        target_mask = roi.crop(mask_history[target_index]).astype(bool)

        # **One alignment, not a window of them.** D-28.
        #
        # The batch form aligned the target to each of K neighbours and solved over all of them,
        # which costs K alignments and K times the solver work *per frame*. The corrected forward
        # model (D-26) needs K of about 17 before it pays, and that measured out at 0.23 fps against
        # section 6.1's 4 fps target.
        #
        # The accumulator carries one estimate per track forward instead: align to the immediately
        # preceding frame, warp what has been accumulated, fold in what this frame observed. One
        # alignment and one warp regardless of how far back the evidence goes -- 4.33 fps measured,
        # with an unbounded history rather than K.
        #
        # It is also better, for a reason already in docs/phase2-alignment-report.md section 3:
        # shorter baselines align better. This chains one-frame alignments where the batch form
        # reached across the whole window.
        previous_index = target_index - 1
        alignments: list[Any] = []

        if 0 <= previous_index < len(history):
            try:
                # The backward pass exists for `reconstruct_flow`, which warps residuals back
                # along it. The accumulator warps corrections nowhere, so it costs half the
                # alignment time for nothing: measured, +2.81 dB either way, 104 ms against
                # 49 (D-32).
                alignments.append(
                    self._align(crop_target, roi.crop(history[previous_index]), backward=False)
                )
            except Exception as exc:  # noqa: BLE001 - alignment failure degrades, never fails
                emitter.warn(E4002, f"alignment failed: {exc}", track=track.track_id)
                alignments.append(None)

        # The neighbour's own mask is not needed: its evidence is already inside the accumulated
        # estimate, and the operator has to describe how *this* frame observed the scene.

        usable = [a for a in alignments if a is not None and a.usable_fraction >= align_conf_min]
        scored = [a.usable_fraction for a in alignments if a is not None]
        mean_alignment = float(np.mean(scored)) if scored else 0.0

        # Confidence combines what §5.9.4 asks for: observations, alignment, and how much the block
        # size destroyed to begin with.
        block_penalty = float(np.clip(1.0 - (profile.block_size - 4) / 24.0, 0.0, 1.0))
        confidence = float(
            np.clip(0.25 + 0.35 * mean_alignment + 0.4 * block_penalty * (len(usable) > 0), 0.0, 1.0)
        )

        # The gate's parameter is `smoothed_confidence`, and it used to be handed the raw
        # per-frame value. The gate is per track and sticky in both directions, so a long track
        # would open on a run of good frames and coast through the bad ones - measured, that gap
        # was the difference between the gate reaching +0.05 dB and reaching 0.0 by withholding
        # everything (docs/gate-calibration.json).
        smoothed = smoother.update(track.track_id, confidence)
        withheld = gate.should_withhold(track.track_id, smoothed)
        if withheld:
            context.regions_gated += 1

        # The grid phase is measured in frame coordinates, so it has to be re-expressed relative
        # to the crop's own origin before the operator is applied to the crop.
        phase = profile.phase_for(left, top)
        crop_left, crop_top = roi.bounds[0], roi.bounds[1]
        phase = (
            (phase[0] + crop_left) % profile.block_width,
            (phase[1] + crop_top) % profile.block_height,
        )

        # **Fold first, route second.** The accumulator costs about 6 ms; withholding the update
        # until the router approves would mean a track that is gated for three frames arrives at
        # the fourth with no history, which is the deadlock the batch form did not have. Evidence
        # is gathered whenever it can be; the router decides whether the *result* is composited.
        try:
            estimate = accumulator.update(
                track.track_id,
                frame_index=frame_index,
                observation=crop_target,
                previous_observation=(
                    roi.crop(history[previous_index])
                    if 0 <= previous_index < len(history)
                    else None
                ),
                # crop_bounds, not bounds: the reflect padding is part of the crop, and comparing
                # the two rectangles is how an estimate survives the ROI moving with the region.
                bounds=roi.crop_bounds,
                mask=target_mask,
                spec=profile,
                phase=phase,
                flow_to_previous=usable[0].target_to_neighbour if usable else None,
                same_scene=same_scene > 1,
            )
        except Exception as exc:  # noqa: BLE001 - section 5.8.2: degrade, never emit a partial composite
            emitter.warn(
                E4002,
                f"accumulation failed: {exc}",
                track=track.track_id,
                band=band_for(profile.block_size),
            )
            accumulator.reset(track.track_id)
            return TrackOutcome(
                route(RouteInputs(degradation_chain_exhausted=True)), None, confidence, roi, profile
            )

        # With an accumulator, "how many neighbours are aligned right now" is the wrong question.
        # The evidence is the depth of the chain, and the router's minimum reads naturally against
        # it: two observations folded in is one neighbour's worth of evidence.
        evidence = max(accumulator.depth(track.track_id) - 1, 0)

        decision = route(
            RouteInputs(
                has_region=True,
                region_area=region.area,
                min_region_area=min_region_area,
                is_confirmed=track.is_restorable,
                withheld_by_confidence_gate=withheld,
                anchor=anchor,
                motion_pixels_per_frame=track.speed,
                window=window,
                valid_aligned_neighbours=evidence,
                mean_alignment_confidence=mean_alignment,
                align_conf_min=align_conf_min,
            )
        )

        if decision.path is RestorationPath.PASS_THROUGH:
            # A withheld frame still keeps its evidence: the region is unchanged, the chain is not
            # broken, and dropping it would restart accumulation every time the gate blinked.
            return TrackOutcome(
                decision, None, confidence, roi, profile, mean_alignment, evidence
            )

        restored = estimate

        # Back to the frame's coordinates, minus the alignment padding (§5.5.3).
        pad_left, pad_top, pad_right, pad_bottom = roi.reflect
        trimmed = restored[
            pad_top : restored.shape[0] - pad_bottom if pad_bottom else restored.shape[0],
            pad_left : restored.shape[1] - pad_right if pad_right else restored.shape[1],
        ]

        return TrackOutcome(
            decision, trimmed, confidence, roi, profile, mean_alignment, evidence
        )


    def _locate_track(
        self,
        track: Track,
        luma: np.ndarray,
        anchor_history: dict[int, list[AnchorObservation]],
    ) -> tuple[Region, MosaicProfile, Any] | None:
        """Where a track is on this frame and what its mosaic looks like: region, profile, ROI.

        Shared by every restoration path. The grid geometry, the anchoring estimate and the padded
        ROI are the same whether the pixels are then folded from neighbours or upscaled from this
        frame alone. ``None`` when the region has no pixels.
        """
        region: Region = track.region  # type: ignore[assignment]
        left, top, right, bottom = region.box

        patch = luma[top:bottom, left:right]
        if patch.size == 0:
            return None

        profile, _ = estimate_geometry(patch)

        observations = anchor_history.setdefault(track.track_id, [])
        observations.append(
            AnchorObservation(
                box_origin=(left, top),
                phase=(profile.grid_offset_x, profile.grid_offset_y),
            )
        )
        anchor, anchor_confidence = estimate_anchor(
            observations, (profile.block_width, profile.block_height)
        )
        profile = MosaicProfile(
            kind=profile.kind,
            block_width=profile.block_width,
            block_height=profile.block_height,
            grid_offset_x=profile.grid_offset_x,
            grid_offset_y=profile.grid_offset_y,
            anchor=anchor,
            anchor_confidence=anchor_confidence,
            confidence=profile.confidence,
        )

        roi = build_roi(region.box, luma.shape, block_size=profile.block_size)
        return region, profile, roi

    def _backend_for(self, preset: QualityPreset, emitter: Emitter) -> Any:
        """The restorer the preset asks for. See the table where this is called."""
        if preset is QualityPreset.QUALITY:
            return "evidence"
        if preset is QualityPreset.FAST:
            return "bicubic"

        try:
            return Upscaler(MODELS / "restorer" / RESTORER_ID)
        except WorkerError as error:
            if error.code is not E4001:
                raise
            # The network is optional; the floor is not. Saying so is the point of the warning -
            # the earlier defect here was a summary that claimed a backend it had not used.
            emitter.warn(
                W6101,
                f"restorer {RESTORER_ID} unavailable ({error}); using bicubic upscaling instead",
            )
            return "bicubic"

    def _restore_single_frame(
        self,
        tracks: list[Track],
        *,
        luma: np.ndarray,
        planes: np.ndarray,
        plane_height: int,
        refiner: DiffusionRefiner | None,
        refine_stats: RefineStats,
        backend: Any,
        temporal: TemporalSmoother,
        frame_index: int,
        history_length: int,
        anchor_history: dict[int, list[AnchorObservation]],
        same_scene: int,
        preset: QualityPreset,
        window_setting: int | None,
        min_region_area: int,
        align_conf_min: float,
        gate: ConfidenceGate,
        smoother: ConfidenceSmoother,
        emitter: Emitter,
        context: JobContext,
    ) -> list[tuple[Track, TrackOutcome]]:
        """Restores every track on this frame from this frame alone. D-43.

        Decimates each region to its block resolution, upscales - all regions in one batch when a
        network is in play - and blends each result with the track's previous one. No neighbouring
        frame is consulted, so the confidence reported is the block-size prior alone: there is no
        alignment evidence to add, and claiming any would misdescribe what happened (§7.4).
        """
        located: list[tuple[Track, Region, MosaicProfile, Any]] = []
        outcomes: list[tuple[Track, TrackOutcome]] = []

        for track in tracks:
            where = self._locate_track(track, luma, anchor_history)
            if where is None:
                outcomes.append((track, TrackOutcome(
                    route(RouteInputs(has_region=False)), None, 0.0, None, MosaicProfile()
                )))
                continue
            located.append((track, *where))

        # Everything the router needs, per track, before any pixel is touched.
        to_restore: list[tuple[Track, Region, MosaicProfile, Any, Any, float]] = []
        for track, region, profile, roi in located:
            window = decide_window(
                setting=window_setting,
                preset=preset,
                motion_pixels_per_frame=track.speed,
                anchor=profile.anchor,
                same_scene_frames=same_scene,
                stream_frames=history_length,
            )
            block_penalty = float(np.clip(1.0 - (profile.block_size - 4) / 24.0, 0.0, 1.0))
            confidence = float(np.clip(0.25 + 0.4 * block_penalty, 0.0, 1.0))
            smoothed = smoother.update(track.track_id, confidence)
            withheld = gate.should_withhold(track.track_id, smoothed)
            if withheld:
                context.regions_gated += 1

            decision = route(RouteInputs(
                has_region=True,
                region_area=region.area,
                min_region_area=min_region_area,
                is_confirmed=track.is_restorable,
                withheld_by_confidence_gate=withheld,
                anchor=profile.anchor,
                motion_pixels_per_frame=track.speed,
                window=window,
                valid_aligned_neighbours=0,
                mean_alignment_confidence=0.0,
                align_conf_min=align_conf_min,
            ))
            if decision.path is RestorationPath.PASS_THROUGH:
                outcomes.append((track, TrackOutcome(decision, None, confidence, roi, profile)))
                continue
            to_restore.append((track, region, profile, roi, decision, confidence))

        if not to_restore:
            return outcomes

        crops = [roi.crop(luma) for _, _, _, roi, _, _ in to_restore]
        phases = []
        for _, region, profile, roi, _, _ in to_restore:
            # Grid phase is measured in frame coordinates; re-express it relative to the crop.
            phase = profile.phase_for(region.box[0], region.box[1])
            phases.append((
                (phase[0] + roi.bounds[0]) % profile.block_width,
                (phase[1] + roi.bounds[1]) % profile.block_height,
            ))
        specs = [profile for _, _, profile, _, _, _ in to_restore]

        try:
            if backend == "bicubic":
                restored = [bicubic_restore(c, s, p) for c, s, p in zip(crops, specs, phases)]
            else:
                restored = backend.restore_many(crops, specs, phases)
        except Exception as exc:  # noqa: BLE001 - section 5.8.2: degrade, never emit a partial composite
            emitter.warn(E4002, f"upscaling failed on frame {frame_index}: {exc}")
            for track, _, profile, roi, _, confidence in to_restore:
                outcomes.append((track, TrackOutcome(
                    route(RouteInputs(degradation_chain_exhausted=True)), None, confidence, roi, profile
                )))
            return outcomes

        # The diffusion pass sits between the restorer and the temporal blend: it refines what
        # this frame produced, and the blend then damps what it invented differently from the
        # last frame. Chroma comes from the frame; only the luma change comes back.
        if refiner is not None and refiner.available:
            refined: list[np.ndarray] = []
            for (_, _, _, roi, _, _), pixels in zip(to_restore, restored):
                started_refine = time.time()
                try:
                    u, v = chroma_for(planes, plane_height, roi.crop_bounds)
                    refined.append(refiner.refine_luma(pixels, u, v))
                    refine_stats.regions += 1
                    refine_stats.seconds += time.time() - started_refine
                except WorkerError as error:
                    if not refine_stats.warned:
                        emitter.warn(W6101, f"diffusion refiner unavailable ({error}); regions left unrefined")
                        refine_stats.warned = True
                    refined.append(pixels)
                except Exception as exc:  # noqa: BLE001 - one region failing is not the job failing
                    emitter.warn(E4002, f"refiner failed on frame {frame_index}: {exc}")
                    refined.append(pixels)
            restored = refined

        for (track, _, profile, roi, decision, confidence), pixels, crop in zip(
            to_restore, restored, crops
        ):
            blended = temporal.smooth(
                track.track_id,
                pixels,
                observation=crop,
                bounds=roi.crop_bounds,
                frame_index=frame_index,
                same_scene=same_scene > 1,
            )
            # Back to the frame's coordinates, minus the alignment padding (section 5.5.3).
            pad_left, pad_top, pad_right, pad_bottom = roi.reflect
            trimmed = blended[
                pad_top : blended.shape[0] - pad_bottom if pad_bottom else blended.shape[0],
                pad_left : blended.shape[1] - pad_right if pad_right else blended.shape[1],
            ]
            outcomes.append((track, TrackOutcome(decision, trimmed, confidence, roi, profile)))

        return outcomes


def _encoder_for(settings: dict[str, Any]) -> str:
    """Maps the encoder profile to a codec name. prd.md §5.1.4, D-12.

    NVENC is deliberately absent: PyAV bundles its own FFmpeg without it, so the Speed profile has
    to shell out to `tools/ffmpeg` and that path is not wired up yet. Choosing it silently would
    give the user x265 while the settings said NVENC.
    """
    profile = _setting(settings, "encode", "profile", default="QualityX265")
    if profile == "SpeedNvenc":
        raise WorkerError(
            E4002,
            "the Speed encoder profile needs tools/ffmpeg and is not wired up yet; "
            "PyAV's bundled FFmpeg has no NVENC",
        )
    return "libx265" if _setting(settings, "encode", "codec", default="H265") == "H265" else "libx264"
