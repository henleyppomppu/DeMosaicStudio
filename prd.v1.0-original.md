# Product Requirements Document (PRD)

## Project Name: DeMosaic Studio

**Code Name:** Dynamic Mosaic Auto-Detection & Multi-Frame Restoration\
**Target Platform:** Windows 10/11 x64\
**Primary Acceleration:** NVIDIA CUDA / TensorRT / NVDEC / NVENC\
**Fallback Acceleration:** ONNX Runtime DirectML / CPU\
**Document Status:** Revised Draft\
**Version:** 1.0

------------------------------------------------------------------------

## 1. Executive Summary

### 1.1 Background

Traditional mosaic restoration tools generally require manual ROI
annotation or rely on unstable frame-by-frame detection. When the mosaic
position, shape, block size, or target object changes over time, these
approaches can introduce ROI jitter, boundary artifacts, temporal
flicker, and inconsistent reconstruction.

DeMosaic Studio is a standalone Windows desktop application that
automatically detects and tracks partially mosaicked regions in video,
analyzes the degradation characteristics of each mosaic region, collects
useful information from neighboring frames, performs temporally aligned
multi-frame restoration, and exports a visually consistent reconstructed
video.

The application is designed primarily for local NVIDIA GPU inference and
shall not require cloud processing.

### 1.2 Product Goal

The product shall provide:

-   Automatic mosaic detection and segmentation without manual
    annotation.
-   Stable multi-object tracking for dynamically moving or resizing
    mosaic regions.
-   Mosaic degradation analysis including block size, grid offset, and
    degradation type.
-   Adaptive multi-frame temporal reconstruction using neighboring
    frames.
-   Best-effort restoration when original information cannot be
    recovered.
-   Temporal consistency with minimal flicker and ROI boundary
    artifacts.
-   Hardware-accelerated decode, inference, processing, and encode on
    NVIDIA GPUs.
-   Preservation of audio, timestamps, subtitles, and relevant container
    metadata.
-   Reliable processing of long videos with cancellation, checkpoint,
    and resume support.

### 1.3 Restoration Semantics

DeMosaic Studio performs **best-effort visual restoration**, not
guaranteed recovery of the original hidden content.

If useful source information exists across neighboring frames, the
engine shall prioritize observable temporal information. If information
has been irreversibly destroyed in all available frames, a restoration
model may estimate visually plausible content. Such estimated content
must not be represented internally as verified recovery of the original
pixels.

The engine should optionally calculate a restoration confidence map
indicating how strongly a restored region is supported by observable
temporal information.

------------------------------------------------------------------------

## 2. Scope

### 2.1 In Scope

-   Partially mosaicked video restoration.
-   Multiple simultaneous mosaic regions.
-   Dynamically moving/resizing mosaic regions.
-   Pixelation, blur, and mixed degradation.
-   Multi-frame temporal reconstruction.
-   Single-frame fallback restoration.
-   Local Windows processing.
-   NVIDIA GPU acceleration.
-   DirectML/CPU fallback.
-   Hardware video decoding/encoding where supported.
-   Audio/subtitle preservation.
-   Long-running processing jobs.
-   Before/after preview and diagnostic overlays.

### 2.2 Out of Scope for Initial MVP

-   Cloud inference.
-   Distributed rendering.
-   Guaranteed forensic recovery of destroyed information.
-   Manual frame-by-frame painting tools.
-   Full professional NLE functionality.
-   Training models inside the desktop application.

------------------------------------------------------------------------

## 3. High-Level Architecture

``` text
+--------------------------------------------------------------------------------+
|                         Presentation Layer                                     |
|                                                                                |
|  C# .NET 8 WPF                                                                |
|  - Drag & Drop                                                                |
|  - Video Preview / Before-After                                                |
|  - Mosaic / Track Overlay                                                      |
|  - Progress / ETA                                                              |
|  - Hardware / Quality Settings                                                 |
|  - Job / Resume Management                                                     |
+---------------------------------------+----------------------------------------+
                                        |
                                C API / PInvoke
                                        |
+---------------------------------------v----------------------------------------+
|                         Native Core Engine - C++20                              |
|                                                                                |
|  Video Decode                                                                  |
|      |                                                                         |
|      v                                                                         |
|  Scene Cut Detection                                                           |
|      |                                                                         |
|      v                                                                         |
|  Mosaic Segmentation                                                           |
|      |                                                                         |
|      v                                                                         |
|  Multi-Object Tracking + Temporal Smoothing                                    |
|      |                                                                         |
|      v                                                                         |
|  Mosaic Degradation Analysis                                                   |
|      |                                                                         |
|      v                                                                         |
|  Temporal ROI Stabilization                                                    |
|      |                                                                         |
|      v                                                                         |
|  Adaptive Frame Window                                                         |
|      |                                                                         |
|      v                                                                         |
|  Restoration Strategy Router                                                   |
|      |                                                                         |
|      +--> Multi-Frame Reconstruction / VSR                                     |
|      +--> Single-Frame Restoration                                             |
|      +--> Pass-through                                                         |
|      |                                                                         |
|      v                                                                         |
|  Temporal Consistency / Artifact Suppression                                   |
|      |                                                                         |
|      v                                                                         |
|  Mask-Aware Blending                                                           |
|      |                                                                         |
|      v                                                                         |
|  Video Encode / Audio & Subtitle Mux                                            |
+--------------------------------------------------------------------------------+
```

------------------------------------------------------------------------

## 4. Recommended Technology Stack

  Area                     Primary Technology
  ------------------------ -------------------------------------------
  Desktop UI               C# / .NET 8 / WPF
  Native Engine            C++20
  Training / Experiments   Python / PyTorch
  Media                    FFmpeg
  NVIDIA Decode            NVDEC
  NVIDIA Encode            NVENC
  Detection                Segmentation model, YOLO-family candidate
  Tracking                 ByteTrack + Kalman Filter
  NVIDIA Inference         TensorRT
  Generic Inference        ONNX Runtime
  Non-NVIDIA GPU           DirectML
  GPU Processing           CUDA
  UI/Core Integration      Stable C ABI + P/Invoke
  Installer                Inno Setup or WiX

Python/PyQt may be used for model-development prototypes, but the
production desktop application should use C# UI plus a native C++
engine.

------------------------------------------------------------------------

## 5. Functional Requirements

## 5.1 FR-1: Media I/O

### FR-1.1 Container Support

The application shall support at minimum:

-   MP4
-   MKV
-   AVI
-   MOV

### FR-1.2 Video Codec Support

Input support shall include:

-   H.264 / AVC
-   H.265 / HEVC
-   AV1 where supported by the installed hardware/software stack.

### FR-1.3 Decode Backend

Decode priority:

1.  NVDEC
2.  D3D11VA or equivalent hardware path
3.  FFmpeg software decoding

The engine should minimize GPU-to-CPU copies.

### FR-1.4 Encode Backend

Encode priority:

1.  NVENC
2.  Software encoder fallback

User-selectable output codec should include H.264 and H.265 at minimum.

### FR-1.5 Audio Preservation

The engine shall:

-   Preserve the original audio stream when possible.
-   Prefer remuxing when video duration/timing remains unchanged.
-   Preserve multiple audio tracks when the output container supports
    them.
-   Provide an option to avoid audio transcoding.

### FR-1.6 Subtitle and Metadata Preservation

Where compatible with the destination container, the engine should
preserve:

-   Subtitle streams
-   Chapters
-   Rotation/orientation metadata
-   Relevant container metadata

### FR-1.7 Timestamp Handling

Processing shall be PTS-based rather than relying exclusively on frame
index.

The engine shall correctly support:

-   CFR
-   VFR
-   Audio/video synchronization

------------------------------------------------------------------------

## 5.2 FR-2: Mosaic Detection and Segmentation

### FR-2.1 Mosaic Segmentation

The primary detector shall output:

-   Pixel-level mosaic mask
-   Bounding box
-   Detection confidence
-   Optional degradation class

Segmentation masks shall be the primary representation for restoration
boundaries. Bounding boxes shall primarily be used for tracking,
cropping, scheduling, and memory management.

### FR-2.2 Detection Threshold

Default detection confidence:

`0.45`

User configurable:

`0.10 - 0.90`

### FR-2.3 NMS

Default NMS IoU threshold:

`0.50`

The threshold shall be configurable internally and may be exposed in
advanced settings.

### FR-2.4 Multi-Object Detection

The engine shall support multiple simultaneous mosaic regions within one
frame.

------------------------------------------------------------------------

## 5.3 FR-3: Tracking and Temporal Smoothing

### FR-3.1 Multi-Object Tracking

ByteTrack or an equivalent tracker shall maintain persistent Track IDs.

### FR-3.2 Kalman State

Recommended state:

`x = [cx, cy, w, h, vx, vy, vw, vh]^T`

### FR-3.3 Track States

Each track shall support:

-   TENTATIVE
-   ACTIVE
-   OCCLUDED
-   LOST
-   REACQUIRED
-   TERMINATED

### FR-3.4 Missing Detection Handling

Default maximum missing frames:

`3`

Short detector dropouts shall use motion prediction and mask propagation
instead of immediately terminating restoration.

### FR-3.5 Temporal Smoothing

Bounding-box and ROI motion may use:

-   Kalman filtering
-   EMA
-   Motion-aware smoothing

Smoothing must not introduce excessive lag for fast-moving regions.

------------------------------------------------------------------------

## 5.4 FR-4: Mosaic Degradation Analysis

For each active track, the engine shall estimate a `MosaicProfile`.

``` text
MosaicProfile
{
    type
    block_width
    block_height
    grid_offset_x
    grid_offset_y
    degradation_strength
    temporal_stability
    confidence
}
```

### FR-4.1 Supported Degradation Types

At minimum:

-   Pixelation
-   Gaussian blur
-   Box blur
-   Mixed
-   Unknown

### FR-4.2 Block Geometry

For pixelated mosaics, the engine should estimate:

-   Block width
-   Block height
-   Grid phase/offset

### FR-4.3 Temporal Profile Stabilization

Mosaic parameters should be stabilized across a Track ID to prevent
per-frame parameter oscillation.

------------------------------------------------------------------------

## 5.5 FR-5: Temporal ROI Stabilization

### FR-5.1 Adaptive ROI Padding

Padding shall not be hard-coded to exactly 15%.

Recommended formulation:

``` text
padding = max(
    minimum_padding,
    bbox_size * padding_ratio,
    estimated_mosaic_block * 2
)
```

Default padding ratio:

`15%`

Recommended configurable range:

`10% - 20%`

### FR-5.2 Boundary Handling

When the padded ROI crosses frame boundaries, the engine shall clamp
coordinates and apply suitable tensor padding.

Reflection/replication padding should be preferred where appropriate
over unconditional zero-padding.

### FR-5.3 Tensor Alignment

ROI dimensions shall be aligned to model/backend requirements where
required, e.g. multiples of 8/16/32.

------------------------------------------------------------------------

## 5.6 FR-6: Adaptive Temporal Window

The temporal window shall support:

`K = 3, 5, 7, 9`

Default:

`K = 5`

Suggested policy:

-   Low motion: 7-9 frames
-   Medium motion: 5 frames
-   High motion: 3 frames
-   Scene boundary: truncate/reset temporal context

The scheduler may dynamically select K based on motion, VRAM budget, ROI
size, and quality mode.

------------------------------------------------------------------------

## 5.7 FR-7: Temporal Alignment

The restoration engine shall support sub-pixel temporal alignment.

Permitted implementations include:

-   Optical flow
-   Deformable alignment
-   Feature-space alignment
-   Learned implicit alignment

The PRD does not require DCNv2 specifically.

Any selected implementation must either:

1.  Be natively supported by the deployment backend, or
2.  Provide a tested optimized TensorRT plugin/fallback.

------------------------------------------------------------------------

## 5.8 FR-8: Restoration Strategy Router

The engine shall select an appropriate restoration path per track/frame.

Available paths:

### A. Multi-Frame Restoration

Primary path when sufficient neighboring information exists.

### B. Single-Frame Restoration

Used when:

-   Scene cut prevents temporal aggregation.
-   Temporal alignment confidence is poor.
-   Only one valid frame is available.
-   Motion/occlusion invalidates neighboring frames.

### C. Pass-Through

Used when no restoration is required.

The router shall consider:

-   Mosaic profile
-   Temporal window validity
-   Motion
-   Alignment confidence
-   ROI size
-   GPU memory budget
-   Selected quality mode

------------------------------------------------------------------------

## 5.9 FR-9: Multi-Frame Restoration

The engine shall support model abstraction rather than hard-coding
BasicVSR++.

Example interface:

``` text
IRestorationBackend
    FastRestorationBackend
    BalancedRestorationBackend
    QualityRestorationBackend
```

BasicVSR++ may be evaluated as one implementation candidate.

The architecture must allow replacement with custom degradation-aware
temporal models.

### FR-9.1 Temporal Feature Fusion

The selected model should aggregate useful information from neighboring
frames and minimize temporal instability.

### FR-9.2 Degradation Conditioning

Where supported, `MosaicProfile` parameters should be supplied to the
restoration model or preprocessing pipeline.

### FR-9.3 Restoration Confidence

The engine should calculate a confidence score/map based on factors such
as:

-   Valid temporal observations
-   Alignment quality
-   Occlusion
-   Mosaic severity
-   Model uncertainty

------------------------------------------------------------------------

## 5.10 FR-10: Temporal Consistency

The engine shall explicitly address flicker after reconstruction.

Possible mechanisms include:

-   Flow-guided consistency
-   Recurrent temporal features
-   Previous-output guidance
-   Temporal loss during training
-   Post-restoration temporal stabilization

A restored ROI shall not be accepted solely based on single-frame
perceptual quality if it creates severe frame-to-frame instability.

------------------------------------------------------------------------

## 5.11 FR-11: Mask-Aware Blending

Reconstructed content shall be blended using the segmentation mask
rather than the bounding box alone.

Recommended sequence:

``` text
segmentation mask
    -> controlled dilation
    -> edge-aware feathering
    -> temporal alpha smoothing
    -> compositing
```

Gaussian feathering may be used as one component but shall not be the
only required blending strategy.

The implementation shall minimize:

-   Halo artifacts
-   Visible rectangular ROI borders
-   Ghost edges
-   Temporal alpha flicker

------------------------------------------------------------------------

## 5.12 FR-12: Scene Cut Detection

The engine shall detect scene transitions and invalidate incompatible
temporal context.

Candidate signals:

-   Histogram divergence
-   Perceptual frame difference
-   Optical-flow discontinuity

On a scene cut:

1.  Reset incompatible temporal buffers.
2.  Terminate/reinitialize affected tracks.
3.  Use reduced-window or single-frame restoration until sufficient
    temporal context is rebuilt.

------------------------------------------------------------------------

## 5.13 FR-13: Pipeline Scheduler

The engine shall use an asynchronous bounded processing pipeline.

Conceptual execution:

``` text
Decode Thread
    |
Pinned / GPU Frame Queue
    |
Detection Stream
    |
Tracking / Degradation Analysis
    |
ROI Preprocess Stream
    |
Restoration Stream
    |
Postprocess / Blend Stream
    |
Encode Thread
```

Required behavior:

-   Bounded producer/consumer queues
-   Back-pressure
-   Cancellation
-   Pause/resume
-   Deterministic resource cleanup
-   Reusable tensor/buffer pools
-   CUDA pinned memory where beneficial
-   CUDA streams where beneficial
-   Minimal CPU/GPU copies

------------------------------------------------------------------------

## 5.14 FR-14: GPU Memory Management

The engine shall support dynamic VRAM budgeting.

User modes:

-   Auto
-   4 GB
-   6 GB
-   8 GB
-   12 GB+

Auto mode should use a safe percentage of available VRAM rather than
allocating all available memory.

OOM mitigation order may include:

1.  Reduce restoration batch size.
2.  Reduce temporal window.
3.  Enable/split tiles.
4.  Reduce tile dimensions.
5.  Switch to lower-memory restoration model.
6.  Use fallback backend if appropriate.
7.  Fail the affected job gracefully with actionable diagnostics.

The engine must not silently corrupt output after OOM recovery.

------------------------------------------------------------------------

## 5.15 FR-15: Job Checkpoint and Resume

Long-running jobs shall support checkpointing.

Checkpoint state should include:

``` text
JobState
{
    source_identifier/hash
    source_metadata
    model_versions
    application_version
    settings
    last_completed_pts
    output_temp_path
}
```

Resume shall verify that source file, relevant settings, and model
versions remain compatible.

------------------------------------------------------------------------

## 5.16 FR-16: User Interface

### FR-16.1 Drag and Drop

Users shall be able to drag supported video files into the application.

### FR-16.2 Metadata

Display at minimum:

-   Resolution
-   Duration
-   Nominal FPS
-   CFR/VFR indication
-   Video codec
-   Audio codec
-   File size

### FR-16.3 Preview

Provide:

-   Original preview
-   Restored preview
-   Split-view comparison

### FR-16.4 Diagnostic Overlay

Optional overlays:

-   Mosaic mask
-   Bounding box
-   Track ID
-   Detection confidence
-   Mosaic block estimate
-   Restoration confidence
-   Processing FPS

### FR-16.5 Quality Presets

Expose simple user presets:

-   Fast
-   Balanced
-   Quality

Advanced model/backend details may remain in an advanced settings panel.

### FR-16.6 Hardware Selection

Available options where supported:

-   NVIDIA TensorRT/CUDA
-   DirectML
-   CPU
-   Auto

### FR-16.7 Detection Sensitivity

Range:

`0.10 - 0.90`

### FR-16.8 Job Control

Provide:

-   Start
-   Pause
-   Resume
-   Cancel
-   Open output directory
-   Retry after recoverable failure

------------------------------------------------------------------------

## 6. Non-Functional Requirements

## 6.1 Performance

Reference benchmark configuration:

``` text
Input:
1920x1080
30 FPS
H.264

Reference GPU:
NVIDIA RTX 4070 12 GB

Typical Mosaic Coverage:
<= 15% of frame area

Mode:
Balanced
```

Initial targets:

  Component                                                        Target
  ----------------------------- -----------------------------------------
  Mosaic detector                                              \>= 60 FPS
  Typical restored processing                                  \>= 15 FPS
  Pass-through path               \>= 100 FPS where codec/backend permits
  UI responsiveness                         No blocking during processing

Performance results must report ROI coverage because restoration
throughput is strongly dependent on processed area.

### 6.1.1 Pipeline Startup

Startup/buffering latency shall be measured separately from steady-state
processing throughput.

The original `<200 ms` five-frame requirement shall not be treated as a
universal guarantee because decoder buffering, B-frame reordering, model
initialization, and temporal look-ahead vary by source and backend.

------------------------------------------------------------------------

## 6.2 Stability

The application shall support processing jobs longer than one hour
without unbounded memory growth.

Acceptance criteria should use measurable memory-growth limits rather
than claiming literal `0% memory leak`.

Recommended test:

-   Process representative 2+ hour input.
-   After warm-up, CPU and GPU memory usage must remain bounded within
    an agreed tolerance.
-   No resource accumulation proportional to processed frame count.

------------------------------------------------------------------------

## 6.3 Compatibility

Minimum OS:

-   Windows 10 x64 Build 19041+
-   Windows 11 x64

Primary NVIDIA environment:

-   NVIDIA GPUs supported by the packaged/runtime CUDA and TensorRT
    versions.

Fallback:

-   DirectX 12 compatible GPU via DirectML where model operations are
    supported.
-   CPU execution for compatibility/debugging.

Exact CUDA/TensorRT versions shall be pinned per application release
rather than permanently requiring an unspecified CUDA 12.x combination.

------------------------------------------------------------------------

## 6.4 Reliability

A failure in one processing job shall not require restarting the
application.

Errors shall be categorized at minimum as:

-   Unsupported media
-   Decoder failure
-   Detector failure
-   Restoration failure
-   GPU OOM
-   Encoder failure
-   Disk full
-   Output permission failure
-   Corrupted source
-   Unexpected internal error

Logs shall contain sufficient diagnostic context without exposing
unnecessary user media content.

------------------------------------------------------------------------

## 7. Algorithmic Specifications

## 7.1 Bounding Box Kalman Filter

Recommended state:

``` text
x_k = [cx, cy, w, h, vx, vy, vw, vh]^T
```

Prediction:

``` text
x_k^- = A x_(k-1) + B u_k
```

Kalman gain:

``` text
K_k = P_k^- H^T (H P_k^- H^T + R)^-1
```

Update:

``` text
x_k = x_k^- + K_k(z_k - Hx_k^-)
```

Noise parameters shall be tunable based on motion characteristics.

------------------------------------------------------------------------

## 7.2 Mosaic Profile

Mosaic parameter estimates shall be aggregated temporally per Track ID.

Abrupt parameter changes shall require sufficient evidence before
replacing a stable track profile.

------------------------------------------------------------------------

## 7.3 Alignment Confidence

Temporal frames with poor alignment or heavy occlusion shall be excluded
or down-weighted rather than blindly fused.

------------------------------------------------------------------------

## 7.4 Restoration Confidence

Suggested qualitative interpretation:

-   **High:** substantial observable information from neighboring
    frames.
-   **Medium:** temporal inference contributes significantly.
-   **Low:** result depends heavily on model estimation.

Confidence is a diagnostic measure and must not be interpreted as proof
that reconstructed content matches the original hidden content.

------------------------------------------------------------------------

## 8. Dataset Requirements

## 8.1 Synthetic Degradation Generator

The training/evaluation generator shall randomize:

-   Mosaic block width/height
-   Non-square blocks
-   Grid phase/offset
-   Pixelation
-   Gaussian blur
-   Box blur
-   Mixed pixelation/blur
-   Mosaic opacity where applicable
-   Partial mosaic
-   Dynamic mosaic size
-   Dynamic mosaic position
-   Multiple simultaneous ROIs
-   Camera motion
-   Object motion
-   Motion blur
-   Resize/downscale/upscale
-   Noise
-   JPEG compression
-   H.264 compression
-   H.265 compression
-   Chroma subsampling
-   Bitrate variation

Codec recompression is mandatory in evaluation datasets because
real-world mosaic boundaries often differ significantly from clean
synthetic samples.

## 8.2 Ground Truth

Synthetic training/evaluation shall preserve the original unmodified
video as ground truth.

Pipeline:

``` text
Ground Truth
    -> Synthetic Degradation
    -> Encode/Recompress
    -> Restoration
    -> Compare against Ground Truth
```

## 8.3 Dataset Splits

Train/validation/test splits must prevent near-duplicate frames or clips
from the same source from leaking across splits.

------------------------------------------------------------------------

## 9. Evaluation Metrics

### 9.1 Detection

Measure:

-   Precision
-   Recall
-   mAP50
-   mAP50-95
-   Mask IoU

Target mAP50 may initially be `>= 0.92` on the agreed representative
test dataset, but acceptance must also include recall and mask quality.

### 9.2 Tracking

Measure:

-   IDF1
-   ID switches
-   Track fragmentation
-   Lost/reacquired success rate

### 9.3 Restoration

Measure:

-   PSNR
-   SSIM
-   LPIPS

### 9.4 Temporal Quality

Measure at least one temporal metric such as:

-   Warping error
-   Temporal LPIPS
-   Flicker metric
-   Flow-consistency error

### 9.5 Boundary Quality

Evaluate:

-   Halo artifacts
-   Seam visibility
-   Mask-edge consistency
-   Temporal boundary flicker

### 9.6 Performance

Record:

-   Decode FPS
-   Detection FPS
-   Restoration FPS
-   Encode FPS
-   End-to-end FPS
-   GPU utilization
-   Peak VRAM
-   CPU utilization
-   Peak system RAM
-   ROI coverage percentage

------------------------------------------------------------------------

## 10. Model and Runtime Architecture

Models shall be versioned independently from the desktop application.

Suggested layout:

``` text
models/
    detector/
        model.onnx
        metadata.json

    restoration/
        fast/
        balanced/
        quality/
```

Each model package should declare:

-   Model version
-   Input requirements
-   Supported temporal windows
-   Tensor layout
-   Precision
-   Supported runtimes
-   Required preprocessing
-   Output semantics

TensorRT engines should not be assumed universally portable between all
GPU/runtime combinations. Where necessary, the application shall package
compatible engines or build/cache engines from an intermediate
representation.

------------------------------------------------------------------------

## 11. Precision and Optimization

Preferred inference precision:

1.  FP16
2.  FP32 fallback
3.  INT8 only after quality validation

Optimization techniques may include:

-   Tensor reuse
-   Buffer pooling
-   CUDA Graphs where useful
-   CUDA streams
-   Pinned host memory
-   Dynamic tiling
-   Mixed precision
-   Asynchronous copy/compute overlap

Optimization must not introduce unacceptable restoration-quality
regression.

------------------------------------------------------------------------

## 12. Edge Cases

The system shall explicitly handle:

1.  Mosaic partially outside frame boundaries.
2.  Mosaic appearing/disappearing suddenly.
3.  Mosaic changing size.
4.  Mosaic block size changing.
5.  Multiple overlapping mosaic regions.
6.  Fast object motion.
7.  Camera cuts.
8.  Camera flashes.
9.  Severe motion blur.
10. Heavy compression artifacts.
11. Detector miss for a small number of frames.
12. Complete target occlusion.
13. ROI smaller than model minimum input.
14. ROI larger than VRAM budget.
15. VFR input.
16. Corrupt/missing frames.
17. Unsupported codec/profile.
18. GPU OOM.
19. Encoder failure.
20. Disk space exhaustion.
21. User cancellation.
22. Application restart followed by job resume.

------------------------------------------------------------------------

## 13. Milestones

  -------------------------------------------------------------------------------------
  Phase                         Estimate Deliverables              Acceptance
  ---------------- --------------------- ------------------------- --------------------
  Phase 0 -                    1-2 weeks Decode/inference/encode   End-to-end
  Technical Spike                        PoC, TensorRT             feasibility
                                         compatibility study,      demonstrated
                                         representative dataset    

  Phase 1 -                    2-3 weeks Synthetic degradation     Detection/tracking
  Dataset &                              generator, segmentation   targets evaluated
  Detection                              model, tracking           

  Phase 2 -                    2-4 weeks Temporal alignment,       Multi-frame benefit
  Restoration PoC                        baseline VSR/restoration, demonstrated against
                                         quality benchmark         single-frame
                                                                   baseline

  Phase 3 - Native             2-3 weeks C++ scheduler, NVDEC,     Stable end-to-end
  Pipeline                               TensorRT, NVENC, buffer   processing
                                         management                

  Phase 4 -                      2 weeks WPF UI, job control,      Usable Windows MVP
  Desktop MVP                            preview, presets          

  Phase 5 -                    2-3 weeks Dynamic tiling, adaptive  RTX 3060/4070
  Optimization                           window, VRAM manager,     performance
                                         temporal tuning           validated

  Phase 6 -                    1-2 weeks Installer, crash          Release candidate
  Packaging & QA                         recovery, long-run tests  
  -------------------------------------------------------------------------------------

Schedule assumes pretrained/baseline restoration technology is
available. Developing and training a production-quality custom
degradation-aware temporal model may require additional iterations.

------------------------------------------------------------------------

## 14. MVP Definition

The MVP is complete when the application can:

1.  Load a supported video.
2.  Automatically detect partially mosaicked regions.
3.  Track moving mosaic regions.
4.  Estimate basic mosaic degradation parameters.
5.  Build a valid temporal frame window.
6.  Restore detected regions using at least one temporal model.
7.  Fall back to single-frame restoration when required.
8.  Blend restored regions without obvious rectangular seams.
9.  Preserve audio and synchronization.
10. Export H.264/H.265 video.
11. Use NVIDIA GPU acceleration.
12. Process representative long videos without unbounded memory growth.
13. Recover gracefully from cancellation and GPU OOM.
14. Display before/after preview.
15. Report meaningful performance and diagnostic information.

------------------------------------------------------------------------

## 15. Quality Modes

### Fast

Priorities:

-   Throughput
-   Low VRAM
-   Short temporal window
-   Lightweight restoration

### Balanced

Priorities:

-   Default user experience
-   Moderate temporal window
-   Good temporal consistency
-   Reasonable VRAM use

### Quality

Priorities:

-   Larger temporal context where useful
-   Higher-resolution ROI processing
-   More expensive restoration
-   Stronger temporal consistency

Quality mode shall not simply upscale the ROI more aggressively; it
should control the entire restoration strategy.

------------------------------------------------------------------------

## 16. Acceptance Test Matrix

At minimum test:

### Hardware

-   RTX 3060 class
-   RTX 4070 class
-   Higher-memory RTX GPU
-   DirectML-compatible non-NVIDIA system
-   CPU fallback

### Resolution

-   720p
-   1080p
-   1440p
-   4K

### Mosaic Coverage

-   \<5%

-   5-15%

-   15-30%

-   30%

### Motion

-   Static
-   Slow
-   Medium
-   Fast

### Duration

-   Short clip
-   10 minutes
-   1 hour
-   2+ hours

### Media

-   H.264 CFR
-   H.264 VFR
-   H.265
-   AV1 input where supported
-   Multiple audio tracks
-   Subtitle-containing container

------------------------------------------------------------------------

## 17. Key Engineering Principles

1.  **Segmentation first:** restoration boundaries are mask-based, not
    rectangle-based.
2.  **Analyze before restoring:** estimate mosaic degradation parameters
    before selecting a reconstruction strategy.
3.  **Observable information first:** exploit neighboring-frame evidence
    before model hallucination.
4.  **Temporal quality is a first-class requirement:** single-frame
    visual quality alone is insufficient.
5.  **Model abstraction:** do not couple the product permanently to
    BasicVSR++, DCNv2, or one detector.
6.  **GPU-resident pipeline:** minimize transfers between GPU and system
    memory.
7.  **Adaptive execution:** temporal window, tile size, model, and
    memory use respond to source complexity and hardware.
8.  **PTS correctness:** synchronization is more important than
    frame-index convenience.
9.  **Recoverability:** long-running jobs require checkpointing and
    graceful failure.
10. **Measurable quality:** detection, tracking, restoration, temporal
    consistency, performance, and memory are evaluated independently.

------------------------------------------------------------------------

## 18. Target Processing Flow

``` text
Input Video
    |
    v
FFmpeg Demux
    |
    v
NVDEC / HW Decode
    |
    v
PTS-aware Frame Scheduler
    |
    +----------------------------+
    |                            |
    v                            v
Scene Cut Detection       Mosaic Segmentation
                                 |
                                 v
                         ByteTrack + Kalman
                                 |
                                 v
                       Mosaic Profile Estimation
                                 |
                                 v
                        Temporal ROI Stabilizer
                                 |
                                 v
                       Adaptive Window Builder
                                 |
                                 v
                      Restoration Strategy Router
                         /          |          \
                        /           |           \
                 Multi-Frame    Single-Frame   Pass
                 Restoration    Restoration    Through
                        \           |           /
                         \          |          /
                                 v
                       Temporal Consistency
                                 |
                                 v
                       Mask-Aware Blending
                                 |
                                 v
                            GPU Frame
                                 |
                                 v
                              NVENC
                                 |
                                 v
                 Audio / Subtitle / Metadata Mux
                                 |
                                 v
                           Output Video
```

------------------------------------------------------------------------

## 19. Final Product Principle

DeMosaic Studio shall not be implemented as a simple:

`Mosaic ROI -> VSR -> Paste Back`

pipeline.

The intended architecture is:

`Detect -> Segment -> Track -> Analyze Degradation -> Gather Temporal Evidence -> Align -> Reconstruct -> Estimate Confidence -> Enforce Temporal Consistency -> Mask-Aware Blend -> Encode`

This distinction is fundamental to achieving stable and credible
multi-frame restoration.
