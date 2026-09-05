using System.Globalization;

namespace DeMosaicStudio.Domain.Settings;

/// <summary>User-facing quality preset. prd.md §15.</summary>
public enum QualityPreset
{
    /// <summary>Throughput first: K=3, lightweight backend, low VRAM.</summary>
    Fast,

    /// <summary>Default: K=5, balanced backend.</summary>
    Balanced,

    /// <summary>Larger temporal context and stronger consistency mechanisms.</summary>
    Quality,
}

/// <summary>Encoder profile. prd.md §5.1.4, D-12.</summary>
public enum EncoderProfile
{
    /// <summary>x265 slow, CRF-based. Default; reaches transparency at a lower bitrate.</summary>
    QualityX265,

    /// <summary>NVENC. For long files and iteration.</summary>
    SpeedNvenc,
}

/// <summary>Output video codec. prd.md §5.1.4.</summary>
public enum OutputCodec
{
    /// <summary>H.264 / AVC.</summary>
    H264,

    /// <summary>H.265 / HEVC.</summary>
    H265,
}

/// <summary>
/// Temporal window setting. prd.md §5.6.1.
/// <para>
/// <see cref="IsAuto"/> means the adaptive motion policy chooses K per frame. A fixed value replaces
/// only that motion-based choice: the safety reductions (scene cut, object-anchored grid, VRAM
/// ladder, stream boundary) still apply on top of it. See <c>TemporalWindowPolicy</c>.
/// </para>
/// </summary>
public readonly record struct TemporalWindowSetting
{
    private TemporalWindowSetting(int? fixedValue) => FixedValue = fixedValue;

    /// <summary>The adaptive policy. Default.</summary>
    public static TemporalWindowSetting Auto => new(null);

    /// <summary>The requested fixed window, or null when adaptive.</summary>
    public int? FixedValue { get; }

    /// <summary>True when the adaptive policy chooses K.</summary>
    public bool IsAuto => FixedValue is null;

    /// <summary>The window sizes the pipeline supports. prd.md §5.6.</summary>
    public static IReadOnlyList<int> AllowedValues { get; } = [3, 5, 7, 9];

    /// <summary>Creates a fixed setting, rejecting values outside <see cref="AllowedValues"/>.</summary>
    public static TemporalWindowSetting Fixed(int k) =>
        AllowedValues.Contains(k)
            ? new TemporalWindowSetting(k)
            : throw new ArgumentOutOfRangeException(nameof(k), k, "temporalWindow must be one of 3, 5, 7, 9 (prd.md §5.6).");

    /// <summary>
    /// The canonical fingerprint token. The <b>requested</b> value, never the per-frame resolved K —
    /// resolving depends on scene content and would make the fingerprint unstable (prd.md §9.3).
    /// </summary>
    public override string ToString() =>
        FixedValue is { } k ? k.ToString(CultureInfo.InvariantCulture) : "auto";
}

/// <summary>Settings that affect detection and tracking. In the <c>detection</c> fingerprint (prd.md §9.3).</summary>
public sealed record DetectionSettings
{
    /// <summary>Detection confidence threshold. Default 0.45, range 0.10-0.90 (§5.2.3).</summary>
    public double Confidence { get; init; } = 0.45;

    /// <summary>Box-level NMS IoU used for track association. Default 0.50, range 0.30-0.80.</summary>
    public double NmsIou { get; init; } = 0.50;

    /// <summary>Minimum region area in source pixels. Default 256, range 64-4096.</summary>
    public int MinRegionArea { get; init; } = 256;

    /// <summary>Mask binarization threshold. Default 0.50, range 0.30-0.70.</summary>
    public double MaskThreshold { get; init; } = 0.50;

    /// <summary>Consecutive confirmations before a region is restored. Default 2, range 1-5 (§5.2.5b).</summary>
    public int MinConfirmFrames { get; init; } = 2;

    /// <summary>Frames a track survives without a detection. Default 3, range 0-15 (§5.3.4).</summary>
    public int MaxMissingFrames { get; init; } = 3;

    /// <summary>
    /// Run the detector on every Nth frame and let the tracker carry regions between. Default 1
    /// (every frame), range 1-8 (D-43). The detector is the largest fixed cost per frame once
    /// restoration is cheap, and mosaic regions do not appear and vanish between adjacent frames.
    /// </summary>
    public int DetectEvery { get; init; } = 1;
}

/// <summary>
/// The optional diffusion pass over each restored region (D-44). In the <c>restoration</c>
/// fingerprint: a different model, LoRA or strength changes every restored pixel.
/// </summary>
/// <remarks>
/// The models are not part of the product. The user places them in <c>models/diffusion</c>,
/// <c>models/lora</c> and <c>models/embeddings</c> and picks by name; there is no prompt field,
/// which is how prd.md §2.3 C-4 is satisfied by construction.
/// </remarks>
public sealed record RefineSettings
{
    /// <summary>Whether the pass runs. Off by default; costs nothing when off.</summary>
    public bool Enabled { get; init; }

    /// <summary>
    /// How far the model may depart from the restoration underneath. 0 is none; measured on the
    /// quality fixture, 0.2 beat bicubic on LPIPS by 43%, 0.3 began to invent shapes, 0.5 invented.
    /// </summary>
    public double Strength { get; init; } = 0.2;

    /// <summary>Directory name under <c>models/diffusion</c>. Empty means none chosen.</summary>
    public string Model { get; init; } = string.Empty;

    /// <summary>File name under <c>models/lora</c>, or empty for none.</summary>
    public string Lora { get; init; } = string.Empty;

    /// <summary>File names under <c>models/embeddings</c>.</summary>
    public IReadOnlyList<string> Embeddings { get; init; } = [];

    /// <summary>Denoising steps. With an LCM LoRA 4–8 is the working range.</summary>
    public int Steps { get; init; } = 8;

    /// <summary>The generator seed. Fixed so the same region refines the same way each frame.</summary>
    public int Seed { get; init; } = 7;
}

/// <summary>Settings that affect restoration. In the <c>restoration</c> fingerprint (prd.md §9.3).</summary>
public sealed record RestorationSettings
{
    /// <summary>The optional diffusion pass (D-44).</summary>
    public RefineSettings Refine { get; init; } = new();

    /// <summary>
    /// Quality preset. §15. Default <see cref="QualityPreset.Fast"/> since D-43: measured on the
    /// only footage available, the decimate-and-interpolate floor beat the super-resolution
    /// network on both PSNR and LPIPS, and a default should be what was measured best.
    /// </summary>
    public QualityPreset Preset { get; init; } = QualityPreset.Fast;

    /// <summary>ROI padding ratio. Default 0.15, range 0.10-0.20 (§5.5.1).</summary>
    public double PaddingRatio { get; init; } = 0.15;

    /// <summary>Temporal window. Default auto (§5.6.1).</summary>
    public TemporalWindowSetting TemporalWindow { get; init; } = TemporalWindowSetting.Auto;

    /// <summary>Minimum per-neighbour alignment confidence. Default 0.35 (§5.7).</summary>
    public double AlignConfMin { get; init; } = 0.35;

    /// <summary>
    /// Confidence gate. Default 0.00 = off, which reproduces pre-v3.1 behaviour exactly.
    /// Above zero, regions whose restoration confidence falls below it keep their original
    /// pixels (§5.8.1).
    /// </summary>
    public double MinRestorationConfidence { get; init; }

    /// <summary>Edge-aware feather width in pixels. Default 3, range 1-9 (§5.11).</summary>
    public int FeatherWidth { get; init; } = 3;

    /// <summary>
    /// Weight of the new frame in the single-frame path's temporal blend. Default 0.3 (a 7:3
    /// blend with the previous restoration), range 0-1 (D-43). Ignored by the Quality preset,
    /// which uses the evidence accumulator instead.
    /// </summary>
    public double TemporalAlpha { get; init; } = 0.3;
}

/// <summary>Settings that affect encoding. In the <c>encode</c> fingerprint (prd.md §9.3).</summary>
public sealed record EncodeSettings
{
    /// <summary>Encoder profile. Default x265 Quality (D-12).</summary>
    public EncoderProfile Profile { get; init; } = EncoderProfile.QualityX265;

    /// <summary>Output codec.</summary>
    public OutputCodec Codec { get; init; } = OutputCodec.H265;

    /// <summary>Constant-quality target, derived from the §5.1.8 transparency measurement per profile.</summary>
    public int ConstantQuality { get; init; } = 18;
}

/// <summary>
/// Performance knobs. <b>Excluded from every fingerprint</b> (prd.md §9.3): the OOM ladder changes
/// precision at runtime, and including these would make every resume after a downgrade discard
/// completed work.
/// </summary>
public sealed record PerformanceSettings
{
    /// <summary>VRAM budget in bytes, or null for Auto (70% of free VRAM, floor 2 GB). §5.14.</summary>
    public long? VramBudgetBytes { get; init; }

    /// <summary>Inference precision, or null to resolve against the active device. §5.16.6, §5.17b.</summary>
    public string? Precision { get; init; }

    /// <summary>Tile size in pixels, or null for untiled.</summary>
    public int? TileSize { get; init; }

    /// <summary>Restoration batch size.</summary>
    public int BatchSize { get; init; } = 1;
}

/// <summary>
/// The full resolved settings for a job, after <c>auto</c> model resolution (prd.md §14.3).
/// </summary>
public sealed record JobSettings
{
    /// <summary>Detection and tracking settings.</summary>
    public DetectionSettings Detection { get; init; } = new();

    /// <summary>Restoration settings.</summary>
    public RestorationSettings Restoration { get; init; } = new();

    /// <summary>Encode settings.</summary>
    public EncodeSettings Encode { get; init; } = new();

    /// <summary>Performance knobs. Never fingerprinted.</summary>
    public PerformanceSettings Performance { get; init; } = new();

    /// <summary>
    /// Comparison points, in source time base (prd.md §5.16.11). Diagnostic only: adding or moving
    /// one must never discard a completed restoration, so this is never fingerprinted.
    /// </summary>
    public IReadOnlyList<long> ComparisonPoints { get; init; } = [];
}
