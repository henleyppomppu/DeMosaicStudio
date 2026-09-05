using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Application.Settings;

/// <summary>One editable knob's permitted range, as prd.md states it.</summary>
/// <param name="Minimum">Lowest accepted value.</param>
/// <param name="Maximum">Highest accepted value.</param>
/// <param name="Default">What the shipped configuration uses.</param>
public sealed record Bound(double Minimum, double Maximum, double Default)
{
    /// <summary>Brings a value inside the range.</summary>
    public double Clamp(double value) => Math.Clamp(value, Minimum, Maximum);

    /// <summary>Brings a value inside the range and rounds it to an integer.</summary>
    public int ClampInt(double value) => (int)Math.Round(Clamp(value), MidpointRounding.AwayFromZero);
}

/// <summary>
/// The ranges the settings dialog enforces. prd.md sections 5.2.3, 5.2.5b, 5.3.4, 5.5.1, 5.7, 5.11.
/// <para>
/// The numbers were written only in the XML documentation on <see cref="JobSettings"/>, where
/// nothing could enforce them and a typed-in value reached the worker unchecked. Stating them once,
/// here, is what lets the dialog and a test read the same figure.
/// </para>
/// </summary>
public static class SettingsBounds
{
    /// <summary>Detection confidence threshold (section 5.2.3).</summary>
    public static Bound Confidence { get; } = new(0.10, 0.90, 0.45);

    /// <summary>Mask binarization threshold.</summary>
    public static Bound MaskThreshold { get; } = new(0.30, 0.70, 0.50);

    /// <summary>Box-level NMS IoU used for track association.</summary>
    public static Bound NmsIou { get; } = new(0.30, 0.80, 0.50);

    /// <summary>Minimum region area in source pixels.</summary>
    public static Bound MinRegionArea { get; } = new(64, 4096, 256);

    /// <summary>Consecutive confirmations before a region is restored (section 5.2.5b).</summary>
    public static Bound MinConfirmFrames { get; } = new(1, 5, 2);

    /// <summary>Frames a track survives without a detection (section 5.3.4).</summary>
    public static Bound MaxMissingFrames { get; } = new(0, 15, 3);

    /// <summary>Detector cadence in frames (D-43). Above 8 a new region is found too late to matter.</summary>
    public static Bound DetectEvery { get; } = new(1, 8, 1);

    /// <summary>ROI padding ratio (section 5.5.1).</summary>
    public static Bound PaddingRatio { get; } = new(0.10, 0.20, 0.15);

    /// <summary>Minimum per-neighbour alignment confidence (section 5.7).</summary>
    public static Bound AlignConfMin { get; } = new(0.0, 1.0, 0.35);

    /// <summary>Confidence gate; zero is off (section 5.8.1).</summary>
    public static Bound MinRestorationConfidence { get; } = new(0.0, 1.0, 0.0);

    /// <summary>Edge-aware feather width in pixels (section 5.11).</summary>
    public static Bound FeatherWidth { get; } = new(1, 9, 3);

    /// <summary>Temporal blend weight of the single-frame path (D-43). 1 is no blending.</summary>
    public static Bound TemporalAlpha { get; } = new(0.0, 1.0, 0.3);

    /// <summary>Constant-quality target.</summary>
    public static Bound ConstantQuality { get; } = new(0, 51, 18);

    /// <summary>Brings every value in a settings object inside its range.</summary>
    /// <remarks>
    /// Applied on the way out of the dialog and on the way in from the store, because a settings
    /// file edited by hand is exactly as untrusted as a text box.
    /// </remarks>
    public static JobSettings Clamp(JobSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        return settings with
        {
            Detection = settings.Detection with
            {
                Confidence = Confidence.Clamp(settings.Detection.Confidence),
                MaskThreshold = MaskThreshold.Clamp(settings.Detection.MaskThreshold),
                NmsIou = NmsIou.Clamp(settings.Detection.NmsIou),
                MinRegionArea = MinRegionArea.ClampInt(settings.Detection.MinRegionArea),
                MinConfirmFrames = MinConfirmFrames.ClampInt(settings.Detection.MinConfirmFrames),
                MaxMissingFrames = MaxMissingFrames.ClampInt(settings.Detection.MaxMissingFrames),
                DetectEvery = DetectEvery.ClampInt(settings.Detection.DetectEvery),
            },
            Restoration = settings.Restoration with
            {
                PaddingRatio = PaddingRatio.Clamp(settings.Restoration.PaddingRatio),
                AlignConfMin = AlignConfMin.Clamp(settings.Restoration.AlignConfMin),
                MinRestorationConfidence =
                    MinRestorationConfidence.Clamp(settings.Restoration.MinRestorationConfidence),
                FeatherWidth = FeatherWidth.ClampInt(settings.Restoration.FeatherWidth),
                TemporalAlpha = TemporalAlpha.Clamp(settings.Restoration.TemporalAlpha),
            },
            Encode = settings.Encode with
            {
                ConstantQuality = ConstantQuality.ClampInt(settings.Encode.ConstantQuality),
            },
        };
    }
}
