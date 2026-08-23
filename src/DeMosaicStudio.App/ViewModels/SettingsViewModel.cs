using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using DeMosaicStudio.Application.Settings;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.App.ViewModels;

/// <summary>
/// The settings dialog's editable copy of <see cref="JobSettings"/>.
/// <para>
/// A copy, not the live object: the dialog can be cancelled, and half-applied settings are how a
/// queue ends up running with a mixture of two configurations. <see cref="ToSettings"/> is the only
/// way out, and it clamps.
/// </para>
/// <para>
/// The ranges come from <see cref="SettingsBounds"/> rather than from literals here, so the dialog
/// and the store cannot disagree about what is valid.
/// </para>
/// </summary>
public sealed class SettingsViewModel : INotifyPropertyChanged
{
    private readonly JobSettings _original;

    private double _confidence;
    private double _maskThreshold;
    private int _minRegionArea;
    private int _minConfirmFrames;
    private int _maxMissingFrames;
    private QualityPreset _preset;
    private string _temporalWindow = "auto";
    private double _alignConfMin;
    private double _minRestorationConfidence;
    private int _featherWidth;
    private OutputCodec _codec;
    private EncoderProfile _profile;
    private int _constantQuality;

    /// <summary>Creates an editable copy of the given settings.</summary>
    public SettingsViewModel(JobSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        _original = settings;
        Reset(settings);
    }

    /// <inheritdoc />
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>The quality presets, for the combo box.</summary>
    public static IReadOnlyList<QualityPreset> Presets { get; } = Enum.GetValues<QualityPreset>();

    /// <summary>The output codecs, for the combo box.</summary>
    public static IReadOnlyList<OutputCodec> Codecs { get; } = Enum.GetValues<OutputCodec>();

    /// <summary>The encoder profiles, for the combo box.</summary>
    public static IReadOnlyList<EncoderProfile> Profiles { get; } = Enum.GetValues<EncoderProfile>();

    /// <summary>The temporal window choices, as the protocol spells them.</summary>
    /// <remarks>
    /// Invariant, because these are protocol tokens rather than numbers being shown to someone:
    /// the same strings go into the fingerprint and onto the wire.
    /// </remarks>
    public static IReadOnlyList<string> TemporalWindows { get; } =
    [
        "auto",
        .. TemporalWindowSetting.AllowedValues.Select(k => k.ToString(CultureInfo.InvariantCulture)),
    ];

    /// <summary>Detection confidence threshold (§5.2.3).</summary>
    public double Confidence
    {
        get => _confidence;
        set => Set(ref _confidence, value);
    }

    /// <summary>Mask binarization threshold.</summary>
    public double MaskThreshold
    {
        get => _maskThreshold;
        set => Set(ref _maskThreshold, value);
    }

    /// <summary>Minimum region area in source pixels.</summary>
    public int MinRegionArea
    {
        get => _minRegionArea;
        set => Set(ref _minRegionArea, value);
    }

    /// <summary>Consecutive confirmations before a region is restored (§5.2.5b).</summary>
    public int MinConfirmFrames
    {
        get => _minConfirmFrames;
        set => Set(ref _minConfirmFrames, value);
    }

    /// <summary>Frames a track survives without a detection (§5.3.4).</summary>
    public int MaxMissingFrames
    {
        get => _maxMissingFrames;
        set => Set(ref _maxMissingFrames, value);
    }

    /// <summary>Quality preset (§15).</summary>
    public QualityPreset Preset
    {
        get => _preset;
        set => Set(ref _preset, value);
    }

    /// <summary>Temporal window: <c>auto</c>, or one of 3, 5, 7, 9 (§5.6.1).</summary>
    public string TemporalWindow
    {
        get => _temporalWindow;
        set => Set(ref _temporalWindow, value);
    }

    /// <summary>Minimum per-neighbour alignment confidence (§5.7).</summary>
    public double AlignConfMin
    {
        get => _alignConfMin;
        set => Set(ref _alignConfMin, value);
    }

    /// <summary>
    /// Confidence gate: regions below it keep their original pixels. Zero is off (§5.8.1).
    /// </summary>
    public double MinRestorationConfidence
    {
        get => _minRestorationConfidence;
        set => Set(ref _minRestorationConfidence, value);
    }

    /// <summary>Edge-aware feather width in pixels (§5.11).</summary>
    public int FeatherWidth
    {
        get => _featherWidth;
        set => Set(ref _featherWidth, value);
    }

    /// <summary>Output codec.</summary>
    public OutputCodec Codec
    {
        get => _codec;
        set => Set(ref _codec, value);
    }

    /// <summary>Encoder profile (D-12).</summary>
    public EncoderProfile Profile
    {
        get => _profile;
        set => Set(ref _profile, value);
    }

    /// <summary>Constant-quality target. Lower is better and larger.</summary>
    public int ConstantQuality
    {
        get => _constantQuality;
        set => Set(ref _constantQuality, value);
    }

    /// <summary>Puts every knob back to the shipped default.</summary>
    public void RestoreDefaults() => Reset(new JobSettings());

    /// <summary>Puts every knob back to what the dialog opened with.</summary>
    public void Revert() => Reset(_original);

    /// <summary>
    /// Builds the settings the dialog is describing, clamped to the documented ranges.
    /// </summary>
    /// <remarks>
    /// Performance knobs and comparison points are carried through untouched: the dialog does not
    /// edit them, and rebuilding them from defaults would silently discard whatever set them.
    /// </remarks>
    public JobSettings ToSettings() => SettingsBounds.Clamp(_original with
    {
        Detection = _original.Detection with
        {
            Confidence = Confidence,
            MaskThreshold = MaskThreshold,
            MinRegionArea = MinRegionArea,
            MinConfirmFrames = MinConfirmFrames,
            MaxMissingFrames = MaxMissingFrames,
        },
        Restoration = _original.Restoration with
        {
            Preset = Preset,
            // Invariant on the way back too: this is the same protocol token, round-tripped.
            TemporalWindow = int.TryParse(TemporalWindow, CultureInfo.InvariantCulture, out var k)
                && TemporalWindowSetting.AllowedValues.Contains(k)
                    ? TemporalWindowSetting.Fixed(k)
                    : TemporalWindowSetting.Auto,
            AlignConfMin = AlignConfMin,
            MinRestorationConfidence = MinRestorationConfidence,
            FeatherWidth = FeatherWidth,
        },
        Encode = _original.Encode with
        {
            Profile = Profile,
            Codec = Codec,
            ConstantQuality = ConstantQuality,
        },
    });

    private void Reset(JobSettings settings)
    {
        Confidence = settings.Detection.Confidence;
        MaskThreshold = settings.Detection.MaskThreshold;
        MinRegionArea = settings.Detection.MinRegionArea;
        MinConfirmFrames = settings.Detection.MinConfirmFrames;
        MaxMissingFrames = settings.Detection.MaxMissingFrames;

        Preset = settings.Restoration.Preset;
        TemporalWindow = settings.Restoration.TemporalWindow.ToString();
        AlignConfMin = settings.Restoration.AlignConfMin;
        MinRestorationConfidence = settings.Restoration.MinRestorationConfidence;
        FeatherWidth = settings.Restoration.FeatherWidth;

        Profile = settings.Encode.Profile;
        Codec = settings.Encode.Codec;
        ConstantQuality = settings.Encode.ConstantQuality;
    }

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
