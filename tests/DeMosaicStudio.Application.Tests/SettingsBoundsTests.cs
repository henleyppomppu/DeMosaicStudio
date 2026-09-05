using DeMosaicStudio.Application.Settings;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Application.Tests;

/// <summary>
/// The ranges the settings dialog enforces.
/// </summary>
/// <remarks>
/// They existed only as prose in the XML documentation on <see cref="JobSettings"/>, where nothing
/// could enforce them: a typed-in value went to the worker unchecked. These fix the numbers and the
/// clamping in one place so the dialog and the settings file cannot disagree.
/// </remarks>
public sealed class SettingsBoundsTests
{
    public static TheoryData<string, Bound> All => new()
    {
        { nameof(SettingsBounds.Confidence), SettingsBounds.Confidence },
        { nameof(SettingsBounds.MaskThreshold), SettingsBounds.MaskThreshold },
        { nameof(SettingsBounds.NmsIou), SettingsBounds.NmsIou },
        { nameof(SettingsBounds.MinRegionArea), SettingsBounds.MinRegionArea },
        { nameof(SettingsBounds.MinConfirmFrames), SettingsBounds.MinConfirmFrames },
        { nameof(SettingsBounds.MaxMissingFrames), SettingsBounds.MaxMissingFrames },
        { nameof(SettingsBounds.DetectEvery), SettingsBounds.DetectEvery },
        { nameof(SettingsBounds.TemporalAlpha), SettingsBounds.TemporalAlpha },
        { nameof(SettingsBounds.RefineStrength), SettingsBounds.RefineStrength },
        { nameof(SettingsBounds.RefineSteps), SettingsBounds.RefineSteps },
        { nameof(SettingsBounds.PaddingRatio), SettingsBounds.PaddingRatio },
        { nameof(SettingsBounds.AlignConfMin), SettingsBounds.AlignConfMin },
        { nameof(SettingsBounds.MinRestorationConfidence), SettingsBounds.MinRestorationConfidence },
        { nameof(SettingsBounds.FeatherWidth), SettingsBounds.FeatherWidth },
        { nameof(SettingsBounds.ConstantQuality), SettingsBounds.ConstantQuality },
    };

    [Theory]
    [MemberData(nameof(All))]
    public void Every_range_is_ordered_and_contains_its_own_default(string name, Bound bound)
    {
        Assert.True(bound.Minimum <= bound.Maximum, name);
        Assert.InRange(bound.Default, bound.Minimum, bound.Maximum);
    }

    [Fact]
    public void The_defaults_here_are_the_defaults_the_shipped_settings_use()
    {
        // Two statements of the same number is one too many; this is the one that catches a drift.
        var shipped = new JobSettings();

        Assert.Equal(SettingsBounds.Confidence.Default, shipped.Detection.Confidence);
        Assert.Equal(SettingsBounds.MaskThreshold.Default, shipped.Detection.MaskThreshold);
        Assert.Equal(SettingsBounds.NmsIou.Default, shipped.Detection.NmsIou);
        Assert.Equal(SettingsBounds.MinRegionArea.Default, shipped.Detection.MinRegionArea);
        Assert.Equal(SettingsBounds.MinConfirmFrames.Default, shipped.Detection.MinConfirmFrames);
        Assert.Equal(SettingsBounds.MaxMissingFrames.Default, shipped.Detection.MaxMissingFrames);
        Assert.Equal(SettingsBounds.DetectEvery.Default, shipped.Detection.DetectEvery);
        Assert.Equal(SettingsBounds.TemporalAlpha.Default, shipped.Restoration.TemporalAlpha);
        Assert.Equal(SettingsBounds.RefineStrength.Default, shipped.Restoration.Refine.Strength);
        Assert.Equal(SettingsBounds.RefineSteps.Default, shipped.Restoration.Refine.Steps);
        Assert.Equal(SettingsBounds.PaddingRatio.Default, shipped.Restoration.PaddingRatio);
        Assert.Equal(SettingsBounds.AlignConfMin.Default, shipped.Restoration.AlignConfMin);
        Assert.Equal(
            SettingsBounds.MinRestorationConfidence.Default,
            shipped.Restoration.MinRestorationConfidence);
        Assert.Equal(SettingsBounds.FeatherWidth.Default, shipped.Restoration.FeatherWidth);
        Assert.Equal(SettingsBounds.ConstantQuality.Default, shipped.Encode.ConstantQuality);
    }

    [Fact]
    public void The_shipped_settings_are_already_inside_every_range()
    {
        var shipped = new JobSettings();
        Assert.Equal(shipped, SettingsBounds.Clamp(shipped));
    }

    [Fact]
    public void A_value_below_the_floor_comes_back_at_the_floor()
    {
        var clamped = SettingsBounds.Clamp(new JobSettings
        {
            Detection = new DetectionSettings { Confidence = -5.0, MinRegionArea = 0 },
        });

        Assert.Equal(SettingsBounds.Confidence.Minimum, clamped.Detection.Confidence);
        Assert.Equal(SettingsBounds.MinRegionArea.Minimum, clamped.Detection.MinRegionArea);
    }

    [Fact]
    public void A_value_above_the_ceiling_comes_back_at_the_ceiling()
    {
        var clamped = SettingsBounds.Clamp(new JobSettings
        {
            Detection = new DetectionSettings { Confidence = 40.0 },
            Restoration = new RestorationSettings { FeatherWidth = 900 },
            Encode = new EncodeSettings { ConstantQuality = 900 },
        });

        Assert.Equal(SettingsBounds.Confidence.Maximum, clamped.Detection.Confidence);
        Assert.Equal((int)SettingsBounds.FeatherWidth.Maximum, clamped.Restoration.FeatherWidth);
        Assert.Equal((int)SettingsBounds.ConstantQuality.Maximum, clamped.Encode.ConstantQuality);
    }

    [Fact]
    public void Clamping_leaves_the_choices_it_has_no_range_for_alone()
    {
        var settings = new JobSettings
        {
            Restoration = new RestorationSettings
            {
                Preset = QualityPreset.Quality,
                TemporalWindow = TemporalWindowSetting.Fixed(9),
            },
            Encode = new EncodeSettings { Codec = OutputCodec.H264 },
            Performance = new PerformanceSettings { BatchSize = 4 },
            ComparisonPoints = [12L, 34L],
        };

        var clamped = SettingsBounds.Clamp(settings);

        Assert.Equal(QualityPreset.Quality, clamped.Restoration.Preset);
        Assert.Equal(9, clamped.Restoration.TemporalWindow.FixedValue);
        Assert.Equal(OutputCodec.H264, clamped.Encode.Codec);
        Assert.Equal(4, clamped.Performance.BatchSize);
        Assert.Equal([12L, 34L], clamped.ComparisonPoints);
    }

    [Fact]
    public void The_gate_can_be_switched_off_because_zero_is_inside_its_range()
    {
        // §5.8.1: zero reproduces pre-v3.1 behaviour exactly, so a clamp that pushed it up would
        // silently change what every existing job does.
        var clamped = SettingsBounds.Clamp(new JobSettings
        {
            Restoration = new RestorationSettings { MinRestorationConfidence = 0.0 },
        });

        Assert.Equal(0.0, clamped.Restoration.MinRestorationConfidence);
    }
}
