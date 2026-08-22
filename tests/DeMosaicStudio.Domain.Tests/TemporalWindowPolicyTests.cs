using DeMosaicStudio.Domain.Policies;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>prd.md §5.6 and §5.6.1.</summary>
public sealed class TemporalWindowPolicyTests
{
    private static TemporalWindowInputs Unconstrained(
        TemporalWindowSetting? setting = null,
        QualityPreset preset = QualityPreset.Balanced,
        double motion = 3.0,
        GridAnchor anchor = GridAnchor.Screen) =>
        new(
            setting ?? TemporalWindowSetting.Auto,
            preset,
            motion,
            anchor,
            SameSceneFramesAvailable: 9,
            StreamFramesAvailable: 9,
            VramMaxWindow: 9);

    /// <summary>T-WINDOW-POLICY-01..03 — the motion table, within the preset's ceiling.</summary>
    [Theory]
    [InlineData(0.5, QualityPreset.Balanced, 7)]   // low motion, capped by Balanced
    [InlineData(0.5, QualityPreset.Quality, 9)]    // low motion, Quality allows 9
    [InlineData(3.0, QualityPreset.Balanced, 5)]   // medium motion
    [InlineData(12.0, QualityPreset.Balanced, 3)]  // high motion
    [InlineData(0.5, QualityPreset.Fast, 3)]       // Fast caps at 3 regardless of motion
    public void The_motion_policy_selects_the_window_within_the_preset_ceiling(
        double motion, QualityPreset preset, int expected)
    {
        var decision = TemporalWindowPolicy.Decide(Unconstrained(preset: preset, motion: motion));

        Assert.Equal(expected, decision.EffectiveWindow);
        Assert.Equal(WindowReductionReason.None, decision.Reason);
        Assert.False(decision.WasReduced);
    }

    /// <summary>T-WINDOW-POLICY-04 — a scene boundary truncates to the frames available in the scene.</summary>
    [Fact]
    public void A_scene_boundary_truncates_the_window()
    {
        var inputs = Unconstrained(motion: 0.5) with { SameSceneFramesAvailable = 3 };

        var decision = TemporalWindowPolicy.Decide(inputs);

        Assert.Equal(3, decision.EffectiveWindow);
        Assert.Equal(WindowReductionReason.SceneBoundary, decision.Reason);
        Assert.True(decision.WasReduced);
    }

    /// <summary>
    /// T-WINDOW-POLICY-05 — an object-anchored grid collapses to a single frame. There is no phase
    /// diversity to spend neighbours on (prd.md §1.4.1), so a larger window buys nothing but VRAM.
    /// </summary>
    [Fact]
    public void An_object_anchored_grid_collapses_to_a_single_frame()
    {
        var decision = TemporalWindowPolicy.Decide(Unconstrained(anchor: GridAnchor.ObjectTracked, motion: 0.5));

        Assert.Equal(TemporalWindowPolicy.SingleFrame, decision.EffectiveWindow);
        Assert.Equal(WindowReductionReason.ObjectAnchoredGrid, decision.Reason);
    }

    /// <summary>T-WINDOW-POLICY-06 — VRAM pressure steps the window down.</summary>
    [Fact]
    public void Vram_pressure_steps_the_window_down()
    {
        var inputs = Unconstrained(motion: 0.5) with { VramMaxWindow = 3 };

        var decision = TemporalWindowPolicy.Decide(inputs);

        Assert.Equal(3, decision.EffectiveWindow);
        Assert.Equal(WindowReductionReason.VramPressure, decision.Reason);
    }

    /// <summary>T-WINDOW-OVERRIDE-01 — a fixed setting is honoured in steady state.</summary>
    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(9)]
    public void A_fixed_window_is_honoured_when_nothing_constrains_it(int k)
    {
        // Motion is high, which the adaptive policy would answer with 3. The override wins.
        var decision = TemporalWindowPolicy.Decide(
            Unconstrained(setting: TemporalWindowSetting.Fixed(k), motion: 20.0));

        Assert.Equal(k, decision.EffectiveWindow);
        Assert.Equal(k, decision.RequestedWindow);
        Assert.Equal(WindowReductionReason.None, decision.Reason);
    }

    /// <summary>
    /// T-WINDOW-OVERRIDE-SAFETY-01..04 — every safety reduction still fires with a fixed window set.
    /// <para>
    /// This is the requirement that keeps <c>temporalWindow</c> a quality control rather than a
    /// corruption and OOM lever (prd.md §5.6.1).
    /// </para>
    /// </summary>
    [Fact]
    public void A_fixed_window_does_not_override_the_scene_boundary()
    {
        var inputs = Unconstrained(setting: TemporalWindowSetting.Fixed(9)) with { SameSceneFramesAvailable = 3 };

        var decision = TemporalWindowPolicy.Decide(inputs);

        Assert.Equal(3, decision.EffectiveWindow);
        Assert.Equal(9, decision.RequestedWindow);
        Assert.Equal(WindowReductionReason.SceneBoundary, decision.Reason);
        Assert.True(decision.WasReduced);
    }

    /// <inheritdoc cref="A_fixed_window_does_not_override_the_scene_boundary"/>
    [Fact]
    public void A_fixed_window_does_not_override_an_object_anchored_grid()
    {
        var decision = TemporalWindowPolicy.Decide(
            Unconstrained(setting: TemporalWindowSetting.Fixed(9), anchor: GridAnchor.ObjectTracked));

        Assert.Equal(TemporalWindowPolicy.SingleFrame, decision.EffectiveWindow);
        Assert.Equal(WindowReductionReason.ObjectAnchoredGrid, decision.Reason);
    }

    /// <inheritdoc cref="A_fixed_window_does_not_override_the_scene_boundary"/>
    [Fact]
    public void A_fixed_window_does_not_override_the_vram_ladder()
    {
        var inputs = Unconstrained(setting: TemporalWindowSetting.Fixed(9)) with { VramMaxWindow = 5 };

        var decision = TemporalWindowPolicy.Decide(inputs);

        Assert.Equal(5, decision.EffectiveWindow);
        Assert.Equal(WindowReductionReason.VramPressure, decision.Reason);
    }

    /// <inheritdoc cref="A_fixed_window_does_not_override_the_scene_boundary"/>
    [Fact]
    public void A_fixed_window_does_not_override_the_stream_boundary()
    {
        var inputs = Unconstrained(setting: TemporalWindowSetting.Fixed(9)) with { StreamFramesAvailable = 1 };

        var decision = TemporalWindowPolicy.Decide(inputs);

        Assert.Equal(TemporalWindowPolicy.SingleFrame, decision.EffectiveWindow);
        Assert.Equal(WindowReductionReason.StreamBoundary, decision.Reason);
    }

    /// <summary>
    /// Only odd windows are meaningful because the window is centred on the target frame: six
    /// available frames support a window of five, not six.
    /// </summary>
    [Fact]
    public void An_even_frame_budget_yields_the_next_lower_odd_window()
    {
        var inputs = Unconstrained(motion: 0.5) with { SameSceneFramesAvailable = 6 };

        Assert.Equal(5, TemporalWindowPolicy.Decide(inputs).EffectiveWindow);
    }

    /// <summary>The setting rejects windows the pipeline does not support, rather than clamping silently.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(11)]
    public void An_unsupported_fixed_window_is_rejected(int k) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => TemporalWindowSetting.Fixed(k));
}
