using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Domain.Policies;

/// <summary>How the mosaic grid is anchored. prd.md §1.4.1, §5.4.4.</summary>
public enum GridAnchor
{
    /// <summary>Grid fixed to frame coordinates. Subject motion yields phase diversity: multi-frame pays.</summary>
    Screen,

    /// <summary>Grid moves with the object. The same pixels land in the same block every frame: multi-frame buys nothing.</summary>
    ObjectTracked,

    /// <summary>Not yet determined, or indistinguishable because the subject is static.</summary>
    Unknown,
}

/// <summary>Why the effective window is smaller than the requested one. prd.md §5.6.1, warning W4103.</summary>
public enum WindowReductionReason
{
    /// <summary>The requested window was honoured.</summary>
    None,

    /// <summary>Truncated to the frames available in the same scene (§5.12).</summary>
    SceneBoundary,

    /// <summary>Truncated at the start or end of the stream.</summary>
    StreamBoundary,

    /// <summary>No phase diversity to spend frames on (§5.4.4).</summary>
    ObjectAnchoredGrid,

    /// <summary>Stepped down by the OOM ladder (§5.14 step 2).</summary>
    VramPressure,
}

/// <summary>The window actually used for one frame, and why it differs from the request.</summary>
/// <param name="EffectiveWindow">Frames actually fused, including the target frame.</param>
/// <param name="RequestedWindow">What the settings and motion policy asked for.</param>
/// <param name="Reason">The binding constraint, or <see cref="WindowReductionReason.None"/>.</param>
public readonly record struct TemporalWindowDecision(
    int EffectiveWindow,
    int RequestedWindow,
    WindowReductionReason Reason)
{
    /// <summary>True when a safety rule reduced the window and W4103 should be emitted.</summary>
    public bool WasReduced => EffectiveWindow < RequestedWindow;
}

/// <summary>Everything the window policy needs, as data. No device or file dependency (prd.md §13.2).</summary>
/// <param name="Setting">The user's <c>temporalWindow</c> setting (§5.6.1).</param>
/// <param name="Preset">Quality preset, which caps the window (§15).</param>
/// <param name="MedianFlowPixelsPerFrame">Motion estimate for the region.</param>
/// <param name="GridAnchor">Grid anchoring for the track (§5.4.4).</param>
/// <param name="SameSceneFramesAvailable">Frames available within the current scene, target included (§5.12).</param>
/// <param name="StreamFramesAvailable">Frames available within the stream, target included.</param>
/// <param name="VramMaxWindow">Ceiling imposed by the OOM ladder; 9 when unconstrained (§5.14).</param>
public readonly record struct TemporalWindowInputs(
    TemporalWindowSetting Setting,
    QualityPreset Preset,
    double MedianFlowPixelsPerFrame,
    GridAnchor GridAnchor,
    int SameSceneFramesAvailable,
    int StreamFramesAvailable,
    int VramMaxWindow);

/// <summary>
/// Selects the temporal window. prd.md §5.6 (adaptive policy) and §5.6.1 (user override).
/// <para>
/// The rule that matters: a fixed <c>temporalWindow</c> replaces the <b>motion-based</b> choice only.
/// Every safety reduction still applies on top of it. A user-forced window that ignored them would be
/// a corruption and OOM lever rather than a quality control.
/// </para>
/// </summary>
public static class TemporalWindowPolicy
{
    /// <summary>Window used when there is no temporal context at all.</summary>
    public const int SingleFrame = 1;

    /// <summary>Motion below this is "low". prd.md §5.6.</summary>
    public const double LowMotionThreshold = 1.0;

    /// <summary>Motion above this is "high". prd.md §5.6.</summary>
    public const double HighMotionThreshold = 6.0;

    /// <summary>
    /// Reconciles §5.6's motion table with §15's per-preset windows: the preset sets the ceiling and
    /// motion chooses within it. Balanced therefore lands on 5 at medium motion (§15) and 7 at low
    /// motion (§5.6), which is what both tables say. Recorded as ADR D-13.
    /// </summary>
    public static int PresetMaxWindow(QualityPreset preset) => preset switch
    {
        QualityPreset.Fast => 3,
        QualityPreset.Balanced => 7,
        QualityPreset.Quality => 9,
        _ => throw new ArgumentOutOfRangeException(nameof(preset), preset, "Unknown quality preset."),
    };

    /// <summary>Decides the window for one frame.</summary>
    public static TemporalWindowDecision Decide(TemporalWindowInputs inputs)
    {
        var requested = Requested(inputs);

        // Each constraint proposes a ceiling. The smallest wins; ties resolve by this order, which
        // runs most-specific first so the reported reason is the informative one.
        var constraints = new (int Ceiling, WindowReductionReason Reason)[]
        {
            (inputs.GridAnchor == GridAnchor.ObjectTracked ? SingleFrame : requested, WindowReductionReason.ObjectAnchoredGrid),
            (OddAtMost(inputs.SameSceneFramesAvailable), WindowReductionReason.SceneBoundary),
            (OddAtMost(inputs.StreamFramesAvailable), WindowReductionReason.StreamBoundary),
            (OddAtMost(inputs.VramMaxWindow), WindowReductionReason.VramPressure),
        };

        var effective = requested;
        var reason = WindowReductionReason.None;

        foreach (var (ceiling, candidateReason) in constraints)
        {
            if (ceiling < effective)
            {
                effective = ceiling;
                reason = candidateReason;
            }
        }

        return new TemporalWindowDecision(Math.Max(effective, SingleFrame), requested, reason);
    }

    /// <summary>The window asked for, before any safety reduction: the override, or the motion policy.</summary>
    public static int Requested(TemporalWindowInputs inputs)
    {
        var cap = PresetMaxWindow(inputs.Preset);

        if (inputs.Setting.FixedValue is { } forced)
        {
            // A fixed value overrides the motion policy and the preset's motion-based choice alike.
            // It is still bounded by what the pipeline supports.
            return forced;
        }

        var byMotion = inputs.MedianFlowPixelsPerFrame switch
        {
            < LowMotionThreshold => 9,
            <= HighMotionThreshold => 5,
            _ => 3,
        };

        return Math.Min(byMotion, cap);
    }

    // Windows are centred on the target frame, so only odd sizes are meaningful. A ceiling of 6
    // available frames supports a window of 5, not 6.
    private static int OddAtMost(int value)
    {
        if (value < SingleFrame)
        {
            return SingleFrame;
        }

        var clamped = Math.Min(value, 9);
        return clamped % 2 == 0 ? clamped - 1 : clamped;
    }
}
