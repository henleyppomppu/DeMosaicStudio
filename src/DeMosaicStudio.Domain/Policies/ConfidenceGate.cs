namespace DeMosaicStudio.Domain.Policies;

/// <summary>Restoration confidence bucket. prd.md §5.9.4.</summary>
public enum ConfidenceBucket
{
    /// <summary>Below 0.33 — the result depends heavily on model estimation.</summary>
    Low,

    /// <summary>0.33 to 0.66 — temporal inference contributes significantly.</summary>
    Medium,

    /// <summary>0.66 and above — substantial observable information from neighbouring frames.</summary>
    High,
}

/// <summary>Bucket boundaries, mirrored in <c>worker/demosaic_worker/confidence.py</c> (prd.md §13.4).</summary>
public static class Confidence
{
    /// <summary>Lower bound of <see cref="ConfidenceBucket.High"/>.</summary>
    public const double HighThreshold = 0.66;

    /// <summary>Lower bound of <see cref="ConfidenceBucket.Medium"/>.</summary>
    public const double MediumThreshold = 0.33;

    /// <summary>Classifies a confidence value.</summary>
    public static ConfidenceBucket Bucket(double confidence) => confidence switch
    {
        >= HighThreshold => ConfidenceBucket.High,
        >= MediumThreshold => ConfidenceBucket.Medium,
        _ => ConfidenceBucket.Low,
    };
}

/// <summary>
/// The <c>minRestorationConfidence</c> gate. prd.md §5.8.1.
/// <para>
/// When a region's restoration confidence is too low, the restored ROI is discarded and the original
/// pixels are kept unmodified — nothing is blended, so there is no seam to hide.
/// </para>
/// <para>
/// <b>The decision is per track with hysteresis, never independently per frame.</b> A raw per-frame
/// threshold applied to a marginal region flips between restored and original from frame to frame,
/// which looks far worse than either choice held consistently — and it does so precisely where
/// confidence is borderline, which is exactly where this feature is supposed to help.
/// </para>
/// <para>
/// <b>A track starts gated.</b> Restoration is an intervention: it takes evidence to begin, not
/// evidence to stop.
/// </para>
/// <para>
/// This type holds mutable per-track state and is not thread-safe. One instance per job, driven from
/// the pipeline's ordered stage.
/// </para>
/// </summary>
public sealed class ConfidenceGate
{
    /// <summary>Consecutive frames required before the gate state flips. prd.md §5.8.1 R-8.1c.</summary>
    public const int DefaultHysteresisFrames = 3;

    /// <summary>How far <b>below</b> the threshold confidence must fall to re-close the gate.</summary>
    public const double DefaultReleaseMargin = 0.05;

    private readonly Dictionary<int, TrackGateState> _byTrack = [];
    private readonly double _threshold;
    private readonly int _hysteresisFrames;
    private readonly double _releaseMargin;

    /// <summary>Creates a gate.</summary>
    /// <param name="threshold">
    /// <c>minRestorationConfidence</c>. Zero or below disables the gate entirely, reproducing
    /// pre-v3.1 behaviour exactly (R-8.1b).
    /// </param>
    /// <param name="hysteresisFrames">Consecutive frames required to flip state.</param>
    /// <param name="releaseMargin">Margin above the threshold required to un-gate.</param>
    public ConfidenceGate(
        double threshold,
        int hysteresisFrames = DefaultHysteresisFrames,
        double releaseMargin = DefaultReleaseMargin)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(hysteresisFrames, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(releaseMargin);

        _threshold = threshold;
        _hysteresisFrames = hysteresisFrames;
        _releaseMargin = releaseMargin;
    }

    /// <summary>True when the gate is switched off and no region will ever be withheld.</summary>
    public bool IsDisabled => _threshold <= 0.0;

    /// <summary>Number of regions currently withheld, for the job summary (R-8.1d).</summary>
    public int GatedTrackCount => _byTrack.Values.Count(s => s.IsGated);

    /// <summary>
    /// Feeds one frame's smoothed confidence for a track and returns whether the restored result
    /// should be withheld in favour of the original pixels.
    /// </summary>
    public bool ShouldWithhold(int trackId, double smoothedConfidence)
    {
        if (IsDisabled)
        {
            return false;
        }

        ref var state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(
            _byTrack, trackId, out var existed);

        if (!existed)
        {
            // A track starts gated. Restoration is an intervention: it should take evidence to
            // begin, not evidence to stop. Starting open meant every new track was restored for
            // (hysteresisFrames - 1) frames regardless of confidence — a gate set above every
            // reachable confidence still let two frames per track through.
            state.IsGated = true;
        }

        // The margin sits on the *closing* side. A user who sets minRestorationConfidence to X means
        // "restore where confidence is at least X"; putting the margin on the opening side made X
        // itself unreachable, and made every X within a margin of the confidence ceiling silently
        // mean "never restore".
        if (state.IsGated)
        {
            if (smoothedConfidence >= _threshold)
            {
                state.ConsecutiveAbove++;
                if (state.ConsecutiveAbove >= _hysteresisFrames)
                {
                    state.IsGated = false;
                    state.ConsecutiveBelow = 0;
                    state.ConsecutiveAbove = 0;
                }
            }
            else
            {
                state.ConsecutiveAbove = 0;
            }
        }
        else
        {
            if (smoothedConfidence < _threshold - _releaseMargin)
            {
                state.ConsecutiveBelow++;
                if (state.ConsecutiveBelow >= _hysteresisFrames)
                {
                    state.IsGated = true;
                    state.ConsecutiveAbove = 0;
                    state.ConsecutiveBelow = 0;
                }
            }
            else
            {
                state.ConsecutiveBelow = 0;
            }
        }

        return state.IsGated;
    }

    /// <summary>Drops per-track state when a track terminates, so a long job does not accumulate it.</summary>
    public void Forget(int trackId) => _byTrack.Remove(trackId);

    private struct TrackGateState
    {
        public bool IsGated;
        public int ConsecutiveBelow;
        public int ConsecutiveAbove;
    }
}
