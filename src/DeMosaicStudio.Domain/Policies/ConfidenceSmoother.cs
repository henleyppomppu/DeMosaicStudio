namespace DeMosaicStudio.Domain.Policies;

/// <summary>
/// Smooths a track's restoration confidence before <see cref="ConfidenceGate"/> sees it. prd.md §5.8.1.
/// <para>
/// The gate takes a parameter called <c>smoothedConfidence</c> and the pipeline was handing it the
/// raw per-frame value. The mismatch was not cosmetic. Confidence varies frame to frame; the gate's
/// hysteresis is per track and sticky in both directions; so a long track would open on a run of
/// good frames and then coast through the bad ones. Measured on one clip, the per-frame signal
/// could take the output from −0.82 dB to +0.075 dB, and the gate fed raw confidence could reach
/// only 0.0 — by withholding everything.
/// </para>
/// <para>
/// The time constant is the gate's own hysteresis window rather than a tuned number: the gate
/// reasons over <see cref="ConfidenceGate.DefaultHysteresisFrames"/> frames, so the signal it
/// reasons about is averaged over the same span.
/// </para>
/// <para>
/// This type holds mutable per-track state and is not thread-safe. One instance per job.
/// </para>
/// </summary>
public sealed class ConfidenceSmoother
{
    private readonly Dictionary<int, double> _byTrack = [];
    private readonly double _alpha;

    /// <summary>Creates a smoother whose time constant is <paramref name="window"/> frames.</summary>
    public ConfidenceSmoother(int window = ConfidenceGate.DefaultHysteresisFrames)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);

        _alpha = 1.0 / window;
    }

    /// <summary>The exponential weight given to the newest frame.</summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Feeds one frame's confidence and returns the smoothed value for this track.
    /// <para>
    /// The first frame of a track passes through unchanged: there is nothing to average it with,
    /// and seeding from zero would withhold every track's opening frames for a reason that has
    /// nothing to do with the evidence.
    /// </para>
    /// </summary>
    public double Update(int trackId, double confidence)
    {
        var smoothed = _byTrack.TryGetValue(trackId, out var previous)
            ? (_alpha * confidence) + ((1.0 - _alpha) * previous)
            : confidence;

        _byTrack[trackId] = smoothed;
        return smoothed;
    }

    /// <summary>Drops a terminated track's state.</summary>
    public void Forget(int trackId) => _byTrack.Remove(trackId);
}
