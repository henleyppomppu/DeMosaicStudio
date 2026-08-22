using DeMosaicStudio.Domain.Policies;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>prd.md §5.8.1.</summary>
public sealed class ConfidenceGateTests
{
    private const int TrackId = 7;

    /// <summary>
    /// T-CONF-GATE-DEFAULT-01 — the default of 0.00 gates nothing, so v3.1 reproduces earlier
    /// behaviour exactly until the user opts in (R-8.1b).
    /// </summary>
    [Fact]
    public void The_default_threshold_never_withholds_anything()
    {
        var gate = new ConfidenceGate(threshold: 0.0);

        Assert.True(gate.IsDisabled);

        for (var frame = 0; frame < 100; frame++)
        {
            Assert.False(gate.ShouldWithhold(TrackId, smoothedConfidence: 0.0));
        }

        Assert.Equal(0, gate.GatedTrackCount);
    }

    /// <summary>
    /// T-CONF-GATE-01 — a track starts gated, and sustained low confidence keeps it that way.
    /// <para>
    /// It used to start <i>open</i>, which meant every new track was restored for
    /// (hysteresis - 1) frames no matter how low its confidence. That was found by measuring: a
    /// gate set above every reachable confidence still let two frames per track through.
    /// Restoration is an intervention — it takes evidence to begin, not evidence to stop.
    /// </para>
    /// </summary>
    [Fact]
    public void Sustained_low_confidence_withholds_the_region_after_the_hysteresis_window()
    {
        var gate = new ConfidenceGate(threshold: 0.40);

        Assert.True(gate.ShouldWithhold(TrackId, 0.10));
        Assert.True(gate.ShouldWithhold(TrackId, 0.10));
        Assert.True(gate.ShouldWithhold(TrackId, 0.10));
        Assert.True(gate.ShouldWithhold(TrackId, 0.10));

        Assert.Equal(1, gate.GatedTrackCount);
    }

    /// <summary>A confident track waits out the hysteresis window before it is trusted.</summary>
    [Fact]
    public void A_confident_track_opens_the_gate_and_stays_open()
    {
        var gate = new ConfidenceGate(threshold: 0.40);

        Assert.True(gate.ShouldWithhold(TrackId, 0.90));
        Assert.True(gate.ShouldWithhold(TrackId, 0.90));
        Assert.False(gate.ShouldWithhold(TrackId, 0.90));

        for (var frame = 0; frame < 20; frame++)
        {
            Assert.False(gate.ShouldWithhold(TrackId, 0.90));
        }

        Assert.Equal(0, gate.GatedTrackCount);
    }

    /// <summary>
    /// T-CONF-GATE-HYSTERESIS-01 — a confidence signal oscillating around the threshold produces
    /// <b>no</b> state change, rather than one per frame.
    /// <para>
    /// This is the whole reason the decision is per track rather than per frame. A raw threshold here
    /// would flip the region between restored and original on alternate frames — which looks far
    /// worse than either choice held consistently, and does so precisely where confidence is
    /// borderline.
    /// </para>
    /// </summary>
    [Fact]
    public void An_oscillating_confidence_signal_does_not_flip_the_gate()
    {
        var gate = new ConfidenceGate(threshold: 0.40);

        var transitions = 0;
        var previous = true; // a track starts gated

        for (var frame = 0; frame < 200; frame++)
        {
            var confidence = frame % 2 == 0 ? 0.39 : 0.41;
            var withheld = gate.ShouldWithhold(TrackId, confidence);

            if (withheld != previous)
            {
                transitions++;
                previous = withheld;
            }
        }

        Assert.Equal(0, transitions);
    }

    /// <summary>
    /// A signal that genuinely crosses and stays across flips exactly once in each direction, and the
    /// release requires clearing the threshold by the margin — not merely touching it.
    /// </summary>
    [Fact]
    public void A_sustained_recovery_releases_the_gate_once()
    {
        var gate = new ConfidenceGate(threshold: 0.40);

        for (var frame = 0; frame < 5; frame++)
        {
            gate.ShouldWithhold(TrackId, 0.10);
        }

        Assert.True(gate.ShouldWithhold(TrackId, 0.10));

        // The threshold means what it says: confidence at or above it counts towards release.
        Assert.True(gate.ShouldWithhold(TrackId, 0.40));
        Assert.True(gate.ShouldWithhold(TrackId, 0.42));
        Assert.False(gate.ShouldWithhold(TrackId, 0.40));

        // The margin guards the closing side now, so a dip inside it does not re-close.
        Assert.False(gate.ShouldWithhold(TrackId, 0.36));
        Assert.False(gate.ShouldWithhold(TrackId, 0.36));
        Assert.False(gate.ShouldWithhold(TrackId, 0.36));

        // A fall clear of the margin, held for the window, does.
        Assert.False(gate.ShouldWithhold(TrackId, 0.10));
        Assert.False(gate.ShouldWithhold(TrackId, 0.10));
        Assert.True(gate.ShouldWithhold(TrackId, 0.10));
    }

    /// <summary>Gate state is per track: one bad track does not withhold another.</summary>
    [Fact]
    public void Gate_state_is_tracked_per_track()
    {
        var gate = new ConfidenceGate(threshold: 0.40);

        for (var frame = 0; frame < 5; frame++)
        {
            gate.ShouldWithhold(trackId: 1, smoothedConfidence: 0.05);
            gate.ShouldWithhold(trackId: 2, smoothedConfidence: 0.95);
        }

        Assert.True(gate.ShouldWithhold(trackId: 1, smoothedConfidence: 0.05));
        Assert.False(gate.ShouldWithhold(trackId: 2, smoothedConfidence: 0.95));
        Assert.Equal(1, gate.GatedTrackCount);
    }

    /// <summary>Terminated tracks are forgotten, so a two-hour job does not accumulate state.</summary>
    [Fact]
    public void Forgetting_a_track_drops_its_state()
    {
        var gate = new ConfidenceGate(threshold: 0.40);

        for (var frame = 0; frame < 5; frame++)
        {
            gate.ShouldWithhold(TrackId, 0.05);
        }

        Assert.Equal(1, gate.GatedTrackCount);

        gate.Forget(TrackId);

        Assert.Equal(0, gate.GatedTrackCount);
    }

    /// <summary>prd.md §5.9.4 bucket boundaries, mirrored on the worker side.</summary>
    [Theory]
    [InlineData(0.00, ConfidenceBucket.Low)]
    [InlineData(0.32, ConfidenceBucket.Low)]
    [InlineData(0.33, ConfidenceBucket.Medium)]
    [InlineData(0.65, ConfidenceBucket.Medium)]
    [InlineData(0.66, ConfidenceBucket.High)]
    [InlineData(1.00, ConfidenceBucket.High)]
    public void Confidence_buckets_use_the_documented_boundaries(double value, ConfidenceBucket expected) =>
        Assert.Equal(expected, Confidence.Bucket(value));
}
