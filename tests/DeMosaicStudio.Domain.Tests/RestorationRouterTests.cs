using DeMosaicStudio.Domain.Policies;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>prd.md §5.8, §5.8.1, §5.8.2.</summary>
public sealed class RestorationRouterTests
{
    private static RouteInputs Healthy() => new(
        HasRegion: true,
        RegionAreaPixels: 4096,
        MinRegionArea: 256,
        IsConfirmed: true,
        UserDisabled: false,
        WithheldByConfidenceGate: false,
        DegradationChainExhausted: false,
        GridAnchor: GridAnchor.Screen,
        MedianFlowPixelsPerFrame: 0.5,   // slow: inside the measured operating window (D-16)
        WindowDecision: new TemporalWindowDecision(3, 3, WindowReductionReason.None),
        ValidAlignedNeighbours: 4,
        MeanAlignmentConfidence: 0.80,
        AlignConfMin: 0.35,
        OcclusionInvalidatedNeighbours: false);

    /// <summary>The happy path fuses the window.</summary>
    [Fact]
    public void Sufficient_evidence_routes_to_multi_frame()
    {
        var decision = RestorationRouter.Route(Healthy());

        Assert.Equal(RestorationPath.MultiFrame, decision.Path);
        Assert.Equal(RouteReason.SufficientTemporalEvidence, decision.Reason);
    }

    /// <summary>Each Path C condition, with the reason it must report.</summary>
    [Fact]
    public void No_region_passes_through() =>
        AssertRoute(Healthy() with { HasRegion = false }, RestorationPath.PassThrough, RouteReason.NoRegion);

    /// <inheritdoc cref="No_region_passes_through"/>
    [Fact]
    public void A_region_below_the_minimum_area_passes_through() =>
        AssertRoute(Healthy() with { RegionAreaPixels = 100 }, RestorationPath.PassThrough, RouteReason.RegionTooSmall);

    /// <inheritdoc cref="No_region_passes_through"/>
    [Fact]
    public void An_unconfirmed_region_passes_through() =>
        AssertRoute(Healthy() with { IsConfirmed = false }, RestorationPath.PassThrough, RouteReason.RegionUnconfirmed);

    /// <inheritdoc cref="No_region_passes_through"/>
    [Fact]
    public void A_user_disabled_region_passes_through() =>
        AssertRoute(Healthy() with { UserDisabled = true }, RestorationPath.PassThrough, RouteReason.UserDisabled);

    /// <summary>
    /// T-CONF-GATE-01 at the router level — a region withheld by the confidence gate is passed
    /// through, so the output keeps its original pixels rather than a blended low-evidence guess.
    /// </summary>
    [Fact]
    public void A_region_withheld_by_the_confidence_gate_passes_through() =>
        AssertRoute(
            Healthy() with { WithheldByConfidenceGate = true },
            RestorationPath.PassThrough,
            RouteReason.LowConfidenceGate);

    /// <summary>prd.md §5.8.2 — the degradation chain terminates in untouched source pixels.</summary>
    [Fact]
    public void An_exhausted_degradation_chain_passes_through() =>
        AssertRoute(
            Healthy() with { DegradationChainExhausted = true },
            RestorationPath.PassThrough,
            RouteReason.DegradationChainExhausted);

    /// <summary>Each Path B condition.</summary>
    [Fact]
    public void An_object_anchored_grid_routes_to_single_frame() =>
        AssertRoute(
            Healthy() with { GridAnchor = GridAnchor.ObjectTracked },
            RestorationPath.SingleFrame,
            RouteReason.ObjectAnchoredGrid);

    /// <inheritdoc cref="An_object_anchored_grid_routes_to_single_frame"/>
    [Fact]
    public void Poor_alignment_routes_to_single_frame() =>
        AssertRoute(
            Healthy() with { MeanAlignmentConfidence = 0.10 },
            RestorationPath.SingleFrame,
            RouteReason.PoorAlignment);

    /// <inheritdoc cref="An_object_anchored_grid_routes_to_single_frame"/>
    [Fact]
    public void Occlusion_routes_to_single_frame() =>
        AssertRoute(
            Healthy() with { OcclusionInvalidatedNeighbours = true },
            RestorationPath.SingleFrame,
            RouteReason.OcclusionInvalidatedNeighbours);

    /// <inheritdoc cref="An_object_anchored_grid_routes_to_single_frame"/>
    [Fact]
    public void Too_few_valid_neighbours_routes_to_single_frame() =>
        AssertRoute(
            Healthy() with { ValidAlignedNeighbours = 1 },
            RestorationPath.SingleFrame,
            RouteReason.SingleValidFrame);

    /// <summary>
    /// The motion gate used to send both ends of the range to single-frame. Re-measured against the
    /// accumulator, neither end is harmful and the fast end is the best of all (D-31).
    /// </summary>
    [Theory]
    [InlineData(0.05)]   // static
    [InlineData(20.0)]   // fast
    public void Motion_alone_no_longer_forces_single_frame(double motion)
    {
        var decision = RestorationRouter.Route(Healthy() with { MedianFlowPixelsPerFrame = motion });

        Assert.NotEqual(RouteReason.MotionOutsideOperatingWindow, decision.Reason);
    }

    /// <summary>Medium motion needs good alignment before it qualifies; slow motion does not.</summary>
    [Fact]
    public void Medium_motion_with_poor_alignment_routes_to_single_frame() =>
        AssertRoute(
            Healthy() with { MedianFlowPixelsPerFrame = 3.0, MeanAlignmentConfidence = 0.4 },
            RestorationPath.SingleFrame,
            RouteReason.MotionOutsideOperatingWindow);

    /// <inheritdoc cref="An_object_anchored_grid_routes_to_single_frame"/>
    [Fact]
    public void A_scene_cut_truncated_window_routes_to_single_frame() =>
        AssertRoute(
            Healthy() with
            {
                WindowDecision = new TemporalWindowDecision(1, 5, WindowReductionReason.SceneBoundary),
            },
            RestorationPath.SingleFrame,
            RouteReason.SceneCutTruncatedWindow);

    /// <inheritdoc cref="An_object_anchored_grid_routes_to_single_frame"/>
    [Fact]
    public void A_vram_collapsed_window_routes_to_single_frame() =>
        AssertRoute(
            Healthy() with
            {
                WindowDecision = new TemporalWindowDecision(1, 5, WindowReductionReason.VramPressure),
            },
            RestorationPath.SingleFrame,
            RouteReason.VramForcedSingleFrame);

    /// <summary>
    /// T-ROUTER-REASON-01 — the reason enum is exhaustive over the branch conditions: sweeping every
    /// combination of the router's boolean and categorical inputs produces no decision the enum
    /// cannot name, and never produces the same reason for two different paths.
    /// </summary>
    [Fact]
    public void Every_input_combination_yields_a_named_reason()
    {
        var seen = new HashSet<RouteReason>();
        var reasonToPath = new Dictionary<RouteReason, RestorationPath>();

        foreach (var hasRegion in Bools)
        foreach (var confirmed in Bools)
        foreach (var disabled in Bools)
        foreach (var gated in Bools)
        foreach (var exhausted in Bools)
        foreach (var occluded in Bools)
        foreach (var anchor in Enum.GetValues<GridAnchor>())
        foreach (var reduction in Enum.GetValues<WindowReductionReason>())
        foreach (var motion in new[] { 0.1, 0.5, 3.0, 12.0 })
        foreach (var window in new[] { 1, 3, 5 })
        foreach (var neighbours in new[] { 0, 1, 4 })
        foreach (var alignment in new[] { 0.10, 0.80 })
        foreach (var area in new[] { 100, 4096 })
        {
            var decision = RestorationRouter.Route(new RouteInputs(
                hasRegion,
                area,
                MinRegionArea: 256,
                confirmed,
                disabled,
                gated,
                exhausted,
                anchor,
                motion,
                new TemporalWindowDecision(window, 5, reduction),
                neighbours,
                alignment,
                AlignConfMin: 0.35,
                occluded));

            Assert.True(Enum.IsDefined(decision.Reason));
            Assert.True(Enum.IsDefined(decision.Path));

            seen.Add(decision.Reason);

            if (reasonToPath.TryGetValue(decision.Reason, out var previousPath))
            {
                Assert.Equal(previousPath, decision.Path);
            }
            else
            {
                reasonToPath[decision.Reason] = decision.Path;
            }
        }

        // Every reason in the closed enum is reachable. An unreachable reason is either a dead branch
        // or a missing one, and both are worth failing on.
        var unreachable = Enum.GetValues<RouteReason>().Except(seen).ToList();
        Assert.Empty(unreachable);
    }

    private static readonly bool[] Bools = [false, true];

    private static void AssertRoute(RouteInputs inputs, RestorationPath path, RouteReason reason)
    {
        var decision = RestorationRouter.Route(inputs);

        Assert.Equal(path, decision.Path);
        Assert.Equal(reason, decision.Reason);
    }
}
