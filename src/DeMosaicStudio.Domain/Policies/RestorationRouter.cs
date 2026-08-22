namespace DeMosaicStudio.Domain.Policies;

/// <summary>The three restoration paths. prd.md §5.8.</summary>
public enum RestorationPath
{
    /// <summary>Fuse the temporal window. Path A.</summary>
    MultiFrame,

    /// <summary>Restore from the target frame alone. Path B.</summary>
    SingleFrame,

    /// <summary>Emit the source pixels unmodified. Path C.</summary>
    PassThrough,
}

/// <summary>
/// Closed enum of routing reasons. prd.md §5.8: "A router that cannot explain itself cannot be
/// debugged on a two-hour file." Every decision carries one of these into diagnostics.
/// </summary>
public enum RouteReason
{
    /// <summary>Enough aligned neighbours and phase diversity to fuse.</summary>
    SufficientTemporalEvidence,

    /// <summary>Window truncated by a scene cut (§5.12).</summary>
    SceneCutTruncatedWindow,

    /// <summary>Mean alignment confidence below <c>align_conf_min</c> (§5.7).</summary>
    PoorAlignment,

    /// <summary>Only the target frame is valid.</summary>
    SingleValidFrame,

    /// <summary>Occlusion invalidated the neighbours.</summary>
    OcclusionInvalidatedNeighbours,

    /// <summary>Grid moves with the object, so neighbours carry no new information (§5.4.4).</summary>
    ObjectAnchoredGrid,

    /// <summary>The OOM ladder forced the window below the multi-frame minimum (§5.14).</summary>
    VramForcedSingleFrame,

    /// <summary>No mosaic region in this frame.</summary>
    NoRegion,

    /// <summary>Region below <c>min_region_area</c> (§5.2.3).</summary>
    RegionTooSmall,

    /// <summary>Region not yet confirmed on enough consecutive frames (§5.2.5b).</summary>
    RegionUnconfirmed,

    /// <summary>The user disabled restoration for this region.</summary>
    UserDisabled,

    /// <summary>Restoration confidence below <c>minRestorationConfidence</c> (§5.8.1).</summary>
    LowConfidenceGate,

    /// <summary>An explicit failure exhausted the degradation chain (§5.8.2).</summary>
    DegradationChainExhausted,
}

/// <summary>The routing decision for one region on one frame.</summary>
/// <param name="Path">Which path runs.</param>
/// <param name="Reason">Why. Always recorded in diagnostics.</param>
public readonly record struct RouteDecision(RestorationPath Path, RouteReason Reason);

/// <summary>Everything the router needs, as data. prd.md §5.8.</summary>
/// <param name="HasRegion">Whether a mosaic region is present at all.</param>
/// <param name="RegionAreaPixels">Region area in source pixels.</param>
/// <param name="MinRegionArea">Threshold from settings (§5.2.3).</param>
/// <param name="IsConfirmed">Whether the region met <c>min_confirm_frames</c> (§5.2.5b).</param>
/// <param name="UserDisabled">Whether the user excluded this region.</param>
/// <param name="WithheldByConfidenceGate">Result of <see cref="ConfidenceGate.ShouldWithhold"/> (§5.8.1).</param>
/// <param name="DegradationChainExhausted">Whether E4002/E4003/E4004 handling ran out of options (§5.8.2).</param>
/// <param name="GridAnchor">Grid anchoring for the track (§5.4.4).</param>
/// <param name="WindowDecision">Effective temporal window and why it was reduced (§5.6.1).</param>
/// <param name="ValidAlignedNeighbours">Neighbours that survived the alignment-confidence filter (§5.7).</param>
/// <param name="MeanAlignmentConfidence">Mean confidence of those neighbours.</param>
/// <param name="AlignConfMin">Threshold from settings (§5.7).</param>
/// <param name="OcclusionInvalidatedNeighbours">Whether occlusion removed the neighbours.</param>
public readonly record struct RouteInputs(
    bool HasRegion,
    int RegionAreaPixels,
    int MinRegionArea,
    bool IsConfirmed,
    bool UserDisabled,
    bool WithheldByConfidenceGate,
    bool DegradationChainExhausted,
    GridAnchor GridAnchor,
    TemporalWindowDecision WindowDecision,
    int ValidAlignedNeighbours,
    double MeanAlignmentConfidence,
    double AlignConfMin,
    bool OcclusionInvalidatedNeighbours);

/// <summary>
/// Selects the restoration path per track, per frame. prd.md §5.8.
/// <para>
/// Pure and total: every input combination maps to exactly one <see cref="RouteDecision"/>, which is
/// what makes the exhaustiveness test in <c>T-ROUTER-REASON-01</c> possible.
/// </para>
/// </summary>
public static class RestorationRouter
{
    /// <summary>Minimum neighbours for the multi-frame path. prd.md §5.8 Path A.</summary>
    public const int MinNeighboursForMultiFrame = 2;

    /// <summary>Routes one region on one frame.</summary>
    public static RouteDecision Route(RouteInputs inputs)
    {
        // Path C first: these are reasons not to touch the pixels at all.
        if (!inputs.HasRegion)
        {
            return new RouteDecision(RestorationPath.PassThrough, RouteReason.NoRegion);
        }

        if (inputs.UserDisabled)
        {
            return new RouteDecision(RestorationPath.PassThrough, RouteReason.UserDisabled);
        }

        if (inputs.RegionAreaPixels < inputs.MinRegionArea)
        {
            return new RouteDecision(RestorationPath.PassThrough, RouteReason.RegionTooSmall);
        }

        if (!inputs.IsConfirmed)
        {
            return new RouteDecision(RestorationPath.PassThrough, RouteReason.RegionUnconfirmed);
        }

        if (inputs.DegradationChainExhausted)
        {
            return new RouteDecision(RestorationPath.PassThrough, RouteReason.DegradationChainExhausted);
        }

        if (inputs.WithheldByConfidenceGate)
        {
            return new RouteDecision(RestorationPath.PassThrough, RouteReason.LowConfidenceGate);
        }

        // Path B: reasons the temporal window cannot be trusted.
        if (inputs.GridAnchor == GridAnchor.ObjectTracked)
        {
            return new RouteDecision(RestorationPath.SingleFrame, RouteReason.ObjectAnchoredGrid);
        }

        if (inputs.OcclusionInvalidatedNeighbours)
        {
            return new RouteDecision(RestorationPath.SingleFrame, RouteReason.OcclusionInvalidatedNeighbours);
        }

        if (inputs.WindowDecision.EffectiveWindow <= TemporalWindowPolicy.SingleFrame)
        {
            var reason = inputs.WindowDecision.Reason switch
            {
                WindowReductionReason.SceneBoundary => RouteReason.SceneCutTruncatedWindow,
                WindowReductionReason.VramPressure => RouteReason.VramForcedSingleFrame,
                WindowReductionReason.ObjectAnchoredGrid => RouteReason.ObjectAnchoredGrid,
                _ => RouteReason.SingleValidFrame,
            };

            return new RouteDecision(RestorationPath.SingleFrame, reason);
        }

        if (inputs.ValidAlignedNeighbours < MinNeighboursForMultiFrame)
        {
            var reason = inputs.WindowDecision.Reason == WindowReductionReason.SceneBoundary
                ? RouteReason.SceneCutTruncatedWindow
                : RouteReason.SingleValidFrame;

            return new RouteDecision(RestorationPath.SingleFrame, reason);
        }

        if (inputs.MeanAlignmentConfidence < inputs.AlignConfMin)
        {
            return new RouteDecision(RestorationPath.SingleFrame, RouteReason.PoorAlignment);
        }

        return new RouteDecision(RestorationPath.MultiFrame, RouteReason.SufficientTemporalEvidence);
    }
}
