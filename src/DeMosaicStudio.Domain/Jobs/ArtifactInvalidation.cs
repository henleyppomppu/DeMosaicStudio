namespace DeMosaicStudio.Domain.Jobs;

/// <summary>Artifacts a job produces, in pipeline order. prd.md §9.2.</summary>
public enum JobArtifact
{
    /// <summary>Detection and tracking output (<c>analysis.jsonl</c>).</summary>
    Analysis,

    /// <summary>Partially or fully written output video.</summary>
    Video,
}

/// <summary>The fingerprints recorded against a job's artifacts. prd.md §9.2.</summary>
/// <param name="Detection">Fingerprint of the detection settings that produced the analysis.</param>
/// <param name="Restoration">Fingerprint of the restoration settings that produced the video.</param>
/// <param name="Encode">Fingerprint of the encode settings that produced the video.</param>
public readonly record struct FingerprintSet(string? Detection, string? Restoration, string? Encode);

/// <summary>
/// The resume invalidation rule. prd.md §9.3.
/// <para>
/// A single global fingerprint forces a full restart whenever anything changes; no fingerprint lets a
/// resume mix artifacts produced under different settings, which is a data-corruption bug rather than
/// a UX one. So each artifact records the fingerprint that produced it, and invalidation cascades
/// <b>top-down from the first changed stage</b>.
/// </para>
/// </summary>
public static class ArtifactInvalidation
{
    /// <summary>
    /// Returns the artifacts that must be discarded when resuming with <paramref name="current"/>
    /// against the <paramref name="recorded"/> fingerprints.
    /// </summary>
    /// <remarks>
    /// A missing or unknown recorded fingerprint compares as <b>changed</b>, never as equal. A
    /// null-lifting comparison that evaluates to <c>false</c> on unknown data silently reuses a
    /// previous file's artifacts for a different source.
    /// </remarks>
    public static IReadOnlySet<JobArtifact> Invalidated(FingerprintSet recorded, FingerprintSet current)
    {
        var discard = new HashSet<JobArtifact>();

        if (Changed(recorded.Detection, current.Detection))
        {
            // Detection feeds everything downstream.
            discard.Add(JobArtifact.Analysis);
            discard.Add(JobArtifact.Video);
        }

        if (Changed(recorded.Restoration, current.Restoration) || Changed(recorded.Encode, current.Encode))
        {
            discard.Add(JobArtifact.Video);
        }

        return discard;
    }

    /// <summary>
    /// True when a recorded fingerprint does not match the current one. Unknown compares as changed.
    /// </summary>
    public static bool Changed(string? recorded, string? current) =>
        recorded is null || current is null || !string.Equals(recorded, current, StringComparison.Ordinal);
}
