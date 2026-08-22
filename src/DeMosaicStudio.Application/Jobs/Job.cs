using DeMosaicStudio.Domain.Diagnostics;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Application.Jobs;

/// <summary>Where a job is in its life. prd.md §8.5.</summary>
/// <remarks>
/// The order matters: <see cref="JobStatus"/> is compared to decide whether a reported state is a
/// step forward or a stale message arriving late. A worker's progress can overtake its own result
/// on a loaded machine, and a finished job that resurrects into "restoring 65%" is the failure this
/// enum exists to make impossible.
/// </remarks>
public enum JobStatus
{
    /// <summary>Queued, nothing started.</summary>
    Pending = 0,

    /// <summary>Being inspected: media facts, hardware, model availability.</summary>
    Probing = 1,

    /// <summary>Detection and tracking, without restoring anything.</summary>
    Analyzing = 2,

    /// <summary>The full pipeline is running.</summary>
    Processing = 3,

    /// <summary>Finished and the output is written.</summary>
    Completed = 4,

    /// <summary>Stopped on request. A terminal state, and reachable from any active one.</summary>
    Cancelled = 5,

    /// <summary>Stopped by an error. Terminal, and carries the code that explains it.</summary>
    Failed = 6,
}

/// <summary>One video's worth of work.</summary>
/// <remarks>
/// A record rather than a mutable object: the queue replaces jobs instead of mutating them, so a
/// view holding an old one sees a consistent snapshot rather than a half-applied update.
/// </remarks>
public sealed record Job
{
    /// <summary>Stable identity, assigned when the job is queued.</summary>
    public required string Id { get; init; }

    /// <summary>What to restore.</summary>
    public required string SourcePath { get; init; }

    /// <summary>Where the result goes.</summary>
    public required string OutputPath { get; init; }

    /// <summary>The settings this job runs with.</summary>
    public JobSettings Settings { get; init; } = new();

    /// <summary>Where it is.</summary>
    public JobStatus Status { get; init; } = JobStatus.Pending;

    /// <summary>Fraction complete in <c>[0, 1]</c>, or null before anything is known.</summary>
    public double? Fraction { get; init; }

    /// <summary>A short line for the user, in their language. Never a stack trace.</summary>
    public string? Message { get; init; }

    /// <summary>Set when <see cref="Status"/> is <see cref="JobStatus.Failed"/>.</summary>
    public ErrorCode? Error { get; init; }

    /// <summary>What the run produced, once it has finished.</summary>
    public JobSummary? Summary { get; init; }

    /// <summary>True once the job can no longer change on its own.</summary>
    public bool IsTerminal =>
        Status is JobStatus.Completed or JobStatus.Cancelled or JobStatus.Failed;

    /// <summary>True while the job is the queue's business rather than the user's.</summary>
    public bool IsActive =>
        Status is JobStatus.Probing or JobStatus.Analyzing or JobStatus.Processing;
}

/// <summary>What a finished run reports. Mirrors the worker's <c>result.summary</c>.</summary>
public sealed record JobSummary
{
    /// <summary>Frames the decoder saw.</summary>
    public int FramesSeen { get; init; }

    /// <summary>Frames whose pixels were changed.</summary>
    public int FramesRestored { get; init; }

    /// <summary>Regions found across the whole job, before gating.</summary>
    public int RegionsDetected { get; init; }

    /// <summary>Regions withheld because their confidence fell below the gate (§5.8.1).</summary>
    public int RegionsGated { get; init; }

    /// <summary>Counts per routing reason, from the router's closed enum.</summary>
    public IReadOnlyDictionary<string, int> RouteReasons { get; init; } =
        new Dictionary<string, int>();

    /// <summary>
    /// The video stream was copied rather than re-encoded, so it is byte-identical (R-1.8a).
    /// </summary>
    public bool Passthrough { get; init; }

    /// <summary>
    /// The output contains estimated pixels. §1.3: where information was destroyed the result is
    /// synthetic, and the user is told so rather than left to assume otherwise.
    /// </summary>
    public bool Synthetic { get; init; }

    /// <summary>The §5.1.7 timeline check, as a line the user can read.</summary>
    public string? Timeline { get; init; }
}
