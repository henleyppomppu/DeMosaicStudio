using DeMosaicStudio.Application.Jobs;
using DeMosaicStudio.Domain.Diagnostics;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Application.Engine;

/// <summary>
/// What the application can ask of a restoration engine. prd.md §8.
/// <para>
/// Deliberately says nothing about how the work happens. AGENTS.md forbids Application from knowing
/// the engine is a process or that it is Python, and this interface is where that line is drawn: an
/// in-process fake and a spawned worker satisfy it identically, which is what lets the queue be
/// tested without either.
/// </para>
/// </summary>
public interface IRestorationEngine
{
    /// <summary>Starts the engine and reports what it can do.</summary>
    Task<EngineCapabilities> StartAsync(CancellationToken cancellationToken = default);

    /// <summary>Reads a source's media facts without processing it.</summary>
    Task<MediaFacts> ProbeAsync(string sourcePath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Detects and tracks without restoring, so the user can see what would be altered before
    /// committing to it (§5.2.5c). Writes nothing.
    /// </summary>
    Task<EngineOutcome> AnalyzeAsync(
        EngineRequest request,
        IProgress<EngineProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>Runs the whole pipeline and writes the output.</summary>
    Task<EngineOutcome> ProcessAsync(
        EngineRequest request,
        IProgress<EngineProgress>? progress = null,
        CancellationToken cancellationToken = default);
}

/// <summary>What one job asks the engine to do.</summary>
public sealed record EngineRequest
{
    /// <summary>The job's identity, echoed back on every message about it.</summary>
    public required string JobId { get; init; }

    /// <summary>What to read.</summary>
    public required string SourcePath { get; init; }

    /// <summary>Where to write. Empty for an analysis, which writes nothing.</summary>
    public string OutputPath { get; init; } = string.Empty;

    /// <summary>The settings to run with.</summary>
    public JobSettings Settings { get; init; } = new();

    /// <summary>Whether to reuse artifacts a previous attempt left behind (§9).</summary>
    public bool Resume { get; init; }

    /// <summary>For an analysis: examine every Nth frame. One means every frame.</summary>
    public int SampleEvery { get; init; } = 1;
}

/// <summary>One progress report. Advisory: §8.4 says these may be dropped or arrive late.</summary>
public sealed record EngineProgress
{
    /// <summary>Which job this is about.</summary>
    public required string JobId { get; init; }

    /// <summary>The stage the engine is in, as the protocol spells it.</summary>
    public required string Stage { get; init; }

    /// <summary>Fraction complete in <c>[0, 1]</c>, or null when it is not yet known.</summary>
    public double? Fraction { get; init; }

    /// <summary>Frames per second, when the engine is in a position to say.</summary>
    public double? Fps { get; init; }
}

/// <summary>How a job ended.</summary>
public sealed record EngineOutcome
{
    /// <summary>The terminal status.</summary>
    public required JobStatus Status { get; init; }

    /// <summary>What the run produced, when it produced anything.</summary>
    public JobSummary? Summary { get; init; }

    /// <summary>The numbered failure, when it failed.</summary>
    public ErrorCode? Error { get; init; }

    /// <summary>A line for the user. Never a stack trace, never a full source path (§2.3 C-6).</summary>
    public string? Message { get; init; }
}

/// <summary>What the engine reports it can do, at handshake.</summary>
public sealed record EngineCapabilities
{
    /// <summary>The engine's own version.</summary>
    public string Version { get; init; } = string.Empty;

    /// <summary>The protocol version it speaks.</summary>
    public string ProtocolVersion { get; init; } = string.Empty;

    /// <summary>Whether a usable CUDA device was found — usable, not merely present (§5.17).</summary>
    public bool CudaAvailable { get; init; }

    /// <summary>The device it will run on, for display.</summary>
    public string Device { get; init; } = "cpu";

    /// <summary>Models it can load, as <c>id/version</c> pairs.</summary>
    public IReadOnlyList<string> Models { get; init; } = [];
}

/// <summary>What a probe found. prd.md §5.1.</summary>
public sealed record MediaFacts
{
    /// <summary>Pixel dimensions.</summary>
    public int Width { get; init; }

    /// <summary>Pixel dimensions.</summary>
    public int Height { get; init; }

    /// <summary>Duration in seconds, as the container reports it.</summary>
    public double DurationSeconds { get; init; }

    /// <summary>The video codec's name.</summary>
    public string VideoCodec { get; init; } = string.Empty;

    /// <summary>
    /// Whether the source's timeline is variable. §5.1.7's guarantee is the same either way, but
    /// the user is told, because a VFR source is where timing bugs live.
    /// </summary>
    public bool IsVariableFrameRate { get; init; }

    /// <summary>How many audio streams the source carries. They are copied, never re-encoded.</summary>
    public int AudioStreams { get; init; }

    /// <summary>How many subtitle streams the source carries.</summary>
    public int SubtitleStreams { get; init; }
}
