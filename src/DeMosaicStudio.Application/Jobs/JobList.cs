using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Domain.Diagnostics;

namespace DeMosaicStudio.Application.Jobs;

/// <summary>
/// The jobs, and what may be done to them — the queue the user sees. prd.md §8.5.
/// <para>
/// Named <c>JobList</c> rather than <c>JobQueue</c> because it is not one: it has no FIFO
/// semantics, and the suffix would promise them.
/// </para>
/// <para>
/// Holds no threads and starts nothing. It is a state machine over a list, so every rule it
/// enforces — what can be cancelled, what a late message does, what "start" means when nothing is
/// selected — is testable without a worker, a GPU or a clock (§13.2).
/// </para>
/// <para>
/// <b>Not thread-safe.</b> One instance, driven from the application's ordered stage. A list
/// mutated from a free-threaded callback is how a finished job resurrects into "processing 65%",
/// and the transition rule alone cannot save it if two updates interleave.
/// </para>
/// </summary>
public sealed class JobList
{
    private readonly List<Job> _jobs = [];

    /// <summary>Raised after any change, once, with the whole list.</summary>
    /// <remarks>
    /// The whole list rather than a delta: a view that rebuilds from a snapshot cannot drift, and
    /// the lists here are small enough that the alternative buys nothing but bugs.
    /// </remarks>
    public event Action<IReadOnlyList<Job>>? Changed;

    /// <summary>Every job, in the order it was added.</summary>
    public IReadOnlyList<Job> Jobs => _jobs;

    /// <summary>Jobs that could still run, oldest first.</summary>
    public IReadOnlyList<Job> Runnable =>
        _jobs.Where(job => job.Status == JobStatus.Pending).ToList();

    /// <summary>The job the engine is working on, if any.</summary>
    public Job? Active => _jobs.FirstOrDefault(job => job.IsActive);

    /// <summary>Adds a job and returns it.</summary>
    public Job Add(Job job)
    {
        _jobs.Add(job);
        Raise();
        return job;
    }

    /// <summary>Looks a job up by id.</summary>
    public Job? Find(string id) => _jobs.FirstOrDefault(job => job.Id == id);

    /// <summary>
    /// Applies a status change if <see cref="JobStatusTransition"/> allows it, and returns whether
    /// anything moved.
    /// </summary>
    /// <remarks>
    /// A refused change is not an error. §8.4 says the host drops progress that arrives after a
    /// terminal result or moves backwards, and this is where that happens.
    /// </remarks>
    public bool Report(string id, JobStatus status, string? message = null)
    {
        var index = _jobs.FindIndex(job => job.Id == id);
        if (index < 0 || !JobStatusTransition.IsAllowed(_jobs[index].Status, status))
        {
            return false;
        }

        _jobs[index] = _jobs[index] with { Status = status, Message = message ?? _jobs[index].Message };
        Raise();
        return true;
    }

    /// <summary>Applies a progress report, dropping it if it would move a job backwards.</summary>
    public bool Report(EngineProgress progress)
    {
        var index = _jobs.FindIndex(job => job.Id == progress.JobId);
        if (index < 0 || _jobs[index].IsTerminal)
        {
            return false;
        }

        var job = _jobs[index];
        var status = StageToStatus(progress.Stage);

        if (status is { } wanted && !JobStatusTransition.IsAllowed(job.Status, wanted))
        {
            // The stage went backwards. The fraction that came with it is about that stage, so it
            // is dropped too rather than applied to the one the job is really in.
            return false;
        }

        // §8.4: the fraction is monotonically non-decreasing within a job.
        if (progress.Fraction is { } fraction && fraction < (job.Fraction ?? 0.0))
        {
            return false;
        }

        _jobs[index] = job with
        {
            Status = status ?? job.Status,
            Fraction = progress.Fraction ?? job.Fraction,
        };
        Raise();
        return true;
    }

    /// <summary>Applies a terminal outcome.</summary>
    public bool Complete(string id, EngineOutcome outcome)
    {
        var index = _jobs.FindIndex(job => job.Id == id);
        if (index < 0 || !JobStatusTransition.IsAllowed(_jobs[index].Status, outcome.Status))
        {
            return false;
        }

        _jobs[index] = _jobs[index] with
        {
            Status = outcome.Status,
            Summary = outcome.Summary,
            Error = outcome.Error,
            Message = outcome.Message ?? _jobs[index].Message,
            Fraction = outcome.Status == JobStatus.Completed ? 1.0 : _jobs[index].Fraction,
        };
        Raise();
        return true;
    }

    /// <summary>
    /// Re-queues a finished job as a **new** one, so the original stays as the record of what
    /// happened.
    /// </summary>
    /// <remarks>
    /// Reversing a terminal status would erase the failure the user is retrying, and the transition
    /// rule forbids it for exactly that reason. A retry is a new attempt, not an undo.
    /// </remarks>
    public Job? Retry(string id, string newId)
    {
        var job = Find(id);
        if (job is null || !job.IsTerminal)
        {
            return null;
        }

        return Add(new Job
        {
            Id = newId,
            SourcePath = job.SourcePath,
            OutputPath = job.OutputPath,
            Settings = job.Settings,
        });
    }

    /// <summary>Removes jobs by id, refusing to remove one the engine is working on.</summary>
    /// <returns>The ids that were actually removed.</returns>
    public IReadOnlyList<string> Remove(IEnumerable<string> ids)
    {
        var wanted = ids.ToHashSet(StringComparer.Ordinal);
        var removed = _jobs.Where(job => wanted.Contains(job.Id) && !job.IsActive)
                           .Select(job => job.Id)
                           .ToList();

        if (removed.Count == 0)
        {
            return removed;
        }

        _jobs.RemoveAll(job => removed.Contains(job.Id));
        Raise();
        return removed;
    }

    private void Raise() => Changed?.Invoke(_jobs.ToList());

    /// <summary>Maps a protocol stage name onto a status, or null when it says nothing about one.</summary>
    private static JobStatus? StageToStatus(string stage) => stage.ToLowerInvariant() switch
    {
        "probing" => JobStatus.Probing,
        "analyzing" => JobStatus.Analyzing,
        "restoring" or "detecting" or "tracking" or "encoding" or "muxing" or "finalizing"
            => JobStatus.Processing,
        _ => null,
    };
}
