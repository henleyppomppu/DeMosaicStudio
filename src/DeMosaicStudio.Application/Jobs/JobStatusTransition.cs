namespace DeMosaicStudio.Application.Jobs;

/// <summary>
/// Which status changes are allowed. prd.md §8.4, §8.5.
/// <para>
/// Progress messages are advisory and can arrive out of order — the protocol says so explicitly,
/// and a loaded machine makes it routine. Without a rule, a message that left the worker before its
/// result but arrived after it puts a finished job back into "processing 65%", where it stays.
/// </para>
/// <para>
/// So the rule is stated once, as data, and applied everywhere a status arrives. It is pure and
/// takes its inputs as values, which is what keeps it testable without a worker (§13.2).
/// </para>
/// </summary>
public static class JobStatusTransition
{
    /// <summary>True when a job in <paramref name="from"/> may move to <paramref name="to"/>.</summary>
    /// <remarks>
    /// Three rules, and nothing else:
    /// <list type="number">
    /// <item>A terminal state is terminal. Nothing leaves it — not even another terminal state.</item>
    /// <item>Active stages move forward only. Probing may reach Processing; Processing may not go
    /// back to Probing, which is what a late message looks like.</item>
    /// <item>Cancelled and Failed are reachable from anywhere that is not already terminal,
    /// including Pending — a job can be cancelled before it starts.</item>
    /// </list>
    /// <para>
    /// <b>Completed is reachable only from an active stage.</b> A job that never ran cannot have
    /// finished, and a worker claiming otherwise is reporting about something else.
    /// </para>
    /// </remarks>
    public static bool IsAllowed(JobStatus from, JobStatus to)
    {
        if (IsTerminal(from))
        {
            return false;
        }

        if (to is JobStatus.Cancelled or JobStatus.Failed)
        {
            return true;
        }

        if (to == JobStatus.Completed)
        {
            return IsActive(from);
        }

        if (to == JobStatus.Pending)
        {
            // Nothing goes back to the queue on its own. Re-queueing is a retry, which builds a
            // new job rather than reversing this one.
            return false;
        }

        return Rank(to) > Rank(from);
    }

    /// <summary>Applies a status if the rule allows it, and returns the job either way.</summary>
    /// <remarks>
    /// Returning the unchanged job rather than throwing is deliberate: a late message is normal
    /// traffic, not a fault, and §8.4 says the host drops it rather than erroring.
    /// </remarks>
    public static Job WithStatus(this Job job, JobStatus status) =>
        IsAllowed(job.Status, status) ? job with { Status = status } : job;

    /// <summary>True for a state a job cannot leave.</summary>
    public static bool IsTerminal(JobStatus status) =>
        status is JobStatus.Completed or JobStatus.Cancelled or JobStatus.Failed;

    /// <summary>True for a state the engine is working in.</summary>
    public static bool IsActive(JobStatus status) =>
        status is JobStatus.Probing or JobStatus.Analyzing or JobStatus.Processing;

    private static int Rank(JobStatus status) => status switch
    {
        JobStatus.Pending => 0,
        JobStatus.Probing => 1,
        JobStatus.Analyzing => 2,
        JobStatus.Processing => 3,
        _ => 4,
    };
}
