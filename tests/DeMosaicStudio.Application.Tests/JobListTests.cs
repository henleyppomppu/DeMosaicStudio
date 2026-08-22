using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;
using DeMosaicStudio.Domain.Diagnostics;

namespace DeMosaicStudio.Application.Tests;

/// <summary>
/// The queue's rules. prd.md §8.4, §8.5.
/// <para>
/// Almost every test here is about a message arriving at the wrong time. The protocol says progress
/// is advisory and may be dropped or reordered, and a loaded machine makes that routine rather than
/// exotic — a finished job that resurrects into "processing 65%" and stays there is the failure
/// these rules exist to make impossible.
/// </para>
/// </summary>
public sealed class JobListTests
{
    private static Job NewJob(string id = "j1") => new()
    {
        Id = id,
        SourcePath = @"C:\videos\clip.mp4",
        OutputPath = @"C:\videos\clip.restored.mp4",
    };

    private static JobList WithOne(out Job job)
    {
        var list = new JobList();
        job = list.Add(NewJob());
        return list;
    }

    // ------------------------------------------------------------------------------------------
    // The transition rule
    // ------------------------------------------------------------------------------------------

    [Theory]
    [InlineData(JobStatus.Pending, JobStatus.Probing)]
    [InlineData(JobStatus.Pending, JobStatus.Processing)]
    [InlineData(JobStatus.Probing, JobStatus.Processing)]
    [InlineData(JobStatus.Processing, JobStatus.Completed)]
    [InlineData(JobStatus.Pending, JobStatus.Cancelled)]
    [InlineData(JobStatus.Processing, JobStatus.Failed)]
    public void Forward_moves_are_allowed(JobStatus from, JobStatus to) =>
        Assert.True(JobStatusTransition.IsAllowed(from, to));

    [Theory]
    [InlineData(JobStatus.Processing, JobStatus.Probing)]
    [InlineData(JobStatus.Processing, JobStatus.Analyzing)]
    [InlineData(JobStatus.Probing, JobStatus.Pending)]
    public void Backward_moves_are_refused(JobStatus from, JobStatus to) =>
        Assert.False(JobStatusTransition.IsAllowed(from, to));

    [Theory]
    [InlineData(JobStatus.Completed)]
    [InlineData(JobStatus.Cancelled)]
    [InlineData(JobStatus.Failed)]
    public void Nothing_leaves_a_terminal_state(JobStatus terminal)
    {
        foreach (var to in Enum.GetValues<JobStatus>())
        {
            Assert.False(JobStatusTransition.IsAllowed(terminal, to), $"{terminal} -> {to}");
        }
    }

    /// <summary>A job that never ran cannot have finished.</summary>
    [Fact]
    public void Completed_is_reachable_only_from_an_active_stage()
    {
        Assert.False(JobStatusTransition.IsAllowed(JobStatus.Pending, JobStatus.Completed));
        Assert.True(JobStatusTransition.IsAllowed(JobStatus.Processing, JobStatus.Completed));
    }

    // ------------------------------------------------------------------------------------------
    // Late and out-of-order messages
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void Progress_arriving_after_a_result_is_dropped()
    {
        var list = WithOne(out var job);
        list.Report(job.Id, JobStatus.Processing);
        list.Complete(job.Id, new EngineOutcome { Status = JobStatus.Completed });

        var applied = list.Report(new EngineProgress
        {
            JobId = job.Id,
            Stage = "restoring",
            Fraction = 0.65,
        });

        Assert.False(applied);
        Assert.Equal(JobStatus.Completed, list.Find(job.Id)!.Status);
        Assert.Equal(1.0, list.Find(job.Id)!.Fraction);
    }

    [Fact]
    public void Progress_whose_stage_moved_backwards_is_dropped_along_with_its_fraction()
    {
        var list = WithOne(out var job);
        list.Report(new EngineProgress { JobId = job.Id, Stage = "restoring", Fraction = 0.5 });

        var applied = list.Report(new EngineProgress
        {
            JobId = job.Id,
            Stage = "probing",
            Fraction = 0.9,
        });

        Assert.False(applied);

        var current = list.Find(job.Id)!;
        Assert.Equal(JobStatus.Processing, current.Status);
        Assert.Equal(0.5, current.Fraction);
    }

    /// <summary>§8.4: the fraction is monotonically non-decreasing within a job.</summary>
    [Fact]
    public void A_fraction_that_goes_backwards_is_dropped()
    {
        var list = WithOne(out var job);
        list.Report(new EngineProgress { JobId = job.Id, Stage = "restoring", Fraction = 0.7 });

        Assert.False(list.Report(new EngineProgress
        {
            JobId = job.Id,
            Stage = "restoring",
            Fraction = 0.4,
        }));

        Assert.Equal(0.7, list.Find(job.Id)!.Fraction);
    }

    [Fact]
    public void Progress_for_an_unknown_job_is_dropped_rather_than_creating_one()
    {
        var list = new JobList();

        Assert.False(list.Report(new EngineProgress { JobId = "ghost", Stage = "restoring" }));
        Assert.Empty(list.Jobs);
    }

    // ------------------------------------------------------------------------------------------
    // Outcomes
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void A_completed_job_carries_its_summary_and_reaches_one()
    {
        var list = WithOne(out var job);
        list.Report(job.Id, JobStatus.Processing);

        list.Complete(job.Id, new EngineOutcome
        {
            Status = JobStatus.Completed,
            Summary = new JobSummary { FramesSeen = 96, FramesRestored = 95, Synthetic = true },
        });

        var current = list.Find(job.Id)!;
        Assert.Equal(1.0, current.Fraction);
        Assert.True(current.Summary!.Synthetic);
        Assert.True(current.IsTerminal);
    }

    [Fact]
    public void A_failed_job_keeps_its_code_and_does_not_claim_completion()
    {
        var list = WithOne(out var job);
        list.Report(job.Id, JobStatus.Processing);

        list.Complete(job.Id, new EngineOutcome
        {
            Status = JobStatus.Failed,
            Error = ErrorCodes.E4401,
            Message = "GPU memory exhausted",
        });

        var current = list.Find(job.Id)!;
        Assert.Equal(JobStatus.Failed, current.Status);
        Assert.Equal(ErrorCodes.E4401, current.Error);
        Assert.NotEqual(1.0, current.Fraction ?? 0.0);
    }

    [Fact]
    public void A_job_can_be_cancelled_before_it_starts()
    {
        var list = WithOne(out var job);

        Assert.True(list.Report(job.Id, JobStatus.Cancelled));
        Assert.Equal(JobStatus.Cancelled, list.Find(job.Id)!.Status);
    }

    // ------------------------------------------------------------------------------------------
    // Retry and removal
    // ------------------------------------------------------------------------------------------

    /// <summary>A retry is a new attempt, not an undo: the record of the failure survives it.</summary>
    [Fact]
    public void Retrying_adds_a_job_and_leaves_the_original_alone()
    {
        var list = WithOne(out var job);
        list.Report(job.Id, JobStatus.Processing);
        list.Complete(job.Id, new EngineOutcome { Status = JobStatus.Failed, Error = ErrorCodes.E4401 });

        var retried = list.Retry(job.Id, "j2");

        Assert.NotNull(retried);
        Assert.Equal(JobStatus.Pending, retried!.Status);
        Assert.Equal(job.SourcePath, retried.SourcePath);
        Assert.Equal(JobStatus.Failed, list.Find(job.Id)!.Status);
        Assert.Equal(2, list.Jobs.Count);
    }

    [Fact]
    public void A_job_that_has_not_finished_cannot_be_retried()
    {
        var list = WithOne(out var job);
        list.Report(job.Id, JobStatus.Processing);

        Assert.Null(list.Retry(job.Id, "j2"));
    }

    [Fact]
    public void Removing_skips_the_job_the_engine_is_working_on()
    {
        var list = new JobList();
        var idle = list.Add(NewJob("idle"));
        var busy = list.Add(NewJob("busy"));
        list.Report(busy.Id, JobStatus.Processing);

        var removed = list.Remove([idle.Id, busy.Id]);

        Assert.Equal(["idle"], removed);
        Assert.Single(list.Jobs);
        Assert.Equal("busy", list.Jobs[0].Id);
    }

    // ------------------------------------------------------------------------------------------
    // Notification
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void A_refused_change_raises_nothing()
    {
        var list = WithOne(out var job);
        list.Complete(job.Id, new EngineOutcome { Status = JobStatus.Cancelled });

        var raised = 0;
        list.Changed += _ => raised++;

        list.Report(job.Id, JobStatus.Processing);
        list.Report(new EngineProgress { JobId = job.Id, Stage = "restoring", Fraction = 0.5 });

        Assert.Equal(0, raised);
    }

    [Fact]
    public void The_change_event_carries_a_snapshot_that_later_edits_do_not_touch()
    {
        var list = new JobList();
        IReadOnlyList<Job>? seen = null;
        list.Changed += jobs => seen = jobs;

        list.Add(NewJob("first"));
        var afterFirst = seen;

        list.Add(NewJob("second"));

        Assert.Single(afterFirst!);
        Assert.Equal(2, seen!.Count);
    }
}
