using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;

namespace DeMosaicStudio.Application.Tests;

/// <summary>
/// The rate and the estimate reach the job, so the window can say more than a percentage.
/// </summary>
/// <remarks>
/// <para>
/// The defect this closes was not a crash. <c>EngineProgress</c> already carried <c>Fps</c> and the
/// queue threw it away, so the only thing a running job could show was a rounded percentage — and
/// at this pipeline's measured throughput, 0.45 frames a second at 1080p, an hour-long source sits
/// under half a percent for its first ten minutes. It rounds to zero, and zero for ten minutes is
/// indistinguishable from a hang. It was reported as one.
/// </para>
/// </remarks>
public sealed class ProgressRateTests
{
    private static JobList WithRunningJob()
    {
        var jobs = new JobList();
        jobs.Add(new Job { Id = "j", SourcePath = "a.mp4", OutputPath = "b.mp4" });
        return jobs;
    }

    private static EngineProgress Report(double fraction, double? fps, double? eta) => new()
    {
        JobId = "j",
        Stage = "restoring",
        Fraction = fraction,
        Fps = fps,
        EtaSeconds = eta,
    };

    [Fact]
    public void The_rate_and_the_estimate_reach_the_job()
    {
        var jobs = WithRunningJob();

        Assert.True(jobs.Report(Report(0.0025, 0.45, 239_000)));

        var job = jobs.Find("j")!;
        Assert.Equal(0.45, job.Fps);
        Assert.Equal(239_000, job.EtaSeconds);
    }

    [Fact]
    public void A_rate_with_no_estimate_survives_as_exactly_that()
    {
        // The engine says this when the container never reported a duration. The fraction beside it
        // is then a zero that will never move, and the window has to be able to tell the user so
        // rather than showing the zero.
        var jobs = WithRunningJob();
        jobs.Report(Report(0.0, 0.45, null));

        var job = jobs.Find("j")!;
        Assert.Equal(0.45, job.Fps);
        Assert.Null(job.EtaSeconds);
    }

    [Fact]
    public void A_report_with_no_rate_clears_the_old_one_rather_than_leaving_it_stale()
    {
        // A stale "0.45 fps" beside a frozen percentage is the reassurance this exists to avoid.
        var jobs = WithRunningJob();
        jobs.Report(Report(0.01, 0.45, 1000));
        jobs.Report(Report(0.02, null, null));

        var job = jobs.Find("j")!;
        Assert.Null(job.Fps);
        Assert.Null(job.EtaSeconds);
    }

    [Fact]
    public void A_report_that_moves_the_job_backwards_carries_no_rate_with_it()
    {
        var jobs = WithRunningJob();
        jobs.Report(Report(0.50, 0.45, 100));

        Assert.False(jobs.Report(Report(0.10, 99.0, 1)));

        var job = jobs.Find("j")!;
        Assert.Equal(0.50, job.Fraction);
        Assert.Equal(0.45, job.Fps);
    }

    [Fact]
    public void A_run_of_reports_in_the_same_stage_all_land()
    {
        // **The test that was missing.** The worker was emitting progress correctly and the codec
        // was parsing it correctly, and each side had its own passing test. Nothing exercised the
        // sequence across the seam — and the seam refused every report after the first, because
        // `IsAllowed` compared ranks strictly and a job already Processing cannot "move to"
        // Processing. The window showed 복원 중 / 0.0% / 시작하는 중 for the life of the job.
        var jobs = WithRunningJob();
        jobs.Report("j", JobStatus.Probing, "시작하는 중");

        Assert.True(jobs.Report(new EngineProgress
        {
            JobId = "j", Stage = "probing", Fraction = 0.0,
        }));
        Assert.True(jobs.Report(Report(0.0, null, null)));            // the forced restoring 0.0
        Assert.True(jobs.Report(Report(0.0001, 0.44, 240_000)));      // the first real frame
        Assert.True(jobs.Report(Report(0.0002, 0.45, 239_000)));      // and the next

        var job = jobs.Find("j")!;
        Assert.Equal(JobStatus.Processing, job.Status);
        Assert.Equal(0.0002, job.Fraction);
        Assert.Equal(0.45, job.Fps);
        Assert.Equal(239_000, job.EtaSeconds);
    }

    [Theory]
    [InlineData(JobStatus.Probing)]
    [InlineData(JobStatus.Analyzing)]
    [InlineData(JobStatus.Processing)]
    public void Staying_in_an_active_stage_is_allowed(JobStatus stage) =>
        Assert.True(JobStatusTransition.IsAllowed(stage, stage));

    [Fact]
    public void Staying_still_does_not_make_a_terminal_state_leavable()
    {
        // The rule that had to survive the fix: a terminal state is terminal, including against
        // itself. A second `result` for a finished job must not re-open it.
        Assert.False(JobStatusTransition.IsAllowed(JobStatus.Completed, JobStatus.Completed));
        Assert.False(JobStatusTransition.IsAllowed(JobStatus.Cancelled, JobStatus.Cancelled));
        Assert.False(JobStatusTransition.IsAllowed(JobStatus.Failed, JobStatus.Failed));
        Assert.False(JobStatusTransition.IsAllowed(JobStatus.Pending, JobStatus.Pending));
    }

    [Theory]
    // The measurement that prompted this: 0.45 fps at 1080p, against sources of each length.
    [InlineData(10, 30, 0.015)]
    [InlineData(30, 30, 0.005)]
    [InlineData(60, 30, 0.0025)]
    public void Ten_minutes_into_a_long_source_is_a_fraction_a_whole_percent_cannot_show(
        int sourceMinutes, int sourceFps, double expected)
    {
        var frames = sourceMinutes * 60 * sourceFps;
        var done = 600 * 0.45;
        var fraction = done / frames;

        Assert.Equal(expected, fraction, 4);

        // Which is the whole point: rounded to whole percent, two of these three are "0%".
        Assert.True(fraction < 0.02);
    }
}
