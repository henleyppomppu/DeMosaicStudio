using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Threading;
using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;

namespace DeMosaicStudio.App.ViewModels;

/// <summary>
/// What the window shows and what its buttons do.
/// <para>
/// The rules are not here. Which jobs may be cancelled, what a late progress message does, whether
/// a status may move — all of that is <see cref="JobList"/>, which is testable without a window.
/// This owns the two things a view model must: the ordering guarantee, and the display strings.
/// </para>
/// <para>
/// <b>Everything that touches the list runs on the dispatcher.</b> Engine callbacks arrive on
/// thread-pool threads, and a queue mutated from two of them at once cannot be saved by any
/// transition rule. The marshalling is here rather than sprinkled through the engine because this
/// is the layer that has a dispatcher to marshal to.
/// </para>
/// </summary>
public sealed class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly JobList _jobs = new();
    private readonly IRestorationEngine _engine;
    private readonly Dispatcher _dispatcher;

    private CancellationTokenSource? _running;
    private string _status = "Not started";
    private bool _isBusy;
    private int _nextId;

    /// <summary>Creates the view model around an engine.</summary>
    public MainViewModel(IRestorationEngine engine, Dispatcher dispatcher)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));

        _jobs.Changed += OnJobsChanged;
    }

    /// <inheritdoc />
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>The jobs, for the list view.</summary>
    public ObservableCollection<JobRow> Rows { get; } = [];

    /// <summary>A line under the list: what the engine is, or what went wrong.</summary>
    public string Status
    {
        get => _status;
        private set => Set(ref _status, value);
    }

    /// <summary>True while the queue is running, so the buttons can say so.</summary>
    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (Set(ref _isBusy, value))
            {
                Raise(nameof(CanStart));
                Raise(nameof(CanCancel));
            }
        }
    }

    /// <summary>True when there is something to start and nothing running.</summary>
    public bool CanStart => !IsBusy && _jobs.Runnable.Count > 0;

    /// <summary>True while something is running.</summary>
    public bool CanCancel => IsBusy;

    /// <summary>Starts the engine and reports what it found.</summary>
    public async Task InitialiseAsync()
    {
        try
        {
            var capabilities = await _engine.StartAsync().ConfigureAwait(true);
            Status = capabilities.CudaAvailable
                ? $"Engine {capabilities.Version} on {capabilities.Device}"
                : $"Engine {capabilities.Version} on CPU — restoration will be slow";
        }
        catch (Exception exception) when (exception is InvalidOperationException or IOException)
        {
            Status = $"The engine did not start: {exception.Message}";
        }
    }

    /// <summary>Queues one file, choosing an output path beside it.</summary>
    public void Add(string sourcePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var directory = Path.GetDirectoryName(sourcePath) ?? string.Empty;
        var name = Path.GetFileNameWithoutExtension(sourcePath);
        var extension = Path.GetExtension(sourcePath);

        _jobs.Add(new Job
        {
            Id = $"job-{Interlocked.Increment(ref _nextId)}",
            SourcePath = sourcePath,
            OutputPath = Path.Combine(directory, $"{name}.restored{extension}"),
        });

        Raise(nameof(CanStart));
    }

    /// <summary>Removes the given jobs, except one the engine is working on.</summary>
    public void Remove(IEnumerable<string> ids)
    {
        _jobs.Remove(ids);
        Raise(nameof(CanStart));
    }

    /// <summary>Re-queues a finished job as a new attempt.</summary>
    public void Retry(string id)
    {
        _jobs.Retry(id, $"job-{Interlocked.Increment(ref _nextId)}");
        Raise(nameof(CanStart));
    }

    /// <summary>Runs every runnable job, oldest first, until one is cancelled or they run out.</summary>
    public async Task StartAsync()
    {
        if (IsBusy)
        {
            return;
        }

        _running = new CancellationTokenSource();
        IsBusy = true;

        try
        {
            // Re-read the runnable set each time: a job added while the queue runs joins it, and a
            // cancelled one drops out. Snapshotting once would process a list that no longer exists.
            while (!_running.IsCancellationRequested && _jobs.Runnable is [var job, ..])
            {
                await RunAsync(job, _running.Token).ConfigureAwait(true);
            }
        }
        finally
        {
            IsBusy = false;
            _running.Dispose();
            _running = null;
            Raise(nameof(CanStart));
        }
    }

    /// <summary>Asks the engine to stop. Cooperative: the worker still writes its checkpoint.</summary>
    public void Cancel()
    {
        // Cancelling the token stops the loop; the job the engine is on ends by its own terminal
        // result, which is what leaves the checkpoint intact.
        _running?.Cancel();

        foreach (var pending in _jobs.Runnable)
        {
            _jobs.Report(pending.Id, JobStatus.Cancelled, "Cancelled before it started");
        }
    }

    private async Task RunAsync(Job job, CancellationToken cancellationToken)
    {
        _jobs.Report(job.Id, JobStatus.Probing, "Starting");

        var progress = new Progress<EngineProgress>(
            report => _dispatcher.Invoke(() => _jobs.Report(report)));

        try
        {
            var outcome = await _engine.ProcessAsync(
                new EngineRequest
                {
                    JobId = job.Id,
                    SourcePath = job.SourcePath,
                    OutputPath = job.OutputPath,
                    Settings = job.Settings,
                },
                progress,
                cancellationToken).ConfigureAwait(true);

            _jobs.Complete(job.Id, outcome with { Message = Describe(outcome) });
        }
        catch (OperationCanceledException)
        {
            _jobs.Complete(job.Id, new EngineOutcome
            {
                Status = JobStatus.Cancelled,
                Message = "Cancelled",
            });
        }
        catch (Exception exception) when (exception is InvalidOperationException or IOException)
        {
            _jobs.Complete(job.Id, new EngineOutcome
            {
                Status = JobStatus.Failed,
                Message = exception.Message,
            });
        }
    }

    /// <summary>
    /// The line the user reads about a finished job. §1.3: an output containing estimated pixels
    /// says so, in the summary and here, rather than being described as recovered.
    /// </summary>
    private static string Describe(EngineOutcome outcome)
    {
        if (outcome.Status != JobStatus.Completed || outcome.Summary is not { } summary)
        {
            return outcome.Error is { } error
                ? $"{error.Code}: {error.Meaning}"
                : outcome.Message ?? string.Empty;
        }

        if (summary.Passthrough)
        {
            return "Nothing found — the video was copied, not re-encoded";
        }

        return $"{summary.RegionsDetected} regions over {summary.FramesRestored} frames — "
             + "restored areas are estimated, not recovered";
    }

    private void OnJobsChanged(IReadOnlyList<Job> jobs)
    {
        // The engine reports from a thread pool thread; the collection is bound to a view.
        if (!_dispatcher.CheckAccess())
        {
            _dispatcher.Invoke(() => OnJobsChanged(jobs));
            return;
        }

        Rows.Clear();
        foreach (var job in jobs)
        {
            Rows.Add(JobRow.From(job));
        }

        Raise(nameof(CanStart));
    }

    /// <summary>Drops the cancellation source the running queue owns.</summary>
    public void Dispose()
    {
        _running?.Dispose();
        _running = null;
    }

    private bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        Raise(name);
        return true;
    }

    private void Raise(string? name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>One row of the list, already turned into strings.</summary>
/// <remarks>
/// Formatting here rather than in XAML converters keeps it testable and keeps the view dumb.
/// </remarks>
public sealed record JobRow
{
    /// <summary>The job's id, for commands.</summary>
    public required string Id { get; init; }

    /// <summary>The file name alone. §2.3 C-6 keeps full paths out of logs; this is the display.</summary>
    public required string Name { get; init; }

    /// <summary>The status, in words.</summary>
    public required string Status { get; init; }

    /// <summary>Percent complete, or an empty string before anything is known.</summary>
    public required string Progress { get; init; }

    /// <summary>What happened, in one line.</summary>
    public required string Detail { get; init; }

    /// <summary>Whether a retry would do anything.</summary>
    public bool CanRetry { get; init; }

    /// <summary>Builds a row from a job.</summary>
    public static JobRow From(Job job)
    {
        ArgumentNullException.ThrowIfNull(job);

        return new JobRow
        {
            Id = job.Id,
            Name = Path.GetFileName(job.SourcePath),
            Status = job.Status switch
            {
                JobStatus.Pending => "Waiting",
                JobStatus.Probing => "Inspecting",
                JobStatus.Analyzing => "Analysing",
                JobStatus.Processing => "Restoring",
                JobStatus.Completed => "Done",
                JobStatus.Cancelled => "Cancelled",
                _ => "Failed",
            },
            Progress = job.Fraction is { } fraction ? $"{fraction:P0}" : string.Empty,
            Detail = job.Message ?? string.Empty,
            CanRetry = job.IsTerminal,
        };
    }
}
