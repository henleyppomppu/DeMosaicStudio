using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Threading;
using DeMosaicStudio.Application.Diagnostics;
using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;
using DeMosaicStudio.Application.Settings;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.App.ViewModels;

/// <summary>
/// What the window shows and what its buttons do.
/// <para>
/// The rules are not here. Which jobs may be cancelled, what a late progress message does, whether
/// a status may move, which dropped paths become jobs — all of that is <see cref="JobList"/> and
/// <see cref="VideoFiles"/>, which are testable without a window. This owns the two things a view
/// model must: the ordering guarantee, and the display strings.
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
    private readonly ISettingsStore _store;

    private CancellationTokenSource? _running;
    private string _status = "아직 시작하지 않았습니다";
    private bool _isBusy;
    private bool _isStopping;
    private int _nextId;

    /// <summary>Creates the view model around an engine and a place to keep settings.</summary>
    public MainViewModel(IRestorationEngine engine, Dispatcher dispatcher, ISettingsStore store)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
        _store = store ?? throw new ArgumentNullException(nameof(store));

        Settings = _store.Load();
        _jobs.Changed += OnJobsChanged;
    }

    /// <inheritdoc />
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>The jobs, for the list view.</summary>
    public ObservableCollection<JobRow> Rows { get; } = [];

    /// <summary>The settings new jobs are queued with.</summary>
    /// <remarks>
    /// Applied when a job is added, not when it starts: a job already in the queue keeps the
    /// settings it was queued with, so changing them mid-run cannot half-apply to work under way.
    /// </remarks>
    public JobSettings Settings { get; private set; }

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
                Raise(nameof(CanStop));
            }
        }
    }

    /// <summary>True when there is something to start and nothing running.</summary>
    public bool CanStart => !IsBusy && _jobs.Runnable.Count > 0;

    /// <summary>True while something is running that has not already been asked to stop.</summary>
    public bool CanStop => IsBusy && !_isStopping;

    /// <summary>Replaces the settings and persists them.</summary>
    /// <remarks>Clamped here as well as in the dialog: this is the only path in, so it is the one to trust.</remarks>
    public void Apply(JobSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        Settings = SettingsBounds.Clamp(settings);
        Raise(nameof(Settings));

        if (!_store.Save(Settings))
        {
            Status = "설정을 이번 실행에는 적용했지만 다음 실행을 위해 저장하지는 못했습니다";
        }
    }

    /// <summary>Starts the engine and reports what it found.</summary>
    public async Task InitialiseAsync()
    {
        try
        {
            var capabilities = await _engine.StartAsync().ConfigureAwait(true);
            Status = capabilities.CudaAvailable
                ? $"엔진 {capabilities.Version} · {capabilities.Device}"
                : $"엔진 {capabilities.Version} · CPU — 복원이 매우 느립니다";
        }
        catch (Exception exception) when (exception is InvalidOperationException or IOException)
        {
            Status = $"엔진을 시작하지 못했습니다: {exception.Message}";
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
            Settings = Settings,
        });

        Raise(nameof(CanStart));
    }

    /// <summary>
    /// Queues everything the given paths stand for: the videos among them, and the videos inside
    /// any folder among them. Returns how many were queued.
    /// </summary>
    public int AddRange(IEnumerable<string> paths)
    {
        var files = VideoFiles.Expand(paths);
        foreach (var file in files)
        {
            Add(file);
        }

        Status = files.Count switch
        {
            0 => "추가할 것이 없습니다 — 놓은 것 중에 영상 파일이 없습니다",
            1 => $"{Path.GetFileName(files[0])} 을(를) 대기열에 넣었습니다",
            _ => $"영상 {files.Count}개를 대기열에 넣었습니다",
        };

        return files.Count;
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
        _isStopping = false;
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
            _isStopping = false;
            _running.Dispose();
            _running = null;
            Raise(nameof(CanStart));
        }
    }

    /// <summary>Asks the engine to stop. Cooperative: the worker still writes its checkpoint.</summary>
    /// <remarks>
    /// The running job is not marked here. It ends by its own terminal result, which is what leaves
    /// the checkpoint intact — so the status line says what is happening rather than leaving the
    /// button looking as though it did nothing while the worker finishes the frame it is on.
    /// </remarks>
    public void Stop()
    {
        if (!IsBusy)
        {
            return;
        }

        _isStopping = true;
        Raise(nameof(CanStop));

        _running?.Cancel();

        foreach (var pending in _jobs.Runnable)
        {
            _jobs.Report(pending.Id, JobStatus.Cancelled, "시작하기 전에 취소되었습니다");
        }

        Status = _jobs.Active is { } active
            ? $"{Path.GetFileName(active.SourcePath)} 중지 중 — 체크포인트를 쓸 때까지 기다립니다"
            : "중지 중";
    }

    private async Task RunAsync(Job job, CancellationToken cancellationToken)
    {
        _jobs.Report(job.Id, JobStatus.Probing, "시작하는 중");

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
                Message = "취소되었습니다",
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
                ? ErrorText.Line(error)
                : outcome.Message ?? string.Empty;
        }

        if (summary.Passthrough)
        {
            return "찾은 것이 없어 다시 인코딩하지 않고 그대로 복사했습니다";
        }

        return $"{summary.FramesRestored}개 프레임에서 영역 {summary.RegionsDetected}개 — "
             + "복원된 영역은 추정이지 되찾은 원본이 아닙니다";
    }

    /// <summary>Reconciles the rows against the list, in place.</summary>
    /// <remarks>
    /// <b>In place, not cleared and rebuilt.</b> Rebuilding drops the list view's selection, and
    /// progress now arrives several times a second — so a user who selected a job in order to
    /// remove or retry it would have the selection taken away under their hand before they could
    /// press anything. Rows are matched by id and updated.
    /// </remarks>
    private void OnJobsChanged(IReadOnlyList<Job> jobs)
    {
        // The engine reports from a thread pool thread; the collection is bound to a view.
        if (!_dispatcher.CheckAccess())
        {
            _dispatcher.Invoke(() => OnJobsChanged(jobs));
            return;
        }

        var wanted = jobs.Select(job => job.Id).ToHashSet(StringComparer.Ordinal);
        for (var index = Rows.Count - 1; index >= 0; index--)
        {
            if (!wanted.Contains(Rows[index].Id))
            {
                Rows.RemoveAt(index);
            }
        }

        var byId = Rows.ToDictionary(row => row.Id, StringComparer.Ordinal);
        for (var index = 0; index < jobs.Count; index++)
        {
            var job = jobs[index];
            if (!byId.TryGetValue(job.Id, out var existing))
            {
                Rows.Insert(index, JobRow.From(job));
                continue;
            }

            existing.Update(job);

            // The list's own order is the queue's order, and a retry appends, so a row can move.
            var at = Rows.IndexOf(existing);
            if (at != index)
            {
                Rows.Move(at, index);
            }
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
/// Formatting here rather than in XAML converters keeps it testable and keeps the view dumb. The
/// row is mutable and raises change notifications because the alternative — replacing it — takes
/// the user's selection away every time progress moves.
/// </remarks>
public sealed class JobRow : INotifyPropertyChanged
{
    private string _status = string.Empty;
    private string _progress = string.Empty;
    private string _detail = string.Empty;
    private bool _canRetry;

    private JobRow(string id, string name)
    {
        Id = id;
        Name = name;
    }

    /// <inheritdoc />
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>The job's id, for commands.</summary>
    public string Id { get; }

    /// <summary>The file name alone. §2.3 C-6 keeps full paths out of logs; this is the display.</summary>
    public string Name { get; }

    /// <summary>The status, in words.</summary>
    public string Status
    {
        get => _status;
        private set => Set(ref _status, value);
    }

    /// <summary>Percent complete, or an empty string before anything is known.</summary>
    public string Progress
    {
        get => _progress;
        private set => Set(ref _progress, value);
    }

    /// <summary>What happened, in one line.</summary>
    public string Detail
    {
        get => _detail;
        private set => Set(ref _detail, value);
    }

    /// <summary>Whether a retry would do anything.</summary>
    public bool CanRetry
    {
        get => _canRetry;
        private set => Set(ref _canRetry, value);
    }

    /// <summary>Builds a row from a job.</summary>
    public static JobRow From(Job job)
    {
        ArgumentNullException.ThrowIfNull(job);

        var row = new JobRow(job.Id, Path.GetFileName(job.SourcePath));
        row.Update(job);
        return row;
    }

    /// <summary>Brings the row up to date with the job it stands for.</summary>
    public void Update(Job job)
    {
        ArgumentNullException.ThrowIfNull(job);

        Status = job.Status switch
        {
            JobStatus.Pending => "대기 중",
            JobStatus.Probing => "확인 중",
            JobStatus.Analyzing => "분석 중",
            JobStatus.Processing => "복원 중",
            JobStatus.Completed => "완료",
            JobStatus.Cancelled => "취소됨",
            _ => "실패",
        };
        // One decimal below 10%. The whole-percent format was hiding the difference between a job
        // that has not started and one that is 0.25% through a sixty-seven-hour run.
        Progress = job.Fraction is { } fraction
            ? (fraction < 0.10 ? $"{fraction:P1}" : $"{fraction:P0}")
            : string.Empty;
        Detail = Describe(job);
        CanRetry = job.IsTerminal;
    }

    /// <summary>
    /// The detail line: what happened for a finished job, how fast it is going for a running one.
    /// </summary>
    /// <remarks>
    /// A running job shows its rate and its estimate rather than only a percentage, because a
    /// percentage cannot answer the question a user asks ten minutes in. At the measured throughput
    /// — 0.45 frames a second at 1080p — an hour-long source is under half a percent for its first
    /// ten minutes, which rounds to zero and is indistinguishable from a hang.
    /// </remarks>
    private static string Describe(Job job)
    {
        if (!job.IsActive || job.Fps is not { } fps)
        {
            return job.Message ?? string.Empty;
        }

        // A rate with no estimate means the container never said how long the source is — in which
        // case the percentage beside it is a zero that will never move, and saying so is the point.
        return job.EtaSeconds is { } eta
            ? $"{fps:0.##} fps · 남은 시간 {Humanise(TimeSpan.FromSeconds(eta))}"
            : $"{fps:0.##} fps · 원본 길이를 알 수 없어 진행률을 계산할 수 없습니다";
    }

    /// <summary>A duration in the units a person would use for it.</summary>
    private static string Humanise(TimeSpan span) => span switch
    {
        { TotalDays: >= 1 } => $"약 {span.TotalDays:0.#}일",
        { TotalHours: >= 1 } => $"약 {span.TotalHours:0.#}시간",
        { TotalMinutes: >= 1 } => $"약 {span.TotalMinutes:0}분",
        _ => "1분 미만",
    };

    private void Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
