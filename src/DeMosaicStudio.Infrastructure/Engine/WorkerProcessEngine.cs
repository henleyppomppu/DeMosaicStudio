using System.Diagnostics;
using System.Globalization;
using System.Text;
using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;
using DeMosaicStudio.Domain.Diagnostics;
using DeMosaicStudio.Infrastructure.Protocol;

namespace DeMosaicStudio.Infrastructure.Engine;

/// <summary>Where the worker lives and how to start it.</summary>
public sealed record WorkerLocation
{
    /// <summary>The interpreter to run.</summary>
    public required string Interpreter { get; init; }

    /// <summary>The directory <c>demosaic_worker</c> is importable from.</summary>
    public required string WorkerRoot { get; init; }

    /// <summary>The working directory the worker runs in.</summary>
    public string? WorkingDirectory { get; init; }
}

/// <summary>
/// The engine, as a child process speaking JSON Lines. docs/WORKER_PROTOCOL.md.
/// <para>
/// <b>A client, not a shortcut.</b> Nothing here reaches into the worker's code: it writes requests
/// on stdin and reads replies from stdout, which is the boundary the desktop application depends on
/// and therefore the one worth exercising. `scripts/run_job.py` is the same client in Python and
/// exists for the same reason.
/// </para>
/// <para>
/// Infrastructure decides nothing (AGENTS.md). Every policy this class appears to apply — what a
/// stage means, whether a late message counts — belongs to <see cref="JobList"/>, and this only
/// hands it what arrived.
/// </para>
/// </summary>
public sealed class WorkerProcessEngine : IRestorationEngine, IAsyncDisposable, IDisposable
{
    private readonly WorkerLocation _location;
    private readonly string _hostVersion;
    private readonly SemaphoreSlim _oneAtATime = new(1, 1);

    private Process? _process;
    private StreamWriter? _stdin;
    private int _nextId;

    /// <summary>Creates an engine that will start the worker at <paramref name="location"/>.</summary>
    public WorkerProcessEngine(WorkerLocation location, string hostVersion = "1.0")
    {
        ArgumentNullException.ThrowIfNull(location);

        _location = location;
        _hostVersion = hostVersion;
    }

    /// <summary>Raised for every worker log line, so the host can show or record it.</summary>
    /// <remarks>§2.3 C-6: log lines carry no pixel data and no full source path at INFO.</remarks>
    public event Action<string, string>? Logged;

    /// <inheritdoc />
    public async Task<EngineCapabilities> StartAsync(CancellationToken cancellationToken = default)
    {
        if (_process is not null)
        {
            throw new InvalidOperationException("the worker is already running");
        }

        var info = new ProcessStartInfo
        {
            FileName = _location.Interpreter,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = _location.WorkingDirectory ?? Environment.CurrentDirectory,
            // The protocol is UTF-8 JSON Lines. The worker sets its own stdio encoding, and this is
            // the other half of that: a console code page on either side kills the stream on the
            // first non-ASCII log line, and it does so in the middle of a job rather than at start-up.
            StandardOutputEncoding = new UTF8Encoding(false),
            StandardErrorEncoding = new UTF8Encoding(false),
            StandardInputEncoding = new UTF8Encoding(false),
        };
        info.ArgumentList.Add("-m");
        info.ArgumentList.Add("demosaic_worker.main_loop");
        info.Environment["PYTHONPATH"] = _location.WorkerRoot;

        // A missing interpreter arrives as Win32Exception, which is the same failure as Start
        // returning null and deserves the same shape. Left as it is, it escapes the view model's
        // catch and - because start-up is an `async void` - ends the process with no window and no
        // message. The path is in the text because "could not start the worker" without it is
        // unactionable.
        try
        {
            _process = Process.Start(info)
                ?? throw new InvalidOperationException($"could not start {_location.Interpreter}");
        }
        catch (System.ComponentModel.Win32Exception exception)
        {
            throw new InvalidOperationException(
                $"could not start {_location.Interpreter}: {exception.Message}", exception);
        }

        _stdin = _process.StandardInput;

        // **stderr must be drained, always.** It is redirected so a traceback never reaches the
        // user's console, but a redirected pipe that nobody reads fills at 4 KB and the worker
        // blocks on its next write - mid-job, with nothing on the protocol channel to say why -
        // while this side waits for a `result` that will never come. x265 alone prints twenty
        // lines per encode. Lines are forwarded as log entries so a worker that dies with a
        // traceback still leaves it somewhere.
        _process.ErrorDataReceived += (_, args) =>
        {
            if (args.Data is { } line)
            {
                Logged?.Invoke("stderr", line);
            }
        };
        _process.BeginErrorReadLine();

        var ready = await ExchangeAsync(
            WorkerCodec.Hello(NextId(), _hostVersion),
            message => message.Type is "ready" or "error",
            cancellationToken).ConfigureAwait(false);

        if (ready.Type == "error")
        {
            var failure = WorkerCodec.Error(ready.Body);
            throw new InvalidOperationException(
                $"the worker refused the handshake: {failure.Error?.Code} {failure.Message}");
        }

        return WorkerCodec.Capabilities(ready.Body);
    }

    /// <inheritdoc />
    public async Task<MediaFacts> ProbeAsync(
        string sourcePath,
        CancellationToken cancellationToken = default)
    {
        await _oneAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var reply = await ExchangeAsync(
                WorkerCodec.Probe(NextId(), sourcePath),
                message => message.Type is "probeResult" or "error",
                cancellationToken).ConfigureAwait(false);

            if (reply.Type == "error")
            {
                var failure = WorkerCodec.Error(reply.Body);
                throw new InvalidOperationException(
                    $"probe failed: {failure.Error?.Code} {failure.Message}");
            }

            return WorkerCodec.Media(reply.Body);
        }
        finally
        {
            _oneAtATime.Release();
        }
    }

    /// <inheritdoc />
    public Task<EngineOutcome> AnalyzeAsync(
        EngineRequest request,
        IProgress<EngineProgress>? progress = null,
        CancellationToken cancellationToken = default) =>
        RunAsync(request, analyzeOnly: true, progress, cancellationToken);

    /// <inheritdoc />
    public Task<EngineOutcome> ProcessAsync(
        EngineRequest request,
        IProgress<EngineProgress>? progress = null,
        CancellationToken cancellationToken = default) =>
        RunAsync(request, analyzeOnly: false, progress, cancellationToken);

    private async Task<EngineOutcome> RunAsync(
        EngineRequest request,
        bool analyzeOnly,
        IProgress<EngineProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        await _oneAtATime.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await SendAsync(WorkerCodec.Job(NextId(), request, analyzeOnly)).ConfigureAwait(false);

            // Cancellation is cooperative: the worker is asked to stop and answers with a terminal
            // `result`. Killing the process instead would lose the checkpoint it is about to write.
            await using var cancelRegistration = cancellationToken.Register(
                () => _ = SendAsync(WorkerCodec.Cancel(NextId(), request.JobId)))
                .ConfigureAwait(false);

            while (await ReadAsync(CancellationToken.None).ConfigureAwait(false) is { } message)
            {
                switch (message.Type)
                {
                    case "progress" when message.JobId == request.JobId:
                        progress?.Report(WorkerCodec.Progress(message.Body, request.JobId));
                        break;

                    case "log":
                        Logged?.Invoke(
                            message.Body["level"]?.GetValue<string>() ?? "info",
                            message.Body["message"]?.GetValue<string>() ?? string.Empty);
                        break;

                    case "result" when message.JobId == request.JobId:
                        return WorkerCodec.Result(message.Body);

                    case "error":
                        return WorkerCodec.Error(message.Body);
                }
            }

            return new EngineOutcome
            {
                Status = JobStatus.Failed,
                Error = ErrorCodes.E7005,
                Message = "the worker stopped without reporting a result",
            };
        }
        finally
        {
            _oneAtATime.Release();
        }
    }

    /// <summary>Asks the worker to shut down politely, and waits briefly for it.</summary>
    /// <remarks>
    /// The polite path matters: a worker killed mid-job loses the checkpoint it was about to write,
    /// and the next attempt starts from nothing.
    /// </remarks>
    public async ValueTask DisposeAsync()
    {
        if (_process is null)
        {
            return;
        }

        try
        {
            await SendAsync(WorkerCodec.Shutdown(NextId())).ConfigureAwait(false);
            _stdin?.Close();

            using var grace = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            await _process.WaitForExitAsync(grace.Token).ConfigureAwait(false);
        }
        catch (Exception exception) when (exception is IOException or ObjectDisposedException
                                              or OperationCanceledException or InvalidOperationException)
        {
            // A worker that has already died cannot be asked to leave politely, and saying so is
            // not useful to anyone at this point.
        }
        finally
        {
            TearDown();
        }
    }

    /// <summary>Ends the worker without waiting.</summary>
    /// <remarks>
    /// Prefer <see cref="DisposeAsync"/>: this one skips the polite shutdown, so a job in progress
    /// loses whatever it had not yet checkpointed. It exists because a type holding a process and a
    /// semaphore has to be disposable synchronously too.
    /// </remarks>
    public void Dispose() => TearDown();

    private void TearDown()
    {
        if (_process is null)
        {
            return;
        }

        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
            }
        }
        catch (Exception exception) when (exception is InvalidOperationException or NotSupportedException)
        {
            // Already gone.
        }

        _process.Dispose();
        _process = null;
        _stdin = null;
        _oneAtATime.Dispose();
    }

    private async Task<WorkerMessage> ExchangeAsync(
        string request,
        Func<WorkerMessage, bool> wanted,
        CancellationToken cancellationToken)
    {
        await SendAsync(request).ConfigureAwait(false);

        while (await ReadAsync(cancellationToken).ConfigureAwait(false) is { } message)
        {
            if (message.Type == "log")
            {
                Logged?.Invoke(
                    message.Body["level"]?.GetValue<string>() ?? "info",
                    message.Body["message"]?.GetValue<string>() ?? string.Empty);
            }

            if (wanted(message))
            {
                return message;
            }
        }

        throw new InvalidOperationException("the worker stopped without replying");
    }

    private async Task SendAsync(string line)
    {
        if (_stdin is null)
        {
            throw new InvalidOperationException("the worker is not running");
        }

        await _stdin.WriteLineAsync(line).ConfigureAwait(false);
        await _stdin.FlushAsync().ConfigureAwait(false);
    }

    private async Task<WorkerMessage?> ReadAsync(CancellationToken cancellationToken)
    {
        if (_process is null)
        {
            return null;
        }

        while (true)
        {
            var line = await _process.StandardOutput.ReadLineAsync(cancellationToken)
                .ConfigureAwait(false);
            if (line is null)
            {
                return null;
            }

            // A line that does not parse is skipped rather than thrown on: a crashing interpreter
            // can put a traceback on this channel, and dying while reading it would report the
            // wrong failure.
            if (WorkerCodec.Read(line) is { } message)
            {
                return message;
            }
        }
    }

    private string NextId() =>
        Interlocked.Increment(ref _nextId).ToString(CultureInfo.InvariantCulture);
}
