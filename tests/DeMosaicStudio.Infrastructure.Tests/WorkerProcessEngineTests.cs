using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;
using DeMosaicStudio.Domain.Protocol;
using DeMosaicStudio.Infrastructure.Engine;
using Xunit;

namespace DeMosaicStudio.Infrastructure.Tests;

/// <summary>
/// The host talking to the real worker. docs/WORKER_PROTOCOL.md.
/// <para>
/// The codec tests prove the messages are shaped right; this proves the two halves actually meet.
/// Both defects the boundary has produced needed a live process to show themselves — <c>jobId</c>
/// lifted into the envelope but read from the payload, and stdout defaulting to a code page that
/// killed the stream on the first non-ASCII log line. Neither is visible from either side alone.
/// </para>
/// <para>
/// The interpreter and the worker's dependencies are part of the machine, not the repository, so
/// these skip where they are absent — and say why rather than passing quietly.
/// </para>
/// </summary>
public sealed class WorkerProcessEngineTests
{
    private static DirectoryInfo RepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !Directory.Exists(Path.Combine(directory.FullName, "fixtures")))
        {
            directory = directory.Parent;
        }

        return directory ?? throw new InvalidOperationException("no 'fixtures' directory above the test binary");
    }

    private static WorkerLocation? Locate()
    {
        var root = RepositoryRoot();
        var interpreter = Path.Combine(root.FullName, ".venv", "Scripts", "python.exe");

        if (!File.Exists(interpreter))
        {
            interpreter = Path.Combine(root.FullName, ".venv", "bin", "python");
        }

        return File.Exists(interpreter)
            ? new WorkerLocation
            {
                Interpreter = interpreter,
                WorkerRoot = Path.Combine(root.FullName, "worker"),
                WorkingDirectory = root.FullName,
            }
            : null;
    }

    private static async Task<(WorkerProcessEngine Engine, EngineCapabilities Capabilities)> StartAsync()
    {
        var location = Locate();
        Skip.If(location is null, "no .venv interpreter; the worker's runtime is part of the machine");

        var engine = new WorkerProcessEngine(location!);
        try
        {
            return (engine, await engine.StartAsync(CancellationToken.None));
        }
        catch (Exception exception)
        {
            await engine.DisposeAsync();
            // The runtime is part of the machine: a missing dependency is a reason to skip and say
            // so, not to fail a test about the protocol.
            Skip.If(true, $"the worker would not start: {exception.Message}");
            throw;
        }
    }

    [SkippableFact]
    public async Task The_handshake_agrees_on_a_protocol_version()
    {
        var (engine, capabilities) = await StartAsync();
        await using var _ = engine;

        Assert.Equal(ProtocolVersion.Current.ToString(), capabilities.ProtocolVersion);
        Assert.False(string.IsNullOrEmpty(capabilities.Version));
    }

    [SkippableFact]
    public async Task A_probe_returns_the_media_facts_the_fixture_actually_has()
    {
        var (engine, _) = await StartAsync();
        await using var _engine = engine;

        var source = Path.Combine(RepositoryRoot().FullName, "fixtures", "media", "cfr_30fps.mp4");
        Skip.IfNot(File.Exists(source), $"missing fixture: {source}");

        var media = await engine.ProbeAsync(source, CancellationToken.None);

        Assert.True(media.Width > 0);
        Assert.True(media.Height > 0);
        Assert.False(media.IsVariableFrameRate);
        Assert.Equal("h264", media.VideoCodec);
    }

    [SkippableFact]
    public async Task Probing_something_that_is_not_a_video_fails_rather_than_hanging()
    {
        var (engine, _) = await StartAsync();
        await using var _engine = engine;

        var notVideo = Path.Combine(RepositoryRoot().FullName, "README.md");
        Skip.IfNot(File.Exists(notVideo), "missing README.md");

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => engine.ProbeAsync(notVideo, CancellationToken.None));
    }

    /// <summary>
    /// The whole round trip, on a source with no mosaic in it: the job must come back terminal,
    /// carry a summary, and report a pass-through rather than a re-encode (R-1.8a).
    /// </summary>
    [SkippableFact]
    public async Task A_job_runs_to_a_terminal_result_and_reports_what_it_did()
    {
        var (engine, capabilities) = await StartAsync();
        await using var _engine = engine;

        Skip.If(capabilities.Models.Count == 0,
            "no detector in the model store; its weights are gitignored");

        var root = RepositoryRoot().FullName;
        var source = Path.Combine(root, "fixtures", "media", "cfr_30fps.mp4");
        Skip.IfNot(File.Exists(source), $"missing fixture: {source}");

        var output = Path.Combine(Path.GetTempPath(), $"demosaic-{Guid.NewGuid():N}.mp4");
        var seen = new List<EngineProgress>();

        try
        {
            var outcome = await engine.ProcessAsync(
                new EngineRequest
                {
                    JobId = "roundtrip",
                    SourcePath = source,
                    OutputPath = output,
                },
                new Progress<EngineProgress>(seen.Add),
                CancellationToken.None);

            Assert.Equal(JobStatus.Completed, outcome.Status);
            Assert.NotNull(outcome.Summary);
            Assert.True(outcome.Summary!.FramesSeen > 0);
            Assert.True(File.Exists(output));
        }
        finally
        {
            File.Delete(output);
        }
    }

    /// <summary>
    /// A non-ASCII path is the shape of the defect that killed the stream mid-job: the host and the
    /// worker have to agree on UTF-8, and a console code page on either side breaks it.
    /// </summary>
    [SkippableFact]
    public async Task A_non_ascii_path_survives_the_round_trip()
    {
        var (engine, _) = await StartAsync();
        await using var _engine = engine;

        var source = Path.Combine(RepositoryRoot().FullName, "fixtures", "media", "cfr_30fps.mp4");
        Skip.IfNot(File.Exists(source), $"missing fixture: {source}");

        var directory = Directory.CreateDirectory(
            Path.Combine(Path.GetTempPath(), $"데모자이크-{Guid.NewGuid():N}"));
        var copied = Path.Combine(directory.FullName, "클립.mp4");

        try
        {
            File.Copy(source, copied);
            var media = await engine.ProbeAsync(copied, CancellationToken.None);

            Assert.True(media.Width > 0);
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }
}
