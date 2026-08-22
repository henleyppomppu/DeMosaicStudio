using System.Text.Json.Nodes;
using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;
using DeMosaicStudio.Domain.Protocol;
using DeMosaicStudio.Domain.Settings;
using DeMosaicStudio.Infrastructure.Protocol;

namespace DeMosaicStudio.Infrastructure.Tests;

/// <summary>
/// The wire. docs/WORKER_PROTOCOL.md.
/// <para>
/// The protocol has broken twice in ways only a codec test catches quickly: <c>jobId</c> was lifted
/// into the envelope and read from the payload, so every <c>process</c> was refused, and stdout
/// defaulted to a code page that killed the stream on the first non-ASCII log line. Both look like
/// nothing until a job runs.
/// </para>
/// </summary>
public sealed class WorkerCodecTests
{
    private static EngineRequest Request() => new()
    {
        JobId = "j1",
        SourcePath = @"C:\videos\clip.mp4",
        OutputPath = @"C:\videos\clip.restored.mp4",
    };

    private static JsonObject Parse(string line) =>
        JsonNode.Parse(line) as JsonObject ?? throw new InvalidOperationException("not an object");

    // ------------------------------------------------------------------------------------------
    // Requests
    // ------------------------------------------------------------------------------------------

    [Fact]
    public void Every_request_carries_the_envelope()
    {
        var lines = new[]
        {
            WorkerCodec.Hello("1", "test"),
            WorkerCodec.Probe("2", "clip.mp4"),
            WorkerCodec.Job("3", Request(), analyzeOnly: false),
            WorkerCodec.Cancel("4", "j1"),
            WorkerCodec.Shutdown("5"),
        };

        foreach (var line in lines)
        {
            var node = Parse(line);
            Assert.Equal(ProtocolVersion.Current.ToString(), node["v"]!.GetValue<string>());
            Assert.False(string.IsNullOrEmpty(node["type"]!.GetValue<string>()));
            Assert.False(string.IsNullOrEmpty(node["id"]!.GetValue<string>()));
        }
    }

    /// <summary>The defect that refused every <c>process</c>: the id belongs in the envelope.</summary>
    [Fact]
    public void A_job_request_puts_the_job_id_in_the_envelope()
    {
        var node = Parse(WorkerCodec.Job("3", Request(), analyzeOnly: false));

        Assert.Equal("j1", node["jobId"]!.GetValue<string>());
    }

    [Fact]
    public void A_process_request_carries_an_output_path_and_an_analysis_does_not()
    {
        var process = Parse(WorkerCodec.Job("3", Request(), analyzeOnly: false));
        var analyze = Parse(WorkerCodec.Job("4", Request() with { SampleEvery = 4 }, analyzeOnly: true));

        Assert.Equal("process", process["type"]!.GetValue<string>());
        Assert.Equal(@"C:\videos\clip.restored.mp4", process["outputPath"]!.GetValue<string>());

        // `analyze` writes nothing, so an output path would be a claim it cannot honour.
        Assert.Equal("analyze", analyze["type"]!.GetValue<string>());
        Assert.Null(analyze["outputPath"]);
        Assert.Equal(4, analyze["sampleEvery"]!.GetValue<int>());
    }

    [Fact]
    public void Sample_every_is_never_below_one()
    {
        var node = Parse(WorkerCodec.Job("3", Request() with { SampleEvery = 0 }, analyzeOnly: true));

        Assert.Equal(1, node["sampleEvery"]!.GetValue<int>());
    }

    [Fact]
    public void An_auto_window_travels_as_the_word_and_a_fixed_one_as_a_number()
    {
        var auto = Request();
        var fixedWindow = Request() with
        {
            Settings = new JobSettings
            {
                Restoration = new RestorationSettings { TemporalWindow = TemporalWindowSetting.Fixed(5) },
            },
        };

        var autoNode = Parse(WorkerCodec.Job("3", auto, analyzeOnly: false));
        var fixedNode = Parse(WorkerCodec.Job("4", fixedWindow, analyzeOnly: false));

        Assert.Equal("auto", autoNode["settings"]!["restoration"]!["temporalWindow"]!.GetValue<string>());
        Assert.Equal(5, fixedNode["settings"]!["restoration"]!["temporalWindow"]!.GetValue<int>());
    }

    /// <summary>The protocol is UTF-8; escaping would be legal and unreadable.</summary>
    [Fact]
    public void Non_ascii_paths_survive_unescaped()
    {
        var line = WorkerCodec.Probe("2", @"D:\영상\클립.mp4");

        Assert.Contains("영상", line, StringComparison.Ordinal);
        Assert.Equal(@"D:\영상\클립.mp4", Parse(line)["sourcePath"]!.GetValue<string>());
    }

    // ------------------------------------------------------------------------------------------
    // Replies
    // ------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not json at all")]
    [InlineData("[1, 2, 3]")]
    [InlineData("""{"no": "type"}""")]
    public void A_line_that_is_not_a_message_reads_as_nothing_rather_than_throwing(string line)
    {
        // A crashing interpreter can put a traceback on this channel. A host that dies reading one
        // reports the wrong failure.
        Assert.Null(WorkerCodec.Read(line));
    }

    [Fact]
    public void Unknown_fields_are_ignored_rather_than_rejected()
    {
        var message = WorkerCodec.Read(
            """
            {"v":"1.1","type":"progress","id":"9","jobId":"j1","stage":"restoring","fraction":0.5,"somethingNewer":{"a":1}}
            """);

        Assert.NotNull(message);
        var progress = WorkerCodec.Progress(message!.Body, "j1");
        Assert.Equal("restoring", progress.Stage);
        Assert.Equal(0.5, progress.Fraction);
    }

    [Fact]
    public void A_ready_reply_becomes_capabilities()
    {
        var message = WorkerCodec.Read(
            """
            {"v":"1.1","type":"ready","id":"1","workerVersion":"0.1.0","protocolVersion":"1.1","capabilities":{"cudaAvailable":true,"device":"RTX 3080 Ti","models":[{"id":"det-unet","version":"0.2.0"}]}}
            """);

        var capabilities = WorkerCodec.Capabilities(message!.Body);

        Assert.Equal("0.1.0", capabilities.Version);
        Assert.True(capabilities.CudaAvailable);
        Assert.Equal("RTX 3080 Ti", capabilities.Device);
        Assert.Equal(["det-unet/0.2.0"], capabilities.Models);
    }

    [Fact]
    public void A_probe_reply_becomes_media_facts()
    {
        var message = WorkerCodec.Read(
            """
            {"v":"1.1","type":"probeResult","id":"2","media":{"width":1920,"height":800,"durationSeconds":4.0,"videoCodec":"h264","isVfr":false,"audioStreams":[{"index":1}],"subtitleStreams":[]}}
            """);

        var media = WorkerCodec.Media(message!.Body);

        Assert.Equal(1920, media.Width);
        Assert.Equal(4.0, media.DurationSeconds);
        Assert.False(media.IsVariableFrameRate);
        Assert.Equal(1, media.AudioStreams);
        Assert.Equal(0, media.SubtitleStreams);
    }

    [Fact]
    public void A_result_becomes_an_outcome_with_its_summary()
    {
        var message = WorkerCodec.Read(
            """
            {"v":"1.1","type":"result","id":"3","jobId":"j1","status":"completed","summary":{"framesSeen":96,"framesRestored":95,"regionsDetected":177,"regionsGated":0,"routeReasons":{"SufficientTemporalEvidence":161},"passthrough":false,"synthetic":true,"timeline":"frames 96/96"}}
            """);

        var outcome = WorkerCodec.Result(message!.Body);

        Assert.Equal(JobStatus.Completed, outcome.Status);
        Assert.Equal(96, outcome.Summary!.FramesSeen);
        Assert.True(outcome.Summary.Synthetic);
        Assert.Equal(161, outcome.Summary.RouteReasons["SufficientTemporalEvidence"]);
        Assert.Equal("frames 96/96", outcome.Summary.Timeline);
    }

    [Theory]
    [InlineData("completed", JobStatus.Completed)]
    [InlineData("cancelled", JobStatus.Cancelled)]
    [InlineData("failed", JobStatus.Failed)]
    [InlineData("something the host has never heard of", JobStatus.Failed)]
    public void An_unrecognised_status_reads_as_failed(string status, JobStatus expected)
    {
        var message = WorkerCodec.Read(
            $$"""
            {"v":"1.1","type":"result","id":"3","jobId":"j1","status":"{{status}}"}
            """);

        Assert.Equal(expected, WorkerCodec.Result(message!.Body).Status);
    }

    [Fact]
    public void An_error_keeps_its_code()
    {
        var message = WorkerCodec.Read(
            """
            {"v":"1.1","type":"error","id":"4","jobId":"j1","code":"E4401","recoverable":false,"message":"GPU memory exhausted"}
            """);

        var outcome = WorkerCodec.Error(message!.Body);

        Assert.Equal(JobStatus.Failed, outcome.Status);
        Assert.Equal("E4401", outcome.Error!.Code);
        Assert.Equal("GPU memory exhausted", outcome.Message);
    }

    /// <summary>A newer worker's code must not cost the host the whole result.</summary>
    [Fact]
    public void An_unknown_error_code_loses_the_code_and_nothing_else()
    {
        var message = WorkerCodec.Read(
            """
            {"v":"1.1","type":"error","id":"4","code":"E9999","message":"from the future"}
            """);

        var outcome = WorkerCodec.Error(message!.Body);

        Assert.Null(outcome.Error);
        Assert.Equal("from the future", outcome.Message);
        Assert.Equal(JobStatus.Failed, outcome.Status);
    }
}
