using System.Text.Json;
using System.Text.Json.Nodes;
using DeMosaicStudio.Application.Engine;
using DeMosaicStudio.Application.Jobs;
using DeMosaicStudio.Domain.Diagnostics;
using DeMosaicStudio.Domain.Protocol;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Infrastructure.Protocol;

/// <summary>
/// The wire, and nothing else. docs/WORKER_PROTOCOL.md.
/// <para>
/// Pure: strings in, strings out. That is deliberate — the protocol has broken twice in ways a
/// process test would have caught slowly and a codec test catches instantly (<c>jobId</c> lifted
/// into the envelope but read from the payload, and stdout defaulting to a code page that killed
/// the stream on one non-ASCII log line).
/// </para>
/// <para>
/// <b>Unknown fields are ignored, never rejected.</b> That rule is what lets a newer worker add
/// fields without invalidating an older host, and it is enforced here by reading only what is
/// asked for.
/// </para>
/// </summary>
public static class WorkerCodec
{
    private static readonly JsonSerializerOptions Options = new()
    {
        // The protocol is UTF-8 JSON Lines; escaping non-ASCII would be legal and unreadable.
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>Builds the handshake.</summary>
    public static string Hello(string id, string hostVersion)
    {
        var node = Envelope("hello", id);
        node["hostVersion"] = hostVersion;
        node["protocolVersion"] = ProtocolVersion.Current.ToString();
        return Write(node);
    }

    /// <summary>Builds a probe request.</summary>
    public static string Probe(string id, string sourcePath)
    {
        var node = Envelope("probe", id);
        node["sourcePath"] = sourcePath;
        return Write(node);
    }

    /// <summary>Builds an <c>analyze</c> or <c>process</c> request.</summary>
    public static string Job(string id, EngineRequest request, bool analyzeOnly)
    {
        var node = Envelope(analyzeOnly ? "analyze" : "process", id);
        node["jobId"] = request.JobId;
        node["sourcePath"] = request.SourcePath;
        node["settings"] = SettingsNode(request.Settings);

        if (analyzeOnly)
        {
            node["sampleEvery"] = Math.Max(1, request.SampleEvery);
        }
        else
        {
            node["outputPath"] = request.OutputPath;
            node["resume"] = request.Resume;
        }

        return Write(node);
    }

    /// <summary>Builds a cancel.</summary>
    public static string Cancel(string id, string jobId)
    {
        var node = Envelope("cancel", id);
        node["jobId"] = jobId;
        return Write(node);
    }

    /// <summary>Builds a shutdown.</summary>
    public static string Shutdown(string id) => Write(Envelope("shutdown", id));

    /// <summary>Reads one line. Returns null for a blank line or one that is not an object.</summary>
    /// <remarks>
    /// A malformed line is not an exception here. The worker's own stdout is the only thing on this
    /// channel, but a crashing interpreter can put a traceback on it, and a host that dies reading
    /// one reports the wrong failure.
    /// </remarks>
    public static WorkerMessage? Read(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }

        JsonNode? node;
        try
        {
            node = JsonNode.Parse(line);
        }
        catch (JsonException)
        {
            return null;
        }

        if (node is not JsonObject envelope || envelope["type"]?.GetValue<string>() is not { } type)
        {
            return null;
        }

        return new WorkerMessage(type, envelope["jobId"]?.GetValue<string>(), envelope);
    }

    /// <summary>Reads a <c>ready</c> into capabilities.</summary>
    public static EngineCapabilities Capabilities(JsonObject message)
    {
        var capabilities = message["capabilities"] as JsonObject;
        var models = (capabilities?["models"] as JsonArray ?? [])
            .OfType<JsonObject>()
            .Select(model => $"{model["id"]?.GetValue<string>()}/{model["version"]?.GetValue<string>()}")
            .ToList();

        return new EngineCapabilities
        {
            Version = message["workerVersion"]?.GetValue<string>() ?? string.Empty,
            ProtocolVersion = message["protocolVersion"]?.GetValue<string>() ?? string.Empty,
            CudaAvailable = capabilities?["cudaAvailable"]?.GetValue<bool>() ?? false,
            Device = capabilities?["device"]?.GetValue<string>() ?? "cpu",
            Models = models,
        };
    }

    /// <summary>Reads a <c>probeResult</c> into media facts.</summary>
    public static MediaFacts Media(JsonObject message)
    {
        var media = message["media"] as JsonObject ?? [];

        return new MediaFacts
        {
            Width = Int(media, "width"),
            Height = Int(media, "height"),
            DurationSeconds = Double(media, "durationSeconds") ?? 0.0,
            VideoCodec = media["videoCodec"]?.GetValue<string>() ?? string.Empty,
            IsVariableFrameRate = media["isVfr"]?.GetValue<bool>() ?? false,
            AudioStreams = (media["audioStreams"] as JsonArray)?.Count ?? 0,
            SubtitleStreams = (media["subtitleStreams"] as JsonArray)?.Count ?? 0,
        };
    }

    /// <summary>Reads a <c>progress</c>.</summary>
    public static EngineProgress Progress(JsonObject message, string jobId) => new()
    {
        JobId = jobId,
        Stage = message["stage"]?.GetValue<string>() ?? string.Empty,
        Fraction = Double(message, "fraction"),
        Fps = Double(message, "fps"),
        EtaSeconds = Double(message, "eta"),
    };

    /// <summary>Reads a terminal <c>result</c>.</summary>
    public static EngineOutcome Result(JsonObject message)
    {
        var status = message["status"]?.GetValue<string>() switch
        {
            "completed" => JobStatus.Completed,
            "cancelled" => JobStatus.Cancelled,
            _ => JobStatus.Failed,
        };

        return new EngineOutcome
        {
            Status = status,
            Summary = Summary(message["summary"] as JsonObject),
            Error = Code(message["error"] as JsonObject),
        };
    }

    /// <summary>Reads an <c>error</c>.</summary>
    public static EngineOutcome Error(JsonObject message) => new()
    {
        Status = JobStatus.Failed,
        Error = Code(message),
        Message = message["message"]?.GetValue<string>(),
    };

    private static JobSummary? Summary(JsonObject? summary)
    {
        if (summary is null)
        {
            return null;
        }

        var reasons = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var pair in summary["routeReasons"] as JsonObject ?? [])
        {
            if (pair.Value?.GetValue<int>() is { } count)
            {
                reasons[pair.Key] = count;
            }
        }

        return new JobSummary
        {
            FramesSeen = Int(summary, "framesSeen"),
            FramesRestored = Int(summary, "framesRestored"),
            RegionsDetected = Int(summary, "regionsDetected"),
            RegionsGated = Int(summary, "regionsGated"),
            RouteReasons = reasons,
            Passthrough = summary["passthrough"]?.GetValue<bool>() ?? false,
            Synthetic = summary["synthetic"]?.GetValue<bool>() ?? false,
            Timeline = summary["timeline"]?.GetValue<string>(),
        };
    }

    private static ErrorCode? Code(JsonObject? node)
    {
        var text = node?["code"]?.GetValue<string>();
        if (string.IsNullOrEmpty(text))
        {
            return null;
        }

        // An unknown code is not a parse failure: a newer worker may have codes this host has never
        // heard of, and losing the whole result over one of them would be worse than losing the code.
        return ErrorCodes.All.FirstOrDefault(code => code.Code == text);
    }

    private static JsonObject SettingsNode(JobSettings settings) => new()
    {
        ["detection"] = new JsonObject
        {
            ["confidence"] = settings.Detection.Confidence,
            ["maskThreshold"] = settings.Detection.MaskThreshold,
            ["minRegionArea"] = settings.Detection.MinRegionArea,
            ["minConfirmFrames"] = settings.Detection.MinConfirmFrames,
            ["maxMissingFrames"] = settings.Detection.MaxMissingFrames,
            ["detectEvery"] = settings.Detection.DetectEvery,
        },
        ["restoration"] = new JsonObject
        {
            ["preset"] = settings.Restoration.Preset.ToString(),
            ["temporalWindow"] = settings.Restoration.TemporalWindow.IsAuto
                ? "auto"
                : settings.Restoration.TemporalWindow.FixedValue!.Value,
            ["alignConfMin"] = settings.Restoration.AlignConfMin,
            ["minRestorationConfidence"] = settings.Restoration.MinRestorationConfidence,
            ["featherWidth"] = settings.Restoration.FeatherWidth,
            ["temporalAlpha"] = settings.Restoration.TemporalAlpha,
            ["refine"] = new JsonObject
            {
                ["enabled"] = settings.Restoration.Refine.Enabled,
                ["strength"] = settings.Restoration.Refine.Strength,
                ["model"] = settings.Restoration.Refine.Model,
                ["lora"] = string.IsNullOrEmpty(settings.Restoration.Refine.Lora) ? null : settings.Restoration.Refine.Lora,
                ["embeddings"] = new JsonArray(
                    settings.Restoration.Refine.Embeddings.Select(e => (JsonNode?)JsonValue.Create(e)).ToArray()),
                ["negativeEmbeddings"] = new JsonArray(
                    settings.Restoration.Refine.NegativeEmbeddings.Select(e => (JsonNode?)JsonValue.Create(e)).ToArray()),
                ["steps"] = settings.Restoration.Refine.Steps,
                ["seed"] = settings.Restoration.Refine.Seed,
                ["storeRoot"] = string.IsNullOrWhiteSpace(settings.Restoration.Refine.StoreRoot)
                    ? null
                    : settings.Restoration.Refine.StoreRoot,
            },
        },
        ["encode"] = new JsonObject
        {
            ["profile"] = settings.Encode.Profile.ToString(),
            ["codec"] = settings.Encode.Codec.ToString(),
            ["constantQuality"] = settings.Encode.ConstantQuality,
        },
    };

    private static JsonObject Envelope(string type, string id) => new()
    {
        ["v"] = ProtocolVersion.Current.ToString(),
        ["type"] = type,
        ["id"] = id,
    };

    private static string Write(JsonObject node) => node.ToJsonString(Options);

    private static int Int(JsonObject node, string key) => node[key]?.GetValue<int>() ?? 0;

    private static double? Double(JsonObject node, string key) => node[key]?.GetValue<double>();
}

/// <summary>One decoded line: what it is, whose it is, and the object it came from.</summary>
/// <remarks>
/// The raw object travels with it so a reader can take fields this type has never heard of, which
/// is the protocol's forward-compatibility rule expressed in a type.
/// </remarks>
public sealed record WorkerMessage(string Type, string? JobId, JsonObject Body);
