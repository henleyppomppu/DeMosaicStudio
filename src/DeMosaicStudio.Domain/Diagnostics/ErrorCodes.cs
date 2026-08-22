using System.Collections.Frozen;

namespace DeMosaicStudio.Domain.Diagnostics;

/// <summary>
/// The complete table from prd.md §10.2.
/// <para>
/// This is one of two implementations: the other is <c>worker/demosaic_worker/errors.py</c>.
/// They are locked together by the parity fixture <c>fixtures/parity/error_codes.json</c>
/// (§13.4). Adding a code means updating both sides, the fixture, docs/ERROR_CODES.md, and
/// docs/TROUBLESHOOTING.md — the parity test fails otherwise.
/// </para>
/// </summary>
public static class ErrorCodes
{
    // E1xxx — media / input
    public static readonly ErrorCode E1001 = Error("E1001", "File not found or unreadable", recoverable: false);
    public static readonly ErrorCode E1002 = Error("E1002", "Unsupported container", recoverable: false);
    public static readonly ErrorCode E1003 = Error("E1003", "Unsupported video codec or profile", recoverable: false);
    public static readonly ErrorCode E1004 = Error("E1004", "Corrupt source: demux failure", recoverable: false);
    public static readonly ErrorCode E1005 = Error("E1005", "Source has no video stream", recoverable: false);
    public static readonly ErrorCode E1006 = Error("E1006", "Source metadata inconsistent", recoverable: true);

    // E2xxx — decode
    public static readonly ErrorCode E2001 = Error("E2001", "Hardware decoder init failed", recoverable: true);
    public static readonly ErrorCode E2002 = Error("E2002", "Decode error mid-stream, frame unrecoverable", recoverable: true);
    public static readonly ErrorCode E2003 = Error("E2003", "Decode error mid-stream, stream unrecoverable", recoverable: false);
    public static readonly ErrorCode E2004 = Error("E2004", "Timestamp discontinuity beyond tolerance", recoverable: true);

    // E3xxx — detection / tracking
    public static readonly ErrorCode E3001 = Error("E3001", "Detector model load failed", recoverable: false);
    public static readonly ErrorCode E3002 = Error("E3002", "Detector inference failure", recoverable: true);
    public static readonly ErrorCode E3003 = Error("E3003", "Detector output shape mismatch", recoverable: false);
    public static readonly ErrorCode E3201 = Error("E3201", "Track state-machine violation", recoverable: false);

    // E4xxx — restoration
    public static readonly ErrorCode E4001 = Error("E4001", "Restoration model load failed", recoverable: false);
    public static readonly ErrorCode E4002 = Error("E4002", "Restoration inference failure", recoverable: true);
    public static readonly ErrorCode E4003 = Error("E4003", "Alignment failure for the whole window", recoverable: true);
    public static readonly ErrorCode E4004 = Error("E4004", "ROI smaller than model minimum", recoverable: true);
    public static readonly ErrorCode E4401 = Error("E4401", "GPU OOM, mitigation ladder exhausted", recoverable: false);
    public static readonly ErrorCode E4402 = Error("E4402", "Backend/runtime unsupported for this model", recoverable: false);

    // E5xxx — encode / mux
    public static readonly ErrorCode E5001 = Error("E5001", "Encoder init failed", recoverable: true);
    public static readonly ErrorCode E5002 = Error("E5002", "Encode failure mid-stream", recoverable: false);
    public static readonly ErrorCode E5003 = Error("E5003", "Mux failure", recoverable: false);
    public static readonly ErrorCode E5004 = Error("E5004", "Output container cannot carry a source stream", recoverable: true);

    // E6xxx — system
    public static readonly ErrorCode E6001 = Error("E6001", "Disk full", recoverable: true);
    public static readonly ErrorCode E6002 = Error("E6002", "Output path not writable", recoverable: true);
    public static readonly ErrorCode E6003 = Error("E6003", "Output file locked by another process", recoverable: true);
    public static readonly ErrorCode E6004 = Error("E6004", "Insufficient system RAM", recoverable: false);
    public static readonly ErrorCode E6005 = Error("E6005", "Required support library missing or unloadable", recoverable: false);

    // E7xxx — protocol / process
    public static readonly ErrorCode E7001 = Error("E7001", "Protocol major version mismatch", recoverable: false);
    public static readonly ErrorCode E7002 = Error("E7002", "Worker handshake timeout", recoverable: true);
    public static readonly ErrorCode E7003 = Error("E7003", "Worker busy: a job is already running", recoverable: false);
    public static readonly ErrorCode E7004 = Error("E7004", "Worker did not exit within the cancel grace period", recoverable: true);
    public static readonly ErrorCode E7005 = Error("E7005", "Worker crashed", recoverable: true);
    public static readonly ErrorCode E7006 = Error("E7006", "Malformed protocol message", recoverable: false);

    // E9xxx
    public static readonly ErrorCode E9001 = Error("E9001", "Unexpected internal error", recoverable: false);

    // Warnings
    public static readonly ErrorCode W1101 = Warning("W1101", "Fell back to software decode");
    public static readonly ErrorCode W3101 = Warning("W3101", "Region count clamped to max_regions_per_frame");
    public static readonly ErrorCode W4101 = Warning("W4101", "OOM ladder step applied");
    public static readonly ErrorCode W4102 = Warning("W4102", "Region left untouched: confidence below minRestorationConfidence");
    public static readonly ErrorCode W4103 = Warning("W4103", "Requested temporalWindow reduced by a safety rule");
    public static readonly ErrorCode W5101 = Warning("W5101", "Stream dropped for container compatibility");
    public static readonly ErrorCode W6101 = Warning("W6101", "Backend substituted");

    private static readonly ErrorCode[] AllCodes =
    [
        E1001, E1002, E1003, E1004, E1005, E1006,
        E2001, E2002, E2003, E2004,
        E3001, E3002, E3003, E3201,
        E4001, E4002, E4003, E4004, E4401, E4402,
        E5001, E5002, E5003, E5004,
        E6001, E6002, E6003, E6004, E6005,
        E7001, E7002, E7003, E7004, E7005, E7006,
        E9001,
        W1101, W3101, W4101, W4102, W4103, W5101, W6101,
    ];

    private static readonly FrozenDictionary<string, ErrorCode> ByCode =
        AllCodes.ToFrozenDictionary(c => c.Code, StringComparer.Ordinal);

    /// <summary>Every entry in the table, in declaration order.</summary>
    public static IReadOnlyList<ErrorCode> All => AllCodes;

    /// <summary>Looks up an entry by its identifier.</summary>
    public static bool TryGet(string code, out ErrorCode? errorCode) => ByCode.TryGetValue(code, out errorCode);

    /// <summary>Looks up an entry by its identifier, throwing when it is not in the table.</summary>
    public static ErrorCode Get(string code) =>
        TryGet(code, out var found) && found is not null
            ? found
            : throw new KeyNotFoundException($"'{code}' is not in the error table (prd.md §10.2).");

    private static ErrorCode Error(string code, string meaning, bool recoverable) =>
        new(code, meaning, recoverable, ErrorSeverity.Error);

    // Warnings are never failures, so 'recoverable' does not apply and is fixed at false (§10.1).
    private static ErrorCode Warning(string code, string meaning) =>
        new(code, meaning, Recoverable: false, ErrorSeverity.Warning);
}
