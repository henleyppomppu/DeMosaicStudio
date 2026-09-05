using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Domain.Jobs;

/// <summary>Which artifact a fingerprint governs. prd.md §9.3.</summary>
public enum FingerprintScope
{
    /// <summary>Detection and tracking. A change here discards analysis and video.</summary>
    Detection,

    /// <summary>Restoration. A change here discards video only.</summary>
    Restoration,

    /// <summary>Encoding. A change here discards video only.</summary>
    Encode,
}

/// <summary>
/// Per-artifact settings fingerprints. prd.md §9.3.
/// <para>
/// The canonical form is a newline-joined, ordinally sorted list of <c>key=value</c> lines, hashed
/// with SHA-256 and rendered as <c>sha256:</c> plus lowercase hex. It is deliberately not JSON:
/// a hand-built canonical string is far easier to mirror byte-for-byte in
/// <c>worker/demosaic_worker/fingerprints.py</c>, and the two are locked together by the parity
/// fixture <c>fixtures/parity/fingerprints.json</c> (§13.4).
/// </para>
/// <para>
/// Doubles are formatted with exactly four decimals so that C# and Python agree without depending on
/// either language's shortest-round-trip formatting.
/// </para>
/// </summary>
public static class SettingsFingerprint
{
    /// <summary>Computes the fingerprint for one artifact scope.</summary>
    public static string Compute(JobSettings settings, FingerprintScope scope)
    {
        ArgumentNullException.ThrowIfNull(settings);

        var fields = scope switch
        {
            FingerprintScope.Detection => DetectionFields(settings.Detection),
            FingerprintScope.Restoration => RestorationFields(settings.Restoration),
            FingerprintScope.Encode => EncodeFields(settings.Encode),
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, "Unknown fingerprint scope."),
        };

        return Hash(fields);
    }

    /// <summary>
    /// The canonical text that <see cref="Compute"/> hashes. Exposed so that a mismatch in the parity
    /// test can report <i>what</i> differs rather than only that two hashes differ.
    /// </summary>
    public static string Canonicalize(JobSettings settings, FingerprintScope scope)
    {
        ArgumentNullException.ThrowIfNull(settings);

        var fields = scope switch
        {
            FingerprintScope.Detection => DetectionFields(settings.Detection),
            FingerprintScope.Restoration => RestorationFields(settings.Restoration),
            FingerprintScope.Encode => EncodeFields(settings.Encode),
            _ => throw new ArgumentOutOfRangeException(nameof(scope), scope, "Unknown fingerprint scope."),
        };

        return CanonicalText(fields);
    }

    private static IEnumerable<KeyValuePair<string, string>> DetectionFields(DetectionSettings d) =>
    [
        new("confidence", Num(d.Confidence)),
        // How often the detector runs (D-43). Changes which frames carry detections, so it is
        // part of what the detection artifact depends on.
        new("detectEvery", Int(d.DetectEvery)),
        new("maskThreshold", Num(d.MaskThreshold)),
        new("maxMissingFrames", Int(d.MaxMissingFrames)),
        new("minConfirmFrames", Int(d.MinConfirmFrames)),
        new("minRegionArea", Int(d.MinRegionArea)),
        new("nmsIou", Num(d.NmsIou)),
    ];

    private static IEnumerable<KeyValuePair<string, string>> RestorationFields(RestorationSettings r) =>
    [
        new("alignConfMin", Num(r.AlignConfMin)),
        new("featherWidth", Int(r.FeatherWidth)),
        new("minRestorationConfidence", Num(r.MinRestorationConfidence)),
        new("paddingRatio", Num(r.PaddingRatio)),
        new("preset", r.Preset.ToString()),

        // The single-frame path's blend weight (D-43).
        new("temporalAlpha", Num(r.TemporalAlpha)),

        // The requested value, never the per-frame resolved K (prd.md §5.6.1, §9.3).
        new("temporalWindow", r.TemporalWindow.ToString()),
    ];

    private static IEnumerable<KeyValuePair<string, string>> EncodeFields(EncodeSettings e) =>
    [
        new("codec", e.Codec.ToString()),
        new("constantQuality", Int(e.ConstantQuality)),
        new("profile", e.Profile.ToString()),
    ];

    private static string CanonicalText(IEnumerable<KeyValuePair<string, string>> fields)
    {
        var lines = fields
            .Select(f => string.Concat(f.Key, "=", f.Value))
            .OrderBy(line => line, StringComparer.Ordinal);

        return string.Join('\n', lines);
    }

    private static string Hash(IEnumerable<KeyValuePair<string, string>> fields)
    {
        var bytes = Encoding.UTF8.GetBytes(CanonicalText(fields));
        var digest = SHA256.HashData(bytes);
        return string.Concat("sha256:", Convert.ToHexStringLower(digest));
    }

    private static string Num(double value) => value.ToString("0.0000", CultureInfo.InvariantCulture);

    private static string Int(int value) => value.ToString(CultureInfo.InvariantCulture);
}
