using System.Globalization;

namespace DeMosaicStudio.Domain.Protocol;

/// <summary>
/// The host/worker protocol version. prd.md §8.1.
/// <para>
/// This is the <b>mirror</b>. The single definition lives in
/// <c>worker/demosaic_worker/protocol.py</c>; the two are locked by a parity test (§13.4).
/// Never add a second copy on either side — a duplicated constant drifts, and it drifts silently for
/// several revisions before anyone notices.
/// </para>
/// </summary>
public readonly record struct ProtocolVersion(int Major, int Minor) : IComparable<ProtocolVersion>
{
    /// <summary>The version this build speaks.</summary>
    public static ProtocolVersion Current { get; } = new(1, 2);

    /// <summary>Parses "major.minor", rejecting anything else.</summary>
    public static ProtocolVersion Parse(string text)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);

        var parts = text.Split('.');
        if (parts.Length != 2
            || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out var major)
            || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out var minor))
        {
            throw new FormatException($"'{text}' is not a protocol version of the form major.minor.");
        }

        return new ProtocolVersion(major, minor);
    }

    /// <summary>Parses without throwing.</summary>
    public static bool TryParse(string? text, out ProtocolVersion version)
    {
        try
        {
            version = Parse(text!);
            return true;
        }
        catch (Exception e) when (e is ArgumentException or FormatException)
        {
            version = default;
            return false;
        }
    }

    /// <summary>
    /// prd.md §8.1: the host refuses a worker whose <b>major</b> version differs (E7001). Minor
    /// differences are accepted and unknown fields are ignored, which is what lets a newer worker add
    /// fields without invalidating an older host's checkpoints.
    /// </summary>
    public bool IsCompatibleWith(ProtocolVersion other) => Major == other.Major;

    /// <inheritdoc />
    public int CompareTo(ProtocolVersion other) =>
        Major != other.Major ? Major.CompareTo(other.Major) : Minor.CompareTo(other.Minor);

    /// <inheritdoc />
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"{Major}.{Minor}");

    /// <summary>Orders by major then minor.</summary>
    public static bool operator <(ProtocolVersion left, ProtocolVersion right) => left.CompareTo(right) < 0;

    /// <summary>Orders by major then minor.</summary>
    public static bool operator >(ProtocolVersion left, ProtocolVersion right) => left.CompareTo(right) > 0;

    /// <summary>Orders by major then minor.</summary>
    public static bool operator <=(ProtocolVersion left, ProtocolVersion right) => left.CompareTo(right) <= 0;

    /// <summary>Orders by major then minor.</summary>
    public static bool operator >=(ProtocolVersion left, ProtocolVersion right) => left.CompareTo(right) >= 0;
}
