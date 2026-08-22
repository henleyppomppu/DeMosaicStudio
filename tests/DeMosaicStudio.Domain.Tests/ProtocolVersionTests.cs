using System.Text.RegularExpressions;
using DeMosaicStudio.Domain.Protocol;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>prd.md §8.1, §13.4.</summary>
public sealed partial class ProtocolVersionTests
{
    /// <summary>
    /// T-PROTOCOL-VERSION-PARITY-01 — the C# mirror equals the single definition in
    /// <c>worker/demosaic_worker/protocol.py</c>.
    /// <para>
    /// The test reads the Python source rather than trusting a second constant, because a duplicated
    /// version constant is exactly the thing that drifts — and it drifts silently for several
    /// revisions before anyone notices.
    /// </para>
    /// </summary>
    [Fact]
    public void The_host_mirror_matches_the_worker_definition()
    {
        var protocolPy = Path.Combine(
            Directory.GetParent(RepositoryFixtures.Root)!.FullName,
            "worker",
            "demosaic_worker",
            "protocol.py");

        Assert.True(File.Exists(protocolPy), $"Expected the worker protocol module at '{protocolPy}'.");

        var match = ProtocolVersionPattern().Match(File.ReadAllText(protocolPy));

        Assert.True(match.Success, "PROTOCOL_VERSION is not declared as a string literal in protocol.py.");
        Assert.Equal(ProtocolVersion.Current.ToString(), match.Groups["version"].Value);
    }

    /// <summary>prd.md §8.1 — the host refuses a differing major version and accepts a differing minor.</summary>
    [Theory]
    [InlineData("1.0", true)]
    [InlineData("1.4", true)]
    [InlineData("1.99", true)]
    [InlineData("2.0", false)]
    [InlineData("0.9", false)]
    public void Compatibility_is_decided_by_the_major_version(string other, bool expected) =>
        Assert.Equal(expected, ProtocolVersion.Current.IsCompatibleWith(ProtocolVersion.Parse(other)));

    /// <summary>Round-trips through its canonical text form.</summary>
    [Fact]
    public void It_round_trips_through_its_string_form()
    {
        var parsed = ProtocolVersion.Parse(ProtocolVersion.Current.ToString());

        Assert.Equal(ProtocolVersion.Current, parsed);
    }

    /// <summary>Malformed input is rejected rather than coerced.</summary>
    [Theory]
    [InlineData("")]
    [InlineData("1")]
    [InlineData("1.0.0")]
    [InlineData("v1.0")]
    [InlineData("-1.0")]
    [InlineData("1.x")]
    public void Malformed_versions_are_rejected(string text) =>
        Assert.False(ProtocolVersion.TryParse(text, out _));

    /// <summary>Ordering is major-then-minor.</summary>
    [Fact]
    public void Ordering_is_major_then_minor()
    {
        Assert.True(new ProtocolVersion(1, 0) < new ProtocolVersion(1, 1));
        Assert.True(new ProtocolVersion(1, 9) < new ProtocolVersion(2, 0));
        Assert.True(new ProtocolVersion(2, 0) >= new ProtocolVersion(2, 0));
    }

    [GeneratedRegex(@"PROTOCOL_VERSION\s*(?::\s*[^=]+)?=\s*[""'](?<version>\d+\.\d+)[""']")]
    private static partial Regex ProtocolVersionPattern();
}
