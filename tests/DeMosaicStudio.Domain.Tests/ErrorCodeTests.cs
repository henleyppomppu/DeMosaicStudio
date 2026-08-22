using System.Text.Json;
using DeMosaicStudio.Domain.Diagnostics;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>prd.md §10, §13.4.</summary>
public sealed class ErrorCodeTests
{
    private sealed record FixtureEntry(string Code, bool Recoverable, string Severity);

    private static IReadOnlyList<FixtureEntry> LoadFixture()
    {
        var json = File.ReadAllText(RepositoryFixtures.Path("parity", "error_codes.json"));
        using var document = JsonDocument.Parse(json);

        return [.. document.RootElement.GetProperty("codes").EnumerateArray().Select(e => new FixtureEntry(
            e.GetProperty("code").GetString()!,
            e.GetProperty("recoverable").GetBoolean(),
            e.GetProperty("severity").GetString()!))];
    }

    /// <summary>
    /// T-ERROR-PARITY-01 — the C# table and the parity fixture agree exactly, in both directions.
    /// This is half of the cross-language lock; worker/tests carries the other half against the same
    /// fixture, so neither implementation can drift without a red test.
    /// </summary>
    [Fact]
    public void The_error_table_matches_the_parity_fixture_exactly()
    {
        var fixture = LoadFixture();

        var fixtureCodes = fixture.Select(f => f.Code).ToHashSet(StringComparer.Ordinal);
        var tableCodes = ErrorCodes.All.Select(c => c.Code).ToHashSet(StringComparer.Ordinal);

        var missingFromTable = fixtureCodes.Except(tableCodes, StringComparer.Ordinal).Order(StringComparer.Ordinal);
        var missingFromFixture = tableCodes.Except(fixtureCodes, StringComparer.Ordinal).Order(StringComparer.Ordinal);

        Assert.Empty(missingFromTable);
        Assert.Empty(missingFromFixture);

        foreach (var entry in fixture)
        {
            var code = ErrorCodes.Get(entry.Code);
            Assert.Equal(entry.Recoverable, code.Recoverable);
            Assert.Equal(entry.Severity, code.Severity.ToString());
        }
    }

    /// <summary>T-ERROR-UNIQUE-01 — no identifier is defined twice.</summary>
    [Fact]
    public void Every_code_is_unique()
    {
        var duplicates = ErrorCodes.All
            .GroupBy(c => c.Code, StringComparer.Ordinal)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        Assert.Empty(duplicates);
    }

    /// <summary>
    /// T-ERROR-WARNING-01 — warnings are never marked recoverable. prd.md §10.1: a warning is not a
    /// failure, so "may the host retry it" is not a question that applies.
    /// </summary>
    [Fact]
    public void Warnings_are_never_recoverable()
    {
        foreach (var warning in ErrorCodes.All.Where(c => c.IsWarning))
        {
            Assert.False(warning.Recoverable, $"{warning.Code} is a warning but is marked recoverable.");
        }
    }

    /// <summary>T-ERROR-PREFIX-01 — the prefix and the severity agree.</summary>
    [Fact]
    public void The_prefix_letter_matches_the_severity()
    {
        foreach (var code in ErrorCodes.All)
        {
            var expected = code.Code.StartsWith('W') ? ErrorSeverity.Warning : ErrorSeverity.Error;
            Assert.Equal(expected, code.Severity);
        }
    }

    /// <summary>T-ERROR-LOOKUP-01 — an unknown identifier fails loudly rather than returning a default.</summary>
    [Fact]
    public void An_unknown_code_throws_rather_than_returning_a_placeholder()
    {
        Assert.False(ErrorCodes.TryGet("E0000", out _));
        Assert.Throws<KeyNotFoundException>(() => ErrorCodes.Get("E0000"));
    }
}
