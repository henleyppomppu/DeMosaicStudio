using DeMosaicStudio.Application.Diagnostics;
using DeMosaicStudio.Domain.Diagnostics;

namespace DeMosaicStudio.Application.Tests;

/// <summary>
/// Every numbered code has a line the user can read.
/// </summary>
/// <remarks>
/// The failure this prevents is quiet: a code added to <see cref="ErrorCodes"/> and forgotten here
/// still works, and the user simply meets one sentence of English in an otherwise Korean window,
/// on the day something has already gone wrong.
/// </remarks>
public sealed class ErrorTextTests
{
    [Fact]
    public void Every_code_in_the_table_has_a_korean_line()
    {
        var missing = ErrorCodes.All
            .Select(code => code.Code)
            .Except(ErrorText.Translated, StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(missing.Count == 0, $"no Korean line for: {string.Join(", ", missing)}");
    }

    [Fact]
    public void Nothing_is_translated_that_is_not_a_real_code()
    {
        // A typo in a key is otherwise invisible: the entry is simply never reached.
        var unknown = ErrorText.Translated
            .Except(ErrorCodes.All.Select(code => code.Code), StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(unknown.Count == 0, $"not a code in ErrorCodes: {string.Join(", ", unknown)}");
    }

    [Fact]
    public void The_line_keeps_the_number_the_documentation_is_keyed_on()
    {
        var line = ErrorText.Line(ErrorCodes.E4401);

        Assert.StartsWith("E4401", line, StringComparison.Ordinal);
        Assert.Contains("GPU", line, StringComparison.Ordinal);
    }

    [Fact]
    public void An_unknown_code_falls_back_to_the_protocol_meaning_rather_than_throwing()
    {
        // A worker newer than this host can report codes this table has never seen. Losing the
        // failure over its wording would be worse than showing it in English.
        var newer = new ErrorCode(
            "E9999", "Something this host has never heard of", Recoverable: false,
            Severity: ErrorSeverity.Error);

        Assert.Equal("Something this host has never heard of", ErrorText.Describe(newer));
    }

    [Fact]
    public void The_english_meanings_are_left_alone_for_the_parity_fixture()
    {
        // §13.4 locks ErrorCodes to worker/demosaic_worker/errors.py. Translating in place would
        // have broken that, which is why the Korean lives beside it instead of replacing it.
        Assert.Equal("GPU OOM, mitigation ladder exhausted", ErrorCodes.E4401.Meaning);
    }
}
