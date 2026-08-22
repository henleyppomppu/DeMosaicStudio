namespace DeMosaicStudio.Domain.Diagnostics;

/// <summary>Whether an entry fails a job or merely annotates it. prd.md §10.1.</summary>
public enum ErrorSeverity
{
    /// <summary>Fails the operation it belongs to.</summary>
    Error,

    /// <summary>Annotates the job; never fails it.</summary>
    Warning,
}

/// <summary>
/// One numbered entry from prd.md §10.2. Every failure crossing the host/worker boundary carries
/// one of these; a free-text-only error is a defect (§10.1).
/// </summary>
/// <param name="Code">Stable identifier, e.g. <c>E4002</c>. Never renumbered.</param>
/// <param name="Meaning">Short English description, mirrored in docs/ERROR_CODES.md.</param>
/// <param name="Recoverable">
/// Whether the host may auto-retry once with the mitigation implied by the code (§10.3).
/// Always <see langword="false"/> for warnings, which are not failures at all.
/// </param>
/// <param name="Severity">Error or warning.</param>
public sealed record ErrorCode(string Code, string Meaning, bool Recoverable, ErrorSeverity Severity)
{
    /// <summary>True when this entry annotates rather than fails.</summary>
    public bool IsWarning => Severity == ErrorSeverity.Warning;
}
