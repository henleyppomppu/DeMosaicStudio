using DeMosaicStudio.Domain.Diagnostics;

namespace DeMosaicStudio.Domain.Tracking;

/// <summary>Track lifecycle states. prd.md §5.3.3.</summary>
public enum TrackState
{
    /// <summary>Detected but not yet confirmed on enough consecutive frames.</summary>
    Tentative,

    /// <summary>Confirmed and being restored.</summary>
    Active,

    /// <summary>Temporarily hidden; mask propagated by prediction.</summary>
    Occluded,

    /// <summary>Missing beyond <c>max_missing_frames</c>.</summary>
    Lost,

    /// <summary>Found again after being lost; rejoins Active.</summary>
    Reacquired,

    /// <summary>Finished. Terminal.</summary>
    Terminated,
}

/// <summary>Raised when a caller attempts a transition the table forbids. Carries E3201.</summary>
public sealed class TrackStateViolationException : InvalidOperationException
{
    /// <summary>Creates the exception for a specific illegal transition.</summary>
    public TrackStateViolationException(TrackState from, TrackState to)
        : base($"{ErrorCodes.E3201.Code}: illegal track transition {from} -> {to} (prd.md §5.3.3).")
    {
        From = from;
        To = to;
    }

    /// <summary>Creates the exception with a message.</summary>
    public TrackStateViolationException(string message) : base(message)
    {
    }

    /// <summary>Creates the exception with a message and inner exception.</summary>
    public TrackStateViolationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception with no detail.</summary>
    public TrackStateViolationException()
    {
    }

    /// <summary>State the caller tried to leave.</summary>
    public TrackState From { get; }

    /// <summary>State the caller tried to enter.</summary>
    public TrackState To { get; }

    /// <summary>The numbered code this violation reports as.</summary>
    public static ErrorCode Code => ErrorCodes.E3201;
}

/// <summary>
/// Table-driven track state machine. prd.md §5.3.3.
/// <para>
/// Deliberately a table rather than scattered conditionals: forward progress along the active path is
/// allowed, backward transitions are rejected, and <see cref="TrackState.Terminated"/> is reachable
/// only from <see cref="TrackState.Lost"/> or at end of stream. Violations raise E3201 rather than
/// silently correcting — a tracker that quietly repairs its own state hides the bug that caused it.
/// </para>
/// </summary>
public static class TrackStateMachine
{
    private static readonly (TrackState From, TrackState To)[] AllowedTransitions =
    [
        (TrackState.Tentative, TrackState.Active),
        (TrackState.Tentative, TrackState.Lost),
        (TrackState.Active, TrackState.Occluded),
        (TrackState.Active, TrackState.Lost),
        (TrackState.Occluded, TrackState.Active),
        (TrackState.Occluded, TrackState.Lost),
        (TrackState.Lost, TrackState.Reacquired),
        (TrackState.Lost, TrackState.Terminated),
        (TrackState.Reacquired, TrackState.Active),
        (TrackState.Reacquired, TrackState.Lost),
    ];

    private static readonly HashSet<(TrackState, TrackState)> Allowed = [.. AllowedTransitions];

    /// <summary>Every legal transition, for exhaustiveness tests.</summary>
    public static IReadOnlyList<(TrackState From, TrackState To)> Transitions => AllowedTransitions;

    /// <summary>States from which no transition is possible.</summary>
    public static bool IsTerminal(TrackState state) => state == TrackState.Terminated;

    /// <summary>
    /// Whether a transition is legal. End of stream is the one path that may terminate any
    /// non-terminal state (§5.3.3).
    /// </summary>
    public static bool CanTransition(TrackState from, TrackState to, bool endOfStream = false)
    {
        if (IsTerminal(from))
        {
            return false;
        }

        if (endOfStream && to == TrackState.Terminated)
        {
            return true;
        }

        return Allowed.Contains((from, to));
    }

    /// <summary>Applies a transition, throwing <see cref="TrackStateViolationException"/> when illegal.</summary>
    public static TrackState Transition(TrackState from, TrackState to, bool endOfStream = false) =>
        CanTransition(from, to, endOfStream)
            ? to
            : throw new TrackStateViolationException(from, to);
}
