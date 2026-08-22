using DeMosaicStudio.Domain.Diagnostics;
using DeMosaicStudio.Domain.Tracking;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>prd.md §5.3.3.</summary>
public sealed class TrackStateMachineTests
{
    /// <summary>
    /// T-TRACK-STATE-TABLE-01, first half — every transition in the table is accepted and returns the
    /// target state.
    /// </summary>
    [Fact]
    public void Every_transition_in_the_table_is_allowed()
    {
        foreach (var (from, to) in TrackStateMachine.Transitions)
        {
            Assert.True(TrackStateMachine.CanTransition(from, to));
            Assert.Equal(to, TrackStateMachine.Transition(from, to));
        }
    }

    /// <summary>
    /// T-TRACK-STATE-TABLE-01, second half — every transition <b>absent</b> from the table is
    /// rejected. The state machine does not silently correct itself: a tracker that quietly repairs
    /// its own state hides the bug that caused the corruption.
    /// </summary>
    [Fact]
    public void Every_transition_absent_from_the_table_throws()
    {
        var allowed = TrackStateMachine.Transitions.ToHashSet();

        foreach (var from in Enum.GetValues<TrackState>())
        foreach (var to in Enum.GetValues<TrackState>())
        {
            if (allowed.Contains((from, to)))
            {
                continue;
            }

            Assert.False(TrackStateMachine.CanTransition(from, to));

            var violation = Assert.Throws<TrackStateViolationException>(
                () => TrackStateMachine.Transition(from, to));

            Assert.Equal(from, violation.From);
            Assert.Equal(to, violation.To);
            Assert.Contains(ErrorCodes.E3201.Code, violation.Message, StringComparison.Ordinal);
        }
    }

    /// <summary>Terminated is terminal: nothing leaves it, not even end of stream.</summary>
    [Fact]
    public void Nothing_leaves_the_terminated_state()
    {
        foreach (var to in Enum.GetValues<TrackState>())
        {
            Assert.False(TrackStateMachine.CanTransition(TrackState.Terminated, to));
            Assert.False(TrackStateMachine.CanTransition(TrackState.Terminated, to, endOfStream: true));
        }
    }

    /// <summary>
    /// prd.md §5.3.3 — Terminated is reachable only from Lost, or from anywhere at end of stream.
    /// Those are two different things and the table keeps them distinct.
    /// </summary>
    [Fact]
    public void Terminated_is_reachable_from_lost_or_at_end_of_stream_only()
    {
        Assert.True(TrackStateMachine.CanTransition(TrackState.Lost, TrackState.Terminated));

        Assert.False(TrackStateMachine.CanTransition(TrackState.Active, TrackState.Terminated));
        Assert.True(TrackStateMachine.CanTransition(TrackState.Active, TrackState.Terminated, endOfStream: true));

        Assert.False(TrackStateMachine.CanTransition(TrackState.Occluded, TrackState.Terminated));
        Assert.True(TrackStateMachine.CanTransition(TrackState.Occluded, TrackState.Terminated, endOfStream: true));
    }

    /// <summary>A track cannot go backwards along the confirmation path.</summary>
    [Fact]
    public void Backward_transitions_are_rejected()
    {
        Assert.False(TrackStateMachine.CanTransition(TrackState.Active, TrackState.Tentative));
        Assert.False(TrackStateMachine.CanTransition(TrackState.Reacquired, TrackState.Tentative));
        Assert.False(TrackStateMachine.CanTransition(TrackState.Occluded, TrackState.Tentative));
    }

    /// <summary>The normal life of a track, walked end to end.</summary>
    [Fact]
    public void The_ordinary_lifecycle_walks_cleanly()
    {
        var state = TrackState.Tentative;

        state = TrackStateMachine.Transition(state, TrackState.Active);
        state = TrackStateMachine.Transition(state, TrackState.Occluded);
        state = TrackStateMachine.Transition(state, TrackState.Active);
        state = TrackStateMachine.Transition(state, TrackState.Lost);
        state = TrackStateMachine.Transition(state, TrackState.Reacquired);
        state = TrackStateMachine.Transition(state, TrackState.Active);
        state = TrackStateMachine.Transition(state, TrackState.Lost);
        state = TrackStateMachine.Transition(state, TrackState.Terminated);

        Assert.True(TrackStateMachine.IsTerminal(state));
    }
}
