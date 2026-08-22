using System.Globalization;
using System.Text.Json;
using DeMosaicStudio.Domain.Policies;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>
/// Cross-language policy parity. prd.md §13.4.
/// <para>
/// The window policy and the restoration router exist twice — here and in
/// <c>worker/demosaic_worker/policies.py</c> — because the host must be able to tell the user what
/// will happen and the worker must actually do it. The fixture is generated from the Python side by
/// <c>scripts/make_policy_fixture.py</c> and asserted from both, so a change to either
/// implementation that is not mirrored turns one of the two red.
/// </para>
/// <para>
/// This matters more than the usual parity case: these are the policies the Phase 0 and Phase 2
/// measurements rewrote (D-16). A drift here would mean the host promises a multi-frame restoration
/// the worker refuses to attempt, or worse, the reverse.
/// </para>
/// </summary>
public sealed class PolicyParityTests
{
    private static JsonDocument LoadFixture() =>
        JsonDocument.Parse(File.ReadAllText(RepositoryFixtures.Path("parity", "policies.json")));

    private static double Number(JsonElement element, string name) =>
        element.GetProperty(name).GetDouble();

    private static QualityPreset Preset(string name) => name switch
    {
        "Fast" => QualityPreset.Fast,
        "Balanced" => QualityPreset.Balanced,
        "Quality" => QualityPreset.Quality,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown preset"),
    };

    private static GridAnchor Anchor(string name) => name switch
    {
        "SCREEN" => GridAnchor.Screen,
        "OBJECT" => GridAnchor.ObjectTracked,
        "UNKNOWN" => GridAnchor.Unknown,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown anchor"),
    };

    /// <summary>Every window decision in the fixture matches, value and reason.</summary>
    [Fact]
    public void The_window_policy_matches_the_worker()
    {
        using var document = LoadFixture();
        var mismatches = new List<string>();
        var cases = 0;

        foreach (var expected in document.RootElement.GetProperty("windowCases").EnumerateArray())
        {
            cases++;

            var settingElement = expected.GetProperty("setting");
            var setting = settingElement.ValueKind == JsonValueKind.Null
                ? TemporalWindowSetting.Auto
                : TemporalWindowSetting.Fixed(settingElement.GetInt32());

            var decision = TemporalWindowPolicy.Decide(new TemporalWindowInputs(
                setting,
                Preset(expected.GetProperty("preset").GetString()!),
                Number(expected, "motion"),
                Anchor(expected.GetProperty("anchor").GetString()!),
                expected.GetProperty("sameSceneFrames").GetInt32(),
                expected.GetProperty("streamFrames").GetInt32(),
                expected.GetProperty("vramMaxWindow").GetInt32()));

            var wantedEffective = expected.GetProperty("effective").GetInt32();
            var wantedReason = expected.GetProperty("reason").GetString()!;

            if (decision.EffectiveWindow != wantedEffective
                || !string.Equals(decision.Reason.ToString(), wantedReason, StringComparison.Ordinal))
            {
                mismatches.Add(string.Create(
                    CultureInfo.InvariantCulture,
                    $"motion={Number(expected, "motion")} preset={expected.GetProperty("preset")} "
                    + $"setting={settingElement} anchor={expected.GetProperty("anchor")}: "
                    + $"worker says {wantedEffective}/{wantedReason}, host says "
                    + $"{decision.EffectiveWindow}/{decision.Reason}"));
            }
        }

        Assert.True(cases > 0, "the fixture is empty");
        Assert.Empty(mismatches);
    }

    /// <summary>Every routing decision in the fixture matches, path and reason.</summary>
    [Fact]
    public void The_router_matches_the_worker()
    {
        using var document = LoadFixture();
        var mismatches = new List<string>();
        var cases = 0;

        foreach (var expected in document.RootElement.GetProperty("routeCases").EnumerateArray())
        {
            cases++;

            var reduction = Enum.Parse<WindowReductionReason>(
                expected.GetProperty("windowReason").GetString()!);

            var decision = RestorationRouter.Route(new RouteInputs(
                expected.GetProperty("hasRegion").GetBoolean(),
                expected.GetProperty("regionArea").GetInt32(),
                MinRegionArea: 256,
                expected.GetProperty("isConfirmed").GetBoolean(),
                expected.GetProperty("userDisabled").GetBoolean(),
                expected.GetProperty("withheldByConfidenceGate").GetBoolean(),
                expected.GetProperty("degradationChainExhausted").GetBoolean(),
                Anchor(expected.GetProperty("anchor").GetString()!),
                Number(expected, "motion"),
                new TemporalWindowDecision(expected.GetProperty("window").GetInt32(), 5, reduction),
                expected.GetProperty("validAlignedNeighbours").GetInt32(),
                Number(expected, "meanAlignmentConfidence"),
                AlignConfMin: 0.35,
                expected.GetProperty("occlusionInvalidatedNeighbours").GetBoolean()));

            var wantedPath = expected.GetProperty("path").GetString()!;
            var wantedReason = expected.GetProperty("reason").GetString()!;

            if (!string.Equals(decision.Path.ToString(), wantedPath, StringComparison.Ordinal)
                || !string.Equals(decision.Reason.ToString(), wantedReason, StringComparison.Ordinal))
            {
                mismatches.Add(
                    $"worker says {wantedPath}/{wantedReason}, host says {decision.Path}/{decision.Reason}");
            }
        }

        Assert.True(cases > 0, "the fixture is empty");
        Assert.Empty(mismatches);
    }

    /// <summary>
    /// The fixture exercises every reason the enum can produce. A parity fixture that happened to
    /// miss a branch would pass while that branch drifted freely.
    /// </summary>
    [Fact]
    public void The_fixture_covers_every_routing_reason()
    {
        using var document = LoadFixture();

        var covered = document.RootElement
            .GetProperty("routeCases")
            .EnumerateArray()
            .Select(c => c.GetProperty("reason").GetString()!)
            .ToHashSet(StringComparer.Ordinal);

        var missing = Enum.GetNames<RouteReason>()
            .Where(name => !covered.Contains(name))
            .ToList();

        Assert.Empty(missing);
    }

    /// <summary>
    /// Every confidence-gate sequence in the fixture matches, frame by frame.
    /// <para>
    /// The gate was mirrored in both languages and locked by nothing, and it had the same hole in
    /// both: a track started <i>open</i>, so a gate set above every reachable confidence still let
    /// (hysteresis - 1) frames through each time a track appeared. Sequences, not single calls:
    /// the whole point of the gate is what it does over time.
    /// </para>
    /// </summary>
    [Fact]
    public void The_confidence_gate_matches_the_worker()
    {
        using var document = LoadFixture();
        var mismatches = new List<string>();

        foreach (var expected in document.RootElement.GetProperty("confidenceGateCases").EnumerateArray())
        {
            var description = expected.GetProperty("description").GetString()!;
            var gate = new ConfidenceGate(Number(expected, "threshold"));

            var confidences = expected.GetProperty("confidences").EnumerateArray()
                .Select(c => c.GetDouble()).ToList();
            var wanted = expected.GetProperty("withheld").EnumerateArray()
                .Select(w => w.GetBoolean()).ToList();

            for (var i = 0; i < confidences.Count; i++)
            {
                var got = gate.ShouldWithhold(1, confidences[i]);
                if (got != wanted[i])
                {
                    mismatches.Add($"{description}: frame {i} gave {got}, worker gave {wanted[i]}");
                }
            }

            var wantedCount = expected.GetProperty("gatedTrackCount").GetInt32();
            if (gate.GatedTrackCount != wantedCount)
            {
                mismatches.Add(
                    $"{description}: GatedTrackCount {gate.GatedTrackCount}, worker {wantedCount}");
            }
        }

        Assert.Empty(mismatches);
    }

    /// <summary>Gate state is per track, and interleaving two tracks must not mix them.</summary>
    [Fact]
    public void The_confidence_gate_keeps_track_state_apart()
    {
        using var document = LoadFixture();
        var expected = document.RootElement.GetProperty("confidenceGateInterleaved");

        var confidences = expected.GetProperty("confidences").EnumerateArray()
            .Select(c => c.GetDouble()).ToList();
        var gate = new ConfidenceGate(Number(expected, "threshold"));
        var mismatches = new List<string>();
        var frame = 0;

        foreach (var wantedFrame in expected.GetProperty("withheld").EnumerateArray())
        {
            var wanted = wantedFrame.EnumerateArray().Select(w => w.GetBoolean()).ToList();
            var got = new[] { gate.ShouldWithhold(1, confidences[0]), gate.ShouldWithhold(2, confidences[1]) };

            for (var track = 0; track < wanted.Count; track++)
            {
                if (got[track] != wanted[track])
                {
                    mismatches.Add($"frame {frame} track {track + 1}: {got[track]} vs {wanted[track]}");
                }
            }

            frame++;
        }

        Assert.Empty(mismatches);
        Assert.Equal(expected.GetProperty("gatedTrackCount").GetInt32(), gate.GatedTrackCount);
    }

    /// <summary>
    /// Every confidence-smoother sequence in the fixture matches.
    /// <para>
    /// The smoother exists because the gate's parameter is named <c>smoothedConfidence</c> and the
    /// pipeline was handing it a raw per-frame value. Both languages have to damp identically, or
    /// the host's preview of what will be withheld disagrees with what the worker withholds.
    /// </para>
    /// </summary>
    [Fact]
    public void The_confidence_smoother_matches_the_worker()
    {
        using var document = LoadFixture();
        var mismatches = new List<string>();

        foreach (var expected in document.RootElement.GetProperty("confidenceSmootherCases").EnumerateArray())
        {
            var description = expected.GetProperty("description").GetString()!;
            var smoother = new ConfidenceSmoother(expected.GetProperty("window").GetInt32());

            var confidences = expected.GetProperty("confidences").EnumerateArray()
                .Select(c => c.GetDouble()).ToList();
            var wanted = expected.GetProperty("smoothed").EnumerateArray()
                .Select(s => s.GetDouble()).ToList();

            for (var i = 0; i < confidences.Count; i++)
            {
                var got = smoother.Update(1, confidences[i]);
                if (Math.Abs(got - wanted[i]) > 1e-9)
                {
                    mismatches.Add($"{description}: frame {i} gave {got}, worker gave {wanted[i]}");
                }
            }
        }

        Assert.Empty(mismatches);
    }

    /// <summary>Smoother state is per track.</summary>
    [Fact]
    public void The_confidence_smoother_keeps_track_state_apart()
    {
        using var document = LoadFixture();
        var expected = document.RootElement.GetProperty("confidenceSmootherInterleaved");

        var confidences = expected.GetProperty("confidences").EnumerateArray()
            .Select(c => c.GetDouble()).ToList();
        var smoother = new ConfidenceSmoother(expected.GetProperty("window").GetInt32());
        var mismatches = new List<string>();
        var frame = 0;

        foreach (var wantedFrame in expected.GetProperty("smoothed").EnumerateArray())
        {
            var wanted = wantedFrame.EnumerateArray().Select(s => s.GetDouble()).ToList();
            var got = new[] { smoother.Update(1, confidences[0]), smoother.Update(2, confidences[1]) };

            for (var track = 0; track < wanted.Count; track++)
            {
                if (Math.Abs(got[track] - wanted[track]) > 1e-9)
                {
                    mismatches.Add($"frame {frame} track {track + 1}: {got[track]} vs {wanted[track]}");
                }
            }

            frame++;
        }

        Assert.Empty(mismatches);
    }

    /// <summary>The smoother's time constant is the gate's own window, in both languages.</summary>
    [Fact]
    public void The_smoother_window_defaults_to_the_gate_hysteresis()
    {
        Assert.Equal(1.0 / ConfidenceGate.DefaultHysteresisFrames, new ConfidenceSmoother().Alpha, 12);
    }
}
