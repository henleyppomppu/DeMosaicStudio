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
}
