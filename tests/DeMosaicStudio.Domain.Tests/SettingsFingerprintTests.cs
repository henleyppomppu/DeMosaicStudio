using DeMosaicStudio.Domain.Jobs;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Domain.Tests;

/// <summary>prd.md §9.3 and §5.16.10.</summary>
public sealed class SettingsFingerprintTests
{
    private static FingerprintSet Fingerprints(JobSettings settings) => new(
        SettingsFingerprint.Compute(settings, FingerprintScope.Detection),
        SettingsFingerprint.Compute(settings, FingerprintScope.Restoration),
        SettingsFingerprint.Compute(settings, FingerprintScope.Encode));

    private static IReadOnlySet<JobArtifact> Discarded(JobSettings before, JobSettings after) =>
        ArtifactInvalidation.Invalidated(Fingerprints(before), Fingerprints(after));

    /// <summary>Identical settings hash identically, so an unchanged resume discards nothing.</summary>
    [Fact]
    public void Identical_settings_discard_nothing()
    {
        var settings = new JobSettings();

        Assert.Empty(Discarded(settings, settings with { }));
    }

    /// <summary>
    /// T-RESUME-FINGERPRINT-01 — a detection change cascades: it discards the analysis <b>and</b>
    /// everything downstream of it.
    /// </summary>
    [Fact]
    public void A_detection_change_discards_analysis_and_video()
    {
        var before = new JobSettings();
        var after = before with { Detection = before.Detection with { Confidence = 0.60 } };

        var discarded = Discarded(before, after);

        Assert.Contains(JobArtifact.Analysis, discarded);
        Assert.Contains(JobArtifact.Video, discarded);
    }

    /// <summary>T-RESUME-FINGERPRINT-02 — a restoration change discards the video only.</summary>
    [Fact]
    public void A_restoration_change_discards_the_video_only()
    {
        var before = new JobSettings();
        var after = before with { Restoration = before.Restoration with { PaddingRatio = 0.20 } };

        var discarded = Discarded(before, after);

        Assert.DoesNotContain(JobArtifact.Analysis, discarded);
        Assert.Contains(JobArtifact.Video, discarded);
    }

    /// <summary>T-RESUME-FINGERPRINT-03 — an encode change discards the video only.</summary>
    [Fact]
    public void An_encode_change_discards_the_video_only()
    {
        var before = new JobSettings();
        var after = before with { Encode = before.Encode with { ConstantQuality = 22 } };

        var discarded = Discarded(before, after);

        Assert.DoesNotContain(JobArtifact.Analysis, discarded);
        Assert.Contains(JobArtifact.Video, discarded);
    }

    /// <summary>
    /// T-RESUME-FINGERPRINT-04 — performance knobs discard nothing.
    /// <para>
    /// This one is load-bearing: the OOM ladder changes precision <i>during</i> a run (prd.md §5.14),
    /// so if precision were fingerprinted, every resume after a downgrade would throw away the
    /// analysis that had just been completed.
    /// </para>
    /// </summary>
    [Fact]
    public void Performance_knobs_discard_nothing()
    {
        var before = new JobSettings();
        var after = before with
        {
            Performance = new PerformanceSettings
            {
                Precision = "fp32",
                VramBudgetBytes = 4L * 1024 * 1024 * 1024,
                TileSize = 256,
                BatchSize = 4,
            },
        };

        Assert.Empty(Discarded(before, after));
    }

    /// <summary>
    /// T-COMPARE-POINTS-NOFP-01 — comparison points are diagnostic and discard nothing. Otherwise a
    /// user could not inspect a finished job without paying to re-run it (prd.md §5.16.11).
    /// </summary>
    [Fact]
    public void Comparison_points_discard_nothing()
    {
        var before = new JobSettings();
        var after = before with { ComparisonPoints = [61250, 1840500] };

        Assert.Empty(Discarded(before, after));
    }

    /// <summary>
    /// T-WINDOW-FINGERPRINT-01 — <c>auto</c> hashes stably. The fingerprint records the requested
    /// value, so it cannot depend on the per-frame resolved K, which varies with scene content and
    /// would invalidate the checkpoint on every run of the same job (prd.md §5.6.1, §9.3).
    /// </summary>
    [Fact]
    public void The_temporal_window_is_fingerprinted_as_requested_not_as_resolved()
    {
        var auto = new JobSettings();
        var alsoAuto = new JobSettings();

        Assert.Equal(
            SettingsFingerprint.Compute(auto, FingerprintScope.Restoration),
            SettingsFingerprint.Compute(alsoAuto, FingerprintScope.Restoration));

        Assert.Contains("temporalWindow=auto", SettingsFingerprint.Canonicalize(auto, FingerprintScope.Restoration), StringComparison.Ordinal);

        var fixedWindow = auto with
        {
            Restoration = auto.Restoration with { TemporalWindow = TemporalWindowSetting.Fixed(9) },
        };

        Assert.Contains("temporalWindow=9", SettingsFingerprint.Canonicalize(fixedWindow, FingerprintScope.Restoration), StringComparison.Ordinal);
        Assert.Contains(JobArtifact.Video, Discarded(auto, fixedWindow));
    }

    /// <summary>The confidence gate changes output, so it must invalidate the video.</summary>
    [Fact]
    public void Changing_the_confidence_gate_discards_the_video()
    {
        var before = new JobSettings();
        var after = before with
        {
            Restoration = before.Restoration with { MinRestorationConfidence = 0.40 },
        };

        var discarded = Discarded(before, after);

        Assert.DoesNotContain(JobArtifact.Analysis, discarded);
        Assert.Contains(JobArtifact.Video, discarded);
    }

    /// <summary>
    /// T-RESUME-FINGERPRINT-05 — an unknown or missing recorded fingerprint compares as
    /// <b>changed</b>, never as equal.
    /// <para>
    /// A null-lifting comparison that evaluates to <c>false</c> on unknown data silently reuses a
    /// previous file's artifacts for a different source. That is data corruption, not a UX bug.
    /// </para>
    /// </summary>
    [Fact]
    public void An_unknown_fingerprint_counts_as_changed()
    {
        Assert.True(ArtifactInvalidation.Changed(null, "sha256:abc"));
        Assert.True(ArtifactInvalidation.Changed("sha256:abc", null));
        Assert.True(ArtifactInvalidation.Changed(null, null));
        Assert.False(ArtifactInvalidation.Changed("sha256:abc", "sha256:abc"));

        var current = Fingerprints(new JobSettings());
        var nothingRecorded = new FingerprintSet(null, null, null);

        var discarded = ArtifactInvalidation.Invalidated(nothingRecorded, current);

        Assert.Contains(JobArtifact.Analysis, discarded);
        Assert.Contains(JobArtifact.Video, discarded);
    }

    /// <summary>The canonical form is stable and sorted, which is what makes the Python mirror possible.</summary>
    [Fact]
    public void The_canonical_form_is_ordinally_sorted_and_stable()
    {
        var canonical = SettingsFingerprint.Canonicalize(new JobSettings(), FingerprintScope.Restoration);
        var lines = canonical.Split('\n');

        Assert.Equal(lines.OrderBy(l => l, StringComparer.Ordinal), lines);
        Assert.Equal(canonical, SettingsFingerprint.Canonicalize(new JobSettings(), FingerprintScope.Restoration));
        Assert.StartsWith("sha256:", SettingsFingerprint.Compute(new JobSettings(), FingerprintScope.Restoration), StringComparison.Ordinal);
    }

    /// <summary>
    /// T-SETTINGS-FINGERPRINT-MAP-01 — every setting property is either in a fingerprint or
    /// deliberately excluded. A property added without a decision fails here rather than silently
    /// becoming un-fingerprinted (prd.md §5.16.10).
    /// </summary>
    [Theory]
    [InlineData(typeof(DetectionSettings), FingerprintScope.Detection)]
    [InlineData(typeof(RestorationSettings), FingerprintScope.Restoration)]
    [InlineData(typeof(EncodeSettings), FingerprintScope.Encode)]
    public void Every_fingerprinted_setting_appears_in_its_canonical_form(Type settingsType, FingerprintScope scope)
    {
        var canonical = SettingsFingerprint.Canonicalize(new JobSettings(), scope);
        var keys = canonical.Split('\n').Select(l => l.Split('=')[0]).ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (var property in settingsType.GetProperties())
        {
            Assert.True(
                keys.Contains(property.Name),
                $"{settingsType.Name}.{property.Name} is not in the {scope} fingerprint. "
                + "Add it, or move it to PerformanceSettings/JobSettings with a recorded reason (prd.md §9.3, §5.16.10).");
        }
    }
}
