using DeMosaicStudio.Domain.Settings;
using DeMosaicStudio.Infrastructure.Settings;

namespace DeMosaicStudio.Infrastructure.Tests;

/// <summary>
/// Settings surviving a restart.
/// </summary>
/// <remarks>
/// The rule that matters here is the one about failure: a settings file that cannot be read is not
/// an error, it means nothing has been chosen yet. An application that refuses to start over a
/// preference file is a worse failure than the one it is reporting.
/// </remarks>
public sealed class JsonSettingsStoreTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), "demosaic-settings-" + Guid.NewGuid().ToString("N"));

    private string File => Path.Combine(_directory, "settings.json");

    [Fact]
    public void Nothing_stored_yet_reads_as_the_defaults()
    {
        var loaded = new JsonSettingsStore(File).Load();

        Assert.Equal(new DetectionSettings(), loaded.Detection);
        Assert.Equal(new RestorationSettings(), loaded.Restoration);
        Assert.Equal(new EncodeSettings(), loaded.Encode);
    }

    [Fact]
    public void What_is_saved_is_what_comes_back()
    {
        var store = new JsonSettingsStore(File);
        var settings = new JobSettings
        {
            Detection = new DetectionSettings { Confidence = 0.62, MinConfirmFrames = 4 },
            Restoration = new RestorationSettings
            {
                Preset = QualityPreset.Quality,
                TemporalWindow = TemporalWindowSetting.Fixed(7),
                MinRestorationConfidence = 0.88,
                FeatherWidth = 5,
                Refine = new RefineSettings
                {
                    Enabled = true, Strength = 0.25, Model = "sd15", Lora = "lcm",
                    Embeddings = ["zebra", "apple"], Steps = 6,
                },
            },
            Encode = new EncodeSettings
            {
                Codec = OutputCodec.H264,
                Profile = EncoderProfile.SpeedNvenc,
                ConstantQuality = 20,
            },
        };

        Assert.True(store.Save(settings));
        var loaded = store.Load();

        // Compared part by part rather than whole. `JobSettings` is a record, but its
        // `ComparisonPoints` is an `IReadOnlyList<long>`, and record equality on an interface-typed
        // member is reference equality - so a round trip that turns the empty array into an empty
        // list is unequal to itself. Nothing depends on that equality (comparison points are
        // explicitly never fingerprinted), and widening it here would be a Domain change made to
        // suit a test.
        Assert.Equal(settings.Detection, loaded.Detection);
        // Refine carries a list (same reference-equality caveat), so it is compared member-wise.
        Assert.Equal(settings.Restoration with { Refine = new() }, loaded.Restoration with { Refine = new() });
        Assert.Equal(settings.Restoration.Refine.Enabled, loaded.Restoration.Refine.Enabled);
        Assert.Equal(settings.Restoration.Refine.Strength, loaded.Restoration.Refine.Strength);
        Assert.Equal(settings.Restoration.Refine.Model, loaded.Restoration.Refine.Model);
        Assert.Equal(settings.Restoration.Refine.Lora, loaded.Restoration.Refine.Lora);
        Assert.Equal(settings.Restoration.Refine.Embeddings, loaded.Restoration.Refine.Embeddings);
        Assert.Equal(settings.Restoration.Refine.Steps, loaded.Restoration.Refine.Steps);
        Assert.Equal(settings.Encode, loaded.Encode);
        Assert.Equal(settings.Performance, loaded.Performance);
    }

    [Fact]
    public void The_adaptive_temporal_window_survives_the_round_trip()
    {
        // A struct with a private constructor and no settable properties: the default serializer
        // writes it as {} and cannot rebuild it, which would silently turn "auto" into "auto"
        // only by luck and a fixed 7 into nothing at all.
        var store = new JsonSettingsStore(File);
        Assert.True(store.Save(new JobSettings()));

        Assert.True(store.Load().Restoration.TemporalWindow.IsAuto);
        Assert.Contains("\"auto\"", System.IO.File.ReadAllText(File), StringComparison.Ordinal);
    }

    [Fact]
    public void A_fixed_temporal_window_is_written_as_the_number_the_protocol_uses()
    {
        var store = new JsonSettingsStore(File);
        store.Save(new JobSettings
        {
            Restoration = new RestorationSettings { TemporalWindow = TemporalWindowSetting.Fixed(9) },
        });

        Assert.Equal(9, store.Load().Restoration.TemporalWindow.FixedValue);
    }

    [Fact]
    public void A_file_that_is_not_json_reads_as_the_defaults_rather_than_throwing()
    {
        Directory.CreateDirectory(_directory);
        System.IO.File.WriteAllText(File, "{ this is not json");

        Assert.Equal(new DetectionSettings(), new JsonSettingsStore(File).Load().Detection);
    }

    [Fact]
    public void A_hand_edited_value_outside_its_range_is_clamped_on_the_way_in()
    {
        // A settings file is exactly as untrusted as a text box, and this is the path that skips
        // the dialog entirely.
        Directory.CreateDirectory(_directory);
        System.IO.File.WriteAllText(
            File, """{"detection":{"confidence":9.5},"restoration":{"featherWidth":400}}""");

        var loaded = new JsonSettingsStore(File).Load();

        Assert.Equal(0.90, loaded.Detection.Confidence);
        Assert.Equal(9, loaded.Restoration.FeatherWidth);
    }

    [Fact]
    public void An_unknown_field_does_not_lose_the_fields_beside_it()
    {
        Directory.CreateDirectory(_directory);
        System.IO.File.WriteAllText(
            File, """{"somethingNewer":123,"detection":{"confidence":0.55}}""");

        Assert.Equal(0.55, new JsonSettingsStore(File).Load().Detection.Confidence);
    }

    [Fact]
    public void Saving_creates_the_directory_it_needs()
    {
        var store = new JsonSettingsStore(Path.Combine(_directory, "deeper", "settings.json"));

        Assert.True(store.Save(new JobSettings()));
        Assert.True(System.IO.File.Exists(store.FilePath));
    }

    [Fact]
    public void Saving_leaves_no_staging_file_behind()
    {
        var store = new JsonSettingsStore(File);
        store.Save(new JobSettings());

        Assert.False(System.IO.File.Exists(File + ".part"));
    }

    [Fact]
    public void A_path_that_cannot_be_written_is_reported_rather_than_thrown()
    {
        // The directory name is a file, so creating it must fail. Losing the user's edit over a
        // preference that could not be persisted would be the wrong trade.
        Directory.CreateDirectory(_directory);
        var blocker = Path.Combine(_directory, "blocked");
        System.IO.File.WriteAllText(blocker, "not a directory");

        var store = new JsonSettingsStore(Path.Combine(blocker, "settings.json"));

        Assert.False(store.Save(new JobSettings()));
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_directory))
            {
                Directory.Delete(_directory, recursive: true);
            }
        }
        catch (IOException)
        {
            // A temp directory that outlives the test is not a failed test.
        }
    }
}
