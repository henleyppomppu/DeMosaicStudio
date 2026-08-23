using System.Text.Json;
using System.Text.Json.Serialization;
using DeMosaicStudio.Application.Settings;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Infrastructure.Settings;

/// <summary>Settings as a JSON file under the user's local application data.</summary>
/// <remarks>
/// <para>
/// Beside the application rather than in it: a per-user file survives reinstalling, and writing
/// into the install directory needs privileges the application does not have and should not ask
/// for.
/// </para>
/// <para>
/// <b>A missing, corrupt or partly-unreadable file is not an error.</b> It means the user has not
/// chosen anything yet, and refusing to start over a preference file would be a worse failure than
/// the one it reports.
/// </para>
/// </remarks>
public sealed class JsonSettingsStore : ISettingsStore
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter(), new TemporalWindowConverter() },
    };

    private readonly string _path;

    /// <summary>Creates a store over a specific file.</summary>
    public JsonSettingsStore(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _path = path;
    }

    /// <summary>Creates a store at the conventional per-user location.</summary>
    public static JsonSettingsStore Default() => new(Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "DeMosaicStudio",
        "settings.json"));

    /// <summary>The file this store reads and writes.</summary>
    /// <remarks>Not named <c>Path</c>: that shadows <see cref="System.IO.Path"/> inside this type.</remarks>
    public string FilePath => _path;

    /// <inheritdoc />
    public JobSettings Load()
    {
        try
        {
            if (!File.Exists(_path))
            {
                return new JobSettings();
            }

            var loaded = JsonSerializer.Deserialize<JobSettings>(
                File.ReadAllText(_path), Options);

            // Clamped on the way in: a file edited by hand is exactly as untrusted as a text box,
            // and an out-of-range value reaching the worker is what the ranges exist to prevent.
            return loaded is null ? new JobSettings() : SettingsBounds.Clamp(loaded);
        }
        catch (Exception exception) when (exception is IOException or JsonException
                                              or UnauthorizedAccessException or NotSupportedException)
        {
            return new JobSettings();
        }
    }

    /// <inheritdoc />
    public bool Save(JobSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        try
        {
            var directory = Path.GetDirectoryName(_path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Written beside and moved into place: a process ending mid-write would otherwise
            // leave a truncated file that Load quietly reads as "no settings chosen".
            var staging = _path + ".part";
            File.WriteAllText(staging, JsonSerializer.Serialize(settings, Options));
            File.Move(staging, _path, overwrite: true);
            return true;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException
                                              or NotSupportedException)
        {
            return false;
        }
    }
}

/// <summary>
/// Reads and writes <see cref="TemporalWindowSetting"/> as the protocol spells it: the string
/// <c>"auto"</c> or one of the allowed integers.
/// </summary>
/// <remarks>
/// Needed because the type is a struct with a private constructor and no settable properties, which
/// the default serializer turns into <c>{}</c> on the way out and cannot rebuild on the way back.
/// </remarks>
internal sealed class TemporalWindowConverter : JsonConverter<TemporalWindowSetting>
{
    public override TemporalWindowSetting Read(
        ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out var value)
            && TemporalWindowSetting.AllowedValues.Contains(value))
        {
            return TemporalWindowSetting.Fixed(value);
        }

        // Anything else - "auto", a stale value, a type that no longer parses - is the adaptive
        // policy, which is the default and always valid.
        if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
        {
            reader.Skip();
        }

        return TemporalWindowSetting.Auto;
    }

    public override void Write(
        Utf8JsonWriter writer, TemporalWindowSetting value, JsonSerializerOptions options)
    {
        ArgumentNullException.ThrowIfNull(writer);

        if (value.FixedValue is { } fixedValue)
        {
            writer.WriteNumberValue(fixedValue);
            return;
        }

        writer.WriteStringValue("auto");
    }
}
