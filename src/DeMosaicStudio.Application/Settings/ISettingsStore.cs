using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.Application.Settings;

/// <summary>Where the user's settings live between runs.</summary>
/// <remarks>
/// An interface because the Application layer must not know whether that is a file, a registry key
/// or nothing at all - and because a view model that cannot be constructed without a disk is a view
/// model that cannot be tested.
/// </remarks>
public interface ISettingsStore
{
    /// <summary>Reads the stored settings, or the defaults when there are none.</summary>
    /// <remarks>Never throws: unreadable settings are settings that have not been chosen yet.</remarks>
    JobSettings Load();

    /// <summary>Writes the settings. Returns false when they could not be stored.</summary>
    /// <remarks>
    /// A boolean rather than an exception: failing to persist a preference is worth telling the
    /// user about and is not worth losing their edit over.
    /// </remarks>
    bool Save(JobSettings settings);
}
