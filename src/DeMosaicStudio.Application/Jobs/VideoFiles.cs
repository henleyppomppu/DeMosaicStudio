using System.Collections.Immutable;

namespace DeMosaicStudio.Application.Jobs;

/// <summary>
/// Which dropped paths become jobs.
/// <para>
/// Here rather than in the window because it is a decision, not a gesture: the window reports what
/// was dropped and this says what the queue accepts. The file system is reached through two
/// delegates so the rule is testable without one (AGENTS.md), and so the window keeps no rules of
/// its own.
/// </para>
/// </summary>
public static class VideoFiles
{
    /// <summary>Extensions the queue accepts, matching the file dialog's own filter.</summary>
    /// <remarks>
    /// A list rather than "let the worker decide": the worker's answer costs a process round trip
    /// per file, and dropping a folder of a thousand photographs should not spend one on each.
    /// Anything this turns away can still be added through the dialog's "All files" filter.
    /// </remarks>
    public static ImmutableHashSet<string> Extensions { get; } = ImmutableHashSet.Create(
        StringComparer.OrdinalIgnoreCase,
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".m2ts");

    /// <summary>True when the path's extension is one the queue accepts.</summary>
    public static bool IsVideo(string path) =>
        !string.IsNullOrWhiteSpace(path) && Extensions.Contains(Path.GetExtension(path));

    /// <summary>
    /// Expands dropped paths into the video files they stand for: a file if it is one, otherwise
    /// the videos inside the directory.
    /// </summary>
    /// <param name="paths">What was dropped.</param>
    /// <param name="isDirectory">How to tell a directory from a file. Defaults to the file system.</param>
    /// <param name="listDirectory">How to enumerate one, recursively. Defaults to the file system.</param>
    /// <returns>Distinct video paths, in the order they were first seen.</returns>
    public static IReadOnlyList<string> Expand(
        IEnumerable<string> paths,
        Func<string, bool>? isDirectory = null,
        Func<string, IEnumerable<string>>? listDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(paths);

        isDirectory ??= Directory.Exists;
        listDirectory ??= directory =>
            Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories);

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var found = new List<string>();

        foreach (var path in paths)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                continue;
            }

            if (!isDirectory(path))
            {
                if (IsVideo(path) && seen.Add(path))
                {
                    found.Add(path);
                }

                continue;
            }

            // Enumeration is lazy, so a folder that turns out to be unreadable throws part-way
            // through rather than at the call. One such folder in a drop of twenty must not lose
            // the other nineteen, and must not lose what it had already yielded either.
            try
            {
                foreach (var candidate in listDirectory(path))
                {
                    if (IsVideo(candidate) && seen.Add(candidate))
                    {
                        found.Add(candidate);
                    }
                }
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                // Skipped, with whatever it did yield kept.
            }
        }

        return found;
    }
}
