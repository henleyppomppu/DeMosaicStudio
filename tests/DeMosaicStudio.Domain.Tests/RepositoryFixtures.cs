namespace DeMosaicStudio.Domain.Tests;

/// <summary>
/// Locates <c>fixtures/</c> from a test binary. prd.md §13.3.
/// <para>
/// Walks up from the test assembly rather than hard-coding a relative depth, so the tests keep
/// working when the output path changes (configuration, TFM, or a CI layout).
/// </para>
/// </summary>
internal static class RepositoryFixtures
{
    private static readonly Lazy<string> RootLazy = new(FindFixturesRoot);

    /// <summary>Absolute path to the repository's <c>fixtures/</c> directory.</summary>
    public static string Root => RootLazy.Value;

    /// <summary>Absolute path to a file under <c>fixtures/</c>.</summary>
    public static string Path(params string[] relativeParts) =>
        System.IO.Path.Combine([Root, .. relativeParts]);

    private static string FindFixturesRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null)
        {
            var candidate = System.IO.Path.Combine(directory.FullName, "fixtures");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException(
            $"No 'fixtures' directory above '{AppContext.BaseDirectory}'. Run tests from inside the repository.");
    }
}
