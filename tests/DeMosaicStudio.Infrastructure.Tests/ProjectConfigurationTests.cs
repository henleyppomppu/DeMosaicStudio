namespace DeMosaicStudio.Infrastructure.Tests;

/// <summary>
/// Build settings that a compiler cannot check and a running application discovers too late.
/// </summary>
/// <remarks>
/// <para>
/// These read the project files as text, which is unusual for a test and is the point: the rule
/// they protect has no compile-time expression at all. <c>InvariantGlobalization</c> builds
/// cleanly, starts cleanly, and then throws the first time a binding has to convert a value.
/// </para>
/// <para>
/// The measured failure: opening the settings dialog ended the process outright — no dialog, no
/// message, nothing in the window. Diagnosing it took Windows Error Reporting for the exception
/// type and then a UI-automation script driving the button with stderr redirected, because the
/// application had already gone by the time anyone could look.
/// </para>
/// </remarks>
public sealed class ProjectConfigurationTests
{
    private static readonly string Root = FindRoot();

    [Fact]
    public void The_desktop_application_opts_out_of_invariant_globalization()
    {
        // WPF resolves `FrameworkElement.Language` ("en-US") through XmlLanguage.GetSpecificCulture()
        // whenever a binding converts a value — a StringFormat, a converter, or an int reaching a
        // Text property. Under the invariant switch there is no non-neutral culture to find, that
        // call throws, and the exception escapes through the window's message loop.
        var project = File.ReadAllText(
            Path.Combine(Root, "src", "DeMosaicStudio.App", "DeMosaicStudio.App.csproj"));

        Assert.Contains(
            "<InvariantGlobalization>false</InvariantGlobalization>",
            project.Replace(" ", string.Empty, StringComparison.Ordinal),
            StringComparison.Ordinal);
    }

    [Fact]
    public void Every_other_project_still_gets_it_from_the_shared_properties()
    {
        // The opt-out above is deliberately narrow. Invariant globalization is what keeps the
        // domain's parsing and formatting identical on a machine with any locale, and this one is
        // Korean — so the exemption must stay a single project, not a habit.
        var shared = File.ReadAllText(Path.Combine(Root, "Directory.Build.props"));

        Assert.Contains(
            "<InvariantGlobalization>true</InvariantGlobalization>",
            shared.Replace(" ", string.Empty, StringComparison.Ordinal),
            StringComparison.Ordinal);
    }

    private static string FindRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null
               && !File.Exists(Path.Combine(directory.FullName, "DeMosaicStudio.slnx")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName
            ?? throw new DirectoryNotFoundException(
                $"no DeMosaicStudio.slnx above '{AppContext.BaseDirectory}'");
    }
}
