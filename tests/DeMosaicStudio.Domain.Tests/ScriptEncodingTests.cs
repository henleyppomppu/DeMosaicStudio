namespace DeMosaicStudio.Domain.Tests;

/// <summary>
/// T-SCRIPT-ENCODING-01. prd.md §4.4, AC-4.4.
/// <para>
/// Windows PowerShell 5.1 reads a BOM-less file as ANSI — CP949 on the target machine. A CP949 lead
/// byte swallows the following byte, UTF-8 Hangul is three bytes, so the alignment slips and the
/// script fails at <b>parse</b> time rather than at run time. <c>pwsh</c> 7 reads BOM-less UTF-8
/// correctly, so this never reproduces under a modern shell and a syntax check will not find it.
/// </para>
/// <para>
/// The test inspects bytes directly and discovers scripts by globbing, so a newly added script is
/// covered without anyone remembering to add it here.
/// </para>
/// </summary>
public sealed class ScriptEncodingTests
{
    private static readonly byte[] Utf8Bom = [0xEF, 0xBB, 0xBF];

    private static readonly string[] ScriptExtensions = ["*.ps1", "*.psm1", "*.iss"];

    private static string RepositoryRoot => Directory.GetParent(RepositoryFixtures.Root)!.FullName;

    public static TheoryData<string> Scripts()
    {
        var data = new TheoryData<string>();

        foreach (var pattern in ScriptExtensions)
        {
            foreach (var file in Directory.EnumerateFiles(RepositoryRoot, pattern, SearchOption.AllDirectories))
            {
                var relative = Path.GetRelativePath(RepositoryRoot, file);

                // Build output and third-party trees are not ours to encode.
                if (relative.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                    || relative.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                    || relative.StartsWith(".venv", StringComparison.Ordinal)
                    || relative.StartsWith("tools", StringComparison.Ordinal))
                {
                    continue;
                }

                data.Add(relative);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Scripts))]
    public void Every_script_is_saved_as_utf8_with_a_bom(string relativePath)
    {
        var full = Path.Combine(RepositoryRoot, relativePath);
        var head = new byte[3];

        using (var stream = File.OpenRead(full))
        {
            var read = stream.Read(head, 0, head.Length);
            Assert.True(read == 3, $"{relativePath} is too short to carry a BOM.");
        }

        Assert.True(
            head.SequenceEqual(Utf8Bom),
            $"{relativePath} has no UTF-8 BOM. Windows PowerShell 5.1 will read it as ANSI (CP949) "
            + "and fail at parse time. Re-save it as 'UTF-8 with BOM' (prd.md §4.4).");
    }

    /// <summary>
    /// The repository is expected to contain scripts. An empty glob would make the test above pass
    /// vacuously and hide the regression it exists to catch.
    /// </summary>
    [Fact]
    public void At_least_one_script_is_discovered() => Assert.NotEmpty(Scripts());
}
