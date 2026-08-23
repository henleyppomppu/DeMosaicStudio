using DeMosaicStudio.Application.Jobs;

namespace DeMosaicStudio.Application.Tests;

/// <summary>
/// What a drop turns into. The window supplies paths; this decides which of them become jobs.
/// </summary>
/// <remarks>
/// The file system is injected, so these run on a machine with no such folders and no WPF — which
/// is the point of the rule living here rather than in the window's code-behind.
/// </remarks>
public sealed class VideoFilesTests
{
    private static readonly Dictionary<string, string[]> Tree = new(StringComparer.OrdinalIgnoreCase)
    {
        [@"C:\clips"] = [@"C:\clips\a.mp4", @"C:\clips\notes.txt", @"C:\clips\deep\b.MKV"],
        [@"C:\empty"] = [],
    };

    private static IReadOnlyList<string> Expand(params string[] paths) =>
        VideoFiles.Expand(paths, Tree.ContainsKey, directory => Tree[directory]);

    [Fact]
    public void A_video_file_is_taken_as_it_is()
    {
        Assert.Equal([@"D:\one.mp4"], Expand(@"D:\one.mp4"));
    }

    [Theory]
    [InlineData(".mp4")]
    [InlineData(".MKV")]
    [InlineData(".webm")]
    [InlineData(".m2ts")]
    public void The_extension_check_ignores_case(string extension)
    {
        Assert.True(VideoFiles.IsVideo("clip" + extension));
    }

    [Theory]
    [InlineData(@"D:\notes.txt")]
    [InlineData(@"D:\photo.jpg")]
    [InlineData(@"D:\no-extension")]
    [InlineData("")]
    public void Anything_that_is_not_a_video_is_left_out(string path)
    {
        Assert.Empty(Expand(path));
    }

    [Fact]
    public void A_folder_becomes_the_videos_inside_it_including_subfolders()
    {
        Assert.Equal([@"C:\clips\a.mp4", @"C:\clips\deep\b.MKV"], Expand(@"C:\clips"));
    }

    [Fact]
    public void An_empty_folder_contributes_nothing_rather_than_failing()
    {
        Assert.Empty(Expand(@"C:\empty"));
    }

    [Fact]
    public void The_same_file_dropped_twice_is_queued_once()
    {
        Assert.Equal([@"D:\one.mp4"], Expand(@"D:\one.mp4", @"D:\one.mp4"));
    }

    [Fact]
    public void A_file_that_is_also_inside_a_dropped_folder_is_queued_once()
    {
        Assert.Equal(
            [@"C:\clips\a.mp4", @"C:\clips\deep\b.MKV"],
            Expand(@"C:\clips\a.mp4", @"C:\clips"));
    }

    [Fact]
    public void Order_follows_the_drop_not_the_alphabet()
    {
        Assert.Equal([@"D:\z.mp4", @"D:\a.mp4"], Expand(@"D:\z.mp4", @"D:\a.mp4"));
    }

    [Fact]
    public void An_unreadable_folder_does_not_lose_the_rest_of_the_drop()
    {
        var result = VideoFiles.Expand(
            [@"C:\locked", @"D:\one.mp4"],
            path => path == @"C:\locked",
            _ => throw new UnauthorizedAccessException());

        Assert.Equal([@"D:\one.mp4"], result);
    }

    [Fact]
    public void An_unreadable_folder_keeps_what_it_had_already_yielded()
    {
        // Enumeration is lazy, so the failure lands part-way through rather than at the call.
        Assert.Equal([@"C:\part\a.mp4"], VideoFiles.Expand(
            [@"C:\part"],
            path => path == @"C:\part",
            _ => Yielding()));

        static IEnumerable<string> Yielding()
        {
            yield return @"C:\part\a.mp4";
            throw new IOException("the drive went away");
        }
    }

    [Fact]
    public void The_accepted_extensions_all_start_with_a_dot()
    {
        // GetExtension returns ".mp4"; an entry written "mp4" would never match anything.
        Assert.All(VideoFiles.Extensions, extension => Assert.StartsWith(".", extension, StringComparison.Ordinal));
    }
}
