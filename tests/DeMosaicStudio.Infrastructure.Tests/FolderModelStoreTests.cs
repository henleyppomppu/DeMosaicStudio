using DeMosaicStudio.Infrastructure.Settings;

namespace DeMosaicStudio.Infrastructure.Tests;

/// <summary>The three folders, scanned (D-44).</summary>
public sealed class FolderModelStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "demosaic-models-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public void Missing_folders_are_empty_lists_not_errors()
    {
        var store = new FolderModelStore(_root);
        Assert.Empty(store.DiffusionModels());
        Assert.Empty(store.Loras());
        Assert.Empty(store.Embeddings());
    }

    [Fact]
    public void Lists_what_the_user_put_there_by_name_sorted()
    {
        Directory.CreateDirectory(Path.Combine(_root, "diffusion", "zeta-model", "unet"));
        File.WriteAllText(Path.Combine(_root, "diffusion", "zeta-model", "model_index.json"), "{}");
        Directory.CreateDirectory(Path.Combine(_root, "diffusion", "alpha-ckpt"));
        File.WriteAllText(Path.Combine(_root, "diffusion", "alpha-ckpt", "a.safetensors"), "x");
        Directory.CreateDirectory(Path.Combine(_root, "diffusion", "half-copied"));
        File.WriteAllText(Path.Combine(_root, "diffusion", "half-copied", "README.md"), "x");
        Directory.CreateDirectory(Path.Combine(_root, "lora"));
        File.WriteAllText(Path.Combine(_root, "lora", "lcm.safetensors"), "x");
        File.WriteAllText(Path.Combine(_root, "lora", "notes.txt"), "x");
        Directory.CreateDirectory(Path.Combine(_root, "embeddings"));
        File.WriteAllText(Path.Combine(_root, "embeddings", "style.pt"), "x");

        var store = new FolderModelStore(_root);

        Assert.Equal(["alpha-ckpt", "zeta-model"], store.DiffusionModels());
        Assert.Equal(["lcm"], store.Loras());
        Assert.Equal(["style"], store.Embeddings());
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) { Directory.Delete(_root, recursive: true); } } catch (IOException) { }
    }
}
