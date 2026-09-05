using DeMosaicStudio.Application.Settings;

namespace DeMosaicStudio.Infrastructure.Settings;

/// <summary>The model store as three folders beside the worker. D-44.</summary>
/// <remarks>
/// Scans on every call rather than caching: the user adds a file, reopens the dialog, and
/// expects to see it. A folder that does not exist yet is an empty list, not an error - the
/// dialog says where the folders are.
/// </remarks>
public sealed class FolderModelStore : IModelStore
{
    /// <summary>Creates a store rooted at <paramref name="root"/> (the <c>models</c> directory).</summary>
    public FolderModelStore(string root)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        Root = root;
    }

    /// <inheritdoc />
    public string Root { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> DiffusionModels()
    {
        var directory = Path.Combine(Root, "diffusion");
        if (!Directory.Exists(directory))
        {
            return [];
        }

        return Directory.EnumerateDirectories(directory)
            .Where(d => ModelStoreRules.IsPipelineDirectory(
                Directory.EnumerateFiles(d, "*", SearchOption.AllDirectories).Select(Path.GetFileName)!))
            .Select(d => Path.GetFileName(d)!)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <inheritdoc />
    public IReadOnlyList<string> Loras() => Files("lora");

    /// <inheritdoc />
    public IReadOnlyList<string> Embeddings() => Files("embeddings");

    private List<string> Files(string folder)
    {
        var directory = Path.Combine(Root, folder);
        if (!Directory.Exists(directory))
        {
            return [];
        }

        return Directory.EnumerateFiles(directory)
            .Where(f => ModelStoreRules.IsWeightFile(Path.GetFileName(f)))
            .Select(f => Path.GetFileNameWithoutExtension(f)!)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }
}
