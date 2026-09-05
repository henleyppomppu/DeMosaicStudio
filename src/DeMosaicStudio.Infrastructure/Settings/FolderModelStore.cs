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
    /// <summary>Creates a store whose default root is <paramref name="defaultRoot"/> (the <c>models</c> directory beside the program).</summary>
    public FolderModelStore(string defaultRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(defaultRoot);
        DefaultRoot = defaultRoot;
    }

    /// <inheritdoc />
    public string DefaultRoot { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> DiffusionModels(string root)
    {
        var directory = Path.Combine(Resolve(root), "diffusion");
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
    public IReadOnlyList<string> Loras(string root) => Files(root, "lora");

    /// <inheritdoc />
    public IReadOnlyList<string> Embeddings(string root) => Files(root, "embeddings");

    private string Resolve(string root) => string.IsNullOrWhiteSpace(root) ? DefaultRoot : root;

    private List<string> Files(string root, string folder)
    {
        var directory = Path.Combine(Resolve(root), folder);
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
