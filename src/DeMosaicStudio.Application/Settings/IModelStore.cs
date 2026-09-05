namespace DeMosaicStudio.Application.Settings;

/// <summary>What is on offer for the diffusion refiner (D-44): the user's own files, by name.</summary>
/// <remarks>
/// <para>
/// The application bundles no diffusion model and downloads none. The user makes the folders,
/// puts files in them, and picks from what the dialog lists. This is the listing.
/// </para>
/// <para>
/// An interface because the dialog must be constructible without a disk, and because "what
/// counts as a model" is a rule worth a test that does not need one.
/// </para>
/// </remarks>
public interface IModelStore
{
    /// <summary>Directory names under <c>models/diffusion</c> that look like a pipeline or a checkpoint.</summary>
    IReadOnlyList<string> DiffusionModels();

    /// <summary>File names (without extension) under <c>models/lora</c>.</summary>
    IReadOnlyList<string> Loras();

    /// <summary>File names (without extension) under <c>models/embeddings</c>.</summary>
    IReadOnlyList<string> Embeddings();

    /// <summary>Where the folders are, so the dialog can say where to put files.</summary>
    string Root { get; }
}

/// <summary>
/// The rules for what counts, applied to a listing supplied by the caller. Pure, so the window's
/// choices can be tested without folders.
/// </summary>
public static class ModelStoreRules
{
    /// <summary>Extensions a LoRA or embedding file may have.</summary>
    public static IReadOnlySet<string> WeightExtensions { get; } =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".safetensors", ".pt", ".bin", ".ckpt" };

    /// <summary>
    /// A directory is a diffusion model when it holds a diffusers <c>model_index.json</c> or a
    /// single checkpoint file. A directory of only configuration files is a half-copied one and
    /// is not offered.
    /// </summary>
    public static bool IsPipelineDirectory(IEnumerable<string> fileNamesInside)
    {
        ArgumentNullException.ThrowIfNull(fileNamesInside);

        var names = fileNamesInside.ToList();
        return names.Any(n => string.Equals(n, "model_index.json", StringComparison.OrdinalIgnoreCase))
            || names.Any(n => WeightExtensions.Contains(Path.GetExtension(n)));
    }

    /// <summary>A loose file counts when its extension is a weight file's.</summary>
    public static bool IsWeightFile(string fileName) =>
        !string.IsNullOrWhiteSpace(fileName) && WeightExtensions.Contains(Path.GetExtension(fileName));
}
