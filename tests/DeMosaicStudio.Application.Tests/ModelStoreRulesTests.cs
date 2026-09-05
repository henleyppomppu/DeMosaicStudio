using DeMosaicStudio.Application.Settings;

namespace DeMosaicStudio.Application.Tests;

/// <summary>What counts as a model in the user's folders (D-44). Pure rules, no folders needed.</summary>
public sealed class ModelStoreRulesTests
{
    [Fact]
    public void A_diffusers_folder_counts_by_its_index_file()
    {
        Assert.True(ModelStoreRules.IsPipelineDirectory(["model_index.json", "unet/config.json"]));
    }

    [Fact]
    public void A_folder_holding_a_single_checkpoint_counts()
    {
        Assert.True(ModelStoreRules.IsPipelineDirectory(["dreamshaper_8.safetensors"]));
    }

    [Fact]
    public void A_folder_of_only_configuration_is_a_half_copy_and_does_not_count()
    {
        Assert.False(ModelStoreRules.IsPipelineDirectory(["unet/config.json", "README.md"]));
        Assert.False(ModelStoreRules.IsPipelineDirectory([]));
    }

    [Theory]
    [InlineData("lcm-lora-sdv1-5.safetensors", true)]
    [InlineData("embedding.pt", true)]
    [InlineData("weights.BIN", true)]
    [InlineData("notes.txt", false)]
    [InlineData("", false)]
    public void Weight_files_are_recognised_by_extension_case_insensitively(string name, bool expected)
    {
        Assert.Equal(expected, ModelStoreRules.IsWeightFile(name));
    }
}
