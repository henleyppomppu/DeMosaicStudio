using System.Windows;
using DeMosaicStudio.App.ViewModels;
using DeMosaicStudio.Application.Settings;
using DeMosaicStudio.Domain.Settings;

namespace DeMosaicStudio.App.Views;

/// <summary>
/// The settings dialog.
/// <para>
/// Holds no rules: the ranges are enforced by <c>SettingsBounds</c> and the conversion back to
/// <see cref="JobSettings"/> is the view model's. This is the three buttons and nothing else.
/// </para>
/// </summary>
public partial class SettingsWindow : Window
{
    /// <summary>Creates the dialog over an editable copy of the given settings.</summary>
    public SettingsWindow(JobSettings settings, IModelStore store)
    {
        InitializeComponent();
        Model = new SettingsViewModel(settings, store);
        DataContext = Model;
    }

    /// <summary>The editable copy. Read <see cref="SettingsViewModel.ToSettings"/> after OK.</summary>
    public SettingsViewModel Model { get; }

    private void OnAccept(object sender, RoutedEventArgs e) => DialogResult = true;

    private void OnCancel(object sender, RoutedEventArgs e) => DialogResult = false;

    // Deliberately does not close: restoring defaults is an edit, and the user should see what it
    // did before committing to it.
    private void OnRestoreDefaults(object sender, RoutedEventArgs e) => Model.RestoreDefaults();

    private void OnBrowseStore(object sender, RoutedEventArgs e)
    {
        var dialog = new Microsoft.Win32.OpenFolderDialog
        {
            InitialDirectory = System.IO.Directory.Exists(Model.EffectiveStoreRoot) ? Model.EffectiveStoreRoot : null,
        };
        if (dialog.ShowDialog(this) == true)
        {
            Model.StoreRoot = dialog.FolderName;
        }
    }

    // Opens the folder in Explorer, creating the three subfolders first so the user sees where
    // each kind of file goes rather than an empty directory.
    private void OnOpenStore(object sender, RoutedEventArgs e)
    {
        var root = Model.EffectiveStoreRoot;
        try
        {
            foreach (var sub in new[] { "diffusion", "lora", "embeddings" })
            {
                System.IO.Directory.CreateDirectory(System.IO.Path.Combine(root, sub));
            }

            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("explorer.exe", $"\"{root}\"") { UseShellExecute = true });
        }
        catch (Exception exception) when (exception is System.IO.IOException or UnauthorizedAccessException or System.ComponentModel.Win32Exception)
        {
            MessageBox.Show(this, exception.Message, "폴더를 열 수 없습니다", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }
}
