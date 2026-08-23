using System.Windows;
using DeMosaicStudio.App.ViewModels;
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
    public SettingsWindow(JobSettings settings)
    {
        InitializeComponent();
        Model = new SettingsViewModel(settings);
        DataContext = Model;
    }

    /// <summary>The editable copy. Read <see cref="SettingsViewModel.ToSettings"/> after OK.</summary>
    public SettingsViewModel Model { get; }

    private void OnAccept(object sender, RoutedEventArgs e) => DialogResult = true;

    private void OnCancel(object sender, RoutedEventArgs e) => DialogResult = false;

    // Deliberately does not close: restoring defaults is an edit, and the user should see what it
    // did before committing to it.
    private void OnRestoreDefaults(object sender, RoutedEventArgs e) => Model.RestoreDefaults();
}
