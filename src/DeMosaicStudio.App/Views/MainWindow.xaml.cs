using System.Windows;
using DeMosaicStudio.App.ViewModels;
using Microsoft.Win32;

namespace DeMosaicStudio.App.Views;

/// <summary>
/// The window. Holds no rules: every button forwards to the view model, which forwards to the
/// job list, which is where the decisions live and where they are tested.
/// </summary>
public partial class MainWindow : Window
{
    /// <summary>Creates the window.</summary>
    public MainWindow() => InitializeComponent();

    private MainViewModel? ViewModel => DataContext as MainViewModel;

    private void OnAdd(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Multiselect = true,
            Filter = "Video files|*.mp4;*.mkv;*.mov;*.avi;*.webm|All files|*.*",
        };

        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        foreach (var file in dialog.FileNames)
        {
            ViewModel?.Add(file);
        }
    }

    private void OnRemove(object sender, RoutedEventArgs e) =>
        ViewModel?.Remove(Selected());

    private void OnRetry(object sender, RoutedEventArgs e)
    {
        foreach (var id in Selected())
        {
            ViewModel?.Retry(id);
        }
    }

    private async void OnStart(object sender, RoutedEventArgs e)
    {
        if (ViewModel is { } viewModel)
        {
            await viewModel.StartAsync().ConfigureAwait(true);
        }
    }

    private void OnCancel(object sender, RoutedEventArgs e) => ViewModel?.Cancel();

    private List<string> Selected() =>
        Jobs.SelectedItems.OfType<JobRow>().Select(row => row.Id).ToList();
}
