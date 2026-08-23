using System.Windows;
using System.Windows.Input;
using DeMosaicStudio.App.ViewModels;
using DeMosaicStudio.Application.Jobs;
using Microsoft.Win32;

namespace DeMosaicStudio.App.Views;

/// <summary>
/// The window. Holds no rules: every button forwards to the view model, which forwards to the
/// job list, which is where the decisions live and where they are tested.
/// </summary>
public partial class MainWindow : Window
{
    /// <summary>Creates the window.</summary>
    public MainWindow()
    {
        InitializeComponent();

        InputBindings.Add(new KeyBinding(new Relay(_ => OnAdd(this, new RoutedEventArgs())),
            Key.O, ModifierKeys.Control));
        InputBindings.Add(new KeyBinding(new Relay(_ => OnRemove(this, new RoutedEventArgs())),
            Key.Delete, ModifierKeys.None));
    }

    private MainViewModel? ViewModel => DataContext as MainViewModel;

    private void OnAdd(object sender, RoutedEventArgs e)
    {
        // The filter's video list and the one drag-and-drop accepts are the same list, stated once.
        var extensions = string.Join(';', VideoFiles.Extensions.Order().Select(x => "*" + x));
        var dialog = new OpenFileDialog
        {
            Multiselect = true,
            Filter = $"Video files|{extensions}|All files|*.*",
        };

        if (dialog.ShowDialog(this) == true)
        {
            ViewModel?.AddRange(dialog.FileNames);
        }
    }

    private void OnAddFolder(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Multiselect = true };
        if (dialog.ShowDialog(this) == true)
        {
            ViewModel?.AddRange(dialog.FolderNames);
        }
    }

    /// <summary>
    /// Says whether a drop would be accepted, and shows the right cursor for the answer.
    /// </summary>
    /// <remarks>
    /// <c>Handled</c> must be set either way. Left unhandled the event bubbles and the window ends
    /// up showing the "no entry" cursor over a drop it is perfectly willing to take — which is what
    /// "drag and drop does not work" looks like even when the drop itself would have worked.
    /// </remarks>
    private void OnDragOver(object sender, DragEventArgs e)
    {
        ArgumentNullException.ThrowIfNull(e);

        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop)
            ? DragDropEffects.Copy
            : DragDropEffects.None;
        e.Handled = true;
    }

    private void OnDrop(object sender, DragEventArgs e)
    {
        ArgumentNullException.ThrowIfNull(e);

        if (e.Data.GetData(DataFormats.FileDrop) is string[] paths)
        {
            ViewModel?.AddRange(paths);
        }

        e.Handled = true;
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

    private void OnStop(object sender, RoutedEventArgs e) => ViewModel?.Stop();

    private void OnExit(object sender, RoutedEventArgs e) => Close();

    private void OnSettings(object sender, RoutedEventArgs e)
    {
        if (ViewModel is not { } viewModel)
        {
            return;
        }

        var dialog = new SettingsWindow(viewModel.Settings) { Owner = this };
        if (dialog.ShowDialog() == true)
        {
            viewModel.Apply(dialog.Model.ToSettings());
        }
    }

    private List<string> Selected() =>
        Jobs.SelectedItems.OfType<JobRow>().Select(row => row.Id).ToList();

    /// <summary>A command that runs a delegate, for the keyboard shortcuts.</summary>
    /// <remarks>
    /// The menu items and the buttons already carry their own enablement through bindings; these
    /// exist so the accelerators shown next to them are not a lie.
    /// </remarks>
    private sealed class Relay(Action<object?> execute) : ICommand
    {
        public event EventHandler? CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }

        public bool CanExecute(object? parameter) => true;

        public void Execute(object? parameter) => execute(parameter);
    }
}
