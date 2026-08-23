using System.IO;
using System.Windows;
using DeMosaicStudio.App.ViewModels;
using DeMosaicStudio.App.Views;
using DeMosaicStudio.Infrastructure.Engine;
using DeMosaicStudio.Infrastructure.Settings;

namespace DeMosaicStudio.App;

/// <summary>The application. Composition happens here and nowhere else.</summary>
public partial class App : System.Windows.Application, IDisposable
{
    private WorkerProcessEngine? _engine;

    /// <inheritdoc />
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        _engine = new WorkerProcessEngine(Locate());

        var viewModel = new MainViewModel(_engine, Dispatcher, JsonSettingsStore.Default());
        var window = new MainWindow { DataContext = viewModel };

        MainWindow = window;
        window.Show();

        // After the window is up: a failure to start the engine is a line in the status bar, not a
        // dialog before anything is visible.
        await viewModel.InitialiseAsync().ConfigureAwait(true);
    }

    /// <inheritdoc />
    protected override async void OnExit(ExitEventArgs e)
    {
        if (_engine is not null)
        {
            await _engine.DisposeAsync().ConfigureAwait(false);
            _engine = null;
        }

        base.OnExit(e);
    }

    /// <summary>Disposes the engine. The application owns it, so the application ends it.</summary>
    public void Dispose()
    {
        // The engine's own DisposeAsync does the polite shutdown; this is the synchronous path the
        // analyzer asks for, and it runs only if OnExit never did.
        _engine?.Dispose();
        _engine = null;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finds the worker's runtime, which is installed beside the application rather than shipped
    /// inside it: the embedded interpreter and its dependencies are about a gigabyte.
    /// </summary>
    private static WorkerLocation Locate()
    {
        var root = AppContext.BaseDirectory;

        // Development: the repository's own virtual environment, several directories up.
        var directory = new DirectoryInfo(root);
        while (directory is not null && !Directory.Exists(Path.Combine(directory.FullName, "worker")))
        {
            directory = directory.Parent;
        }

        var home = directory?.FullName ?? root;
        var bundled = Path.Combine(home, "tools", "python", "python.exe");
        var development = Path.Combine(home, ".venv", "Scripts", "python.exe");

        return new WorkerLocation
        {
            Interpreter = File.Exists(bundled) ? bundled : development,
            WorkerRoot = Path.Combine(home, "worker"),
            WorkingDirectory = home,
        };
    }
}
