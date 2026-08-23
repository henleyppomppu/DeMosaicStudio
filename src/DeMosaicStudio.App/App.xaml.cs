using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Markup;
using System.Windows.Threading;
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

        UseKorean();

        // Before anything can throw. An exception reaching the message loop ends the process, and
        // a desktop application that vanishes tells the user nothing about why.
        DispatcherUnhandledException += OnUnhandled;

        _engine = new WorkerProcessEngine(Locate());

        var viewModel = new MainViewModel(_engine, Dispatcher, JsonSettingsStore.Default());
        var window = new MainWindow { DataContext = viewModel };

        MainWindow = window;
        window.Show();

        // After the window is up: a failure to start the engine is a line in the status bar, not a
        // dialog before anything is visible.
        await viewModel.InitialiseAsync().ConfigureAwait(true);
    }

    /// <summary>
    /// Turns an unhandled UI exception into a dialog instead of a disappearing window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Not a way of ignoring bugs.</b> The application keeps running because closing it loses
    /// the queue as well, and a failure in one button is rarely a reason to discard the rest — but
    /// the user is told, in the same words the developer would need.
    /// </para>
    /// <para>
    /// This exists because of a real one: opening the settings dialog killed the application
    /// outright, with no message, no dialog and nothing on screen. Diagnosing it meant reading
    /// Windows Error Reporting and then driving the button from a script to capture stderr. A
    /// handler here would have put the exception on the screen the first time.
    /// </para>
    /// </remarks>
    private void OnUnhandled(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        e.Handled = true;

        // §2.3 C-6 keeps pixel data and full source paths out of logs. This is a crash report the
        // user reads on their own screen, so the exception's own text is what it needs to carry.
        MessageBox.Show(
            MainWindow,
            $"{e.Exception.GetType().Name}: {e.Exception.Message}\n\n{e.Exception.StackTrace}",
            "예상하지 못한 오류",
            MessageBoxButton.OK,
            MessageBoxImage.Error);
    }

    /// <summary>Puts the whole application into Korean, rather than whatever the machine says.</summary>
    /// <remarks>
    /// <para>
    /// Three settings, because they are three different things.
    /// <see cref="CultureInfo.CurrentUICulture"/> chooses which resources load;
    /// <see cref="CultureInfo.CurrentCulture"/> chooses how numbers and dates are written; and
    /// <c>FrameworkElement.Language</c> is what a <b>binding</b> consults, which is neither of the
    /// first two — WPF's default for it is the XML language "en-US" and it stays that way however
    /// the thread is set up.
    /// </para>
    /// <para>
    /// Set explicitly rather than left to the machine, so the window reads the same on a machine
    /// that is not Korean and so a change lives in one place. This is also the property whose
    /// resolution ended the process under invariant globalization (D-40): setting it here is not
    /// what fixed that, the opt-out in the project file is.
    /// </para>
    /// </remarks>
    private static void UseKorean()
    {
        var korean = new CultureInfo("ko-KR");
        CultureInfo.DefaultThreadCurrentCulture = korean;
        CultureInfo.DefaultThreadCurrentUICulture = korean;
        CultureInfo.CurrentCulture = korean;
        CultureInfo.CurrentUICulture = korean;

        FrameworkElement.LanguageProperty.OverrideMetadata(
            typeof(FrameworkElement),
            new FrameworkPropertyMetadata(XmlLanguage.GetLanguage(korean.IetfLanguageTag)));
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
