// sirius-app: the SIRIUS microscopy workbench (docs/design/README.md).
//
//   sirius-app [--dataset stack.tif] [--pipeline steps.sirius.toml] [--run]
//
// Everything can also be opened from the File menu; --run runs every
// enabled step as soon as the window is up.

#include <filesystem>

#include <QApplication>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QAction>
#include <QDir>
#include <QDockWidget>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>
#include <QTimer>

#include "core/operation.hpp"
#include "core/tool_api.hpp"
#include "core/workbench.hpp"
#include "qt/dialogs/preferences_dialog.hpp"
#include "qt/main_window.hpp"
#include "qt/worker_launcher.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/workbench_bridge.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("sirius-app"));
    QCoreApplication::setOrganizationName(QStringLiteral("sirius"));
    QCoreApplication::setApplicationVersion(QStringLiteral("0.2"));
    sirius::app::theme::applyTheme(app);

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("SIRIUS microscopy processing workbench"));
    parser.addHelpOption();
    parser.addVersionOption();
    const QCommandLineOption datasetOpt(QStringLiteral("dataset"), QStringLiteral("Dataset to open (TIFF / OME-TIFF / zarr)"),
                                        QStringLiteral("path"));
    const QCommandLineOption pipelineOpt(QStringLiteral("pipeline"), QStringLiteral("Pipeline file (.sirius.toml)"),
                                         QStringLiteral("path"));
    const QCommandLineOption runOpt(QStringLiteral("run"), QStringLiteral("Run every enabled step once the window is up"));
    // Developer aids: grab the window to a PNG (after the run when --run is
    // given) and quit; or just quit after a delay, for headless smoke tests.
    const QCommandLineOption screenshotOpt(QStringLiteral("screenshot"),
                                           QStringLiteral("Save a screenshot of the window to <path> and quit"),
                                           QStringLiteral("path"));
    const QCommandLineOption quitAfterOpt(QStringLiteral("quit-after"), QStringLiteral("Quit after <ms> milliseconds"),
                                          QStringLiteral("ms"));
    // Scripting for smoke tests: tool calls through the assistant's typed API
    // ({"name": "set_view", "args": {"mode": "3d"}}) and menu actions by
    // their text ("Assistant"), applied once the run (if any) has finished.
    const QCommandLineOption toolOpt(QStringLiteral("tool"), QStringLiteral("Call a tool of the assistant API (JSON, repeatable)"),
                                     QStringLiteral("json"));
    const QCommandLineOption actionOpt(QStringLiteral("action"), QStringLiteral("Trigger a menu action by its text (repeatable)"),
                                       QStringLiteral("text"));
    parser.addOptions({datasetOpt, pipelineOpt, runOpt, screenshotOpt, quitAfterOpt, toolOpt, actionOpt});
    parser.process(app);

    sirius::app::registerBuiltinOperations();

    // Per-process scratch for the disk cache and worker files.
    const QString tmp = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    const QString scratch = QDir(tmp).filePath(QStringLiteral("sirius-%1").arg(QCoreApplication::applicationPid()));
    QDir().mkpath(scratch);

    sirius::app::Workbench workbench(std::filesystem::path(sirius::app::toStd(scratch)));
    sirius::app::PreferencesDialog::applyStored(workbench);
    sirius::app::WorkbenchBridge bridge(workbench);
    // Steps that need Python (Torch models) get a worker spawned on demand.
    sirius::app::WorkerLauncher launcher;
    QObject::connect(&launcher, &sirius::app::WorkerLauncher::logged, &bridge,
                     [&workbench](const QString& line) { workbench.logLine("worker: " + sirius::app::toStd(line)); });
    workbench.setLocalWorkerLauncher([&launcher] { return launcher.connect(); });
    sirius::app::MainWindow window(bridge);
    if (parser.isSet(pipelineOpt)) window.openPipelinePath(parser.value(pipelineOpt));
    if (parser.isSet(datasetOpt)) window.openDatasetPath(parser.value(datasetOpt));
    window.show();
    if (parser.isSet(runOpt)) QTimer::singleShot(0, &window, &sirius::app::MainWindow::runAll);
    const QStringList toolCalls = parser.values(toolOpt);
    const QStringList actions = parser.values(actionOpt);
    sirius::app::ToolApi tools(workbench);
    auto script = [&] {
        for (const QString& call : toolCalls) {
            const QJsonObject j = QJsonDocument::fromJson(call.toUtf8()).object();
            const nlohmann::json args = nlohmann::json::parse(QJsonDocument(j.value(QStringLiteral("args")).toObject()).toJson().constData());
            const nlohmann::json r = tools.call(sirius::app::toStd(j.value(QStringLiteral("name")).toString()), args);
            workbench.logLine("tool " + sirius::app::toStd(j.value(QStringLiteral("name")).toString()) + " → " + r.dump().substr(0, 200));
        }
        for (const QString& text : actions) {
            bool found = false;
            for (QAction* a : window.findChildren<QAction*>())
                if (a->text().remove(QLatin1Char('&')) == text || a->text().remove(QLatin1Char('&')).startsWith(text + QChar(0x2026))) {
                    a->trigger();
                    found = true;
                    break;
                }
            if (!found) workbench.logLine("no action named " + sirius::app::toStd(text));
        }
    };
    const bool scripted = !toolCalls.isEmpty() || !actions.isEmpty();
    if (parser.isSet(screenshotOpt) || scripted) {
        const QString path = parser.value(screenshotOpt);
        auto finish = [&window, &app, &script, path, scripted] {
            if (scripted) script();
            if (path.isEmpty()) return;
            QTimer::singleShot(600, &window, [&window, &app, path] {
                // size report: which widget dictates the window's minimum
                QString report = QStringLiteral("window %1x%2 min %3x%4").arg(window.width()).arg(window.height())
                                     .arg(window.minimumSizeHint().width()).arg(window.minimumSizeHint().height());
                if (QWidget* c = window.centralWidget())
                    report += QStringLiteral(" central-min %1x%2").arg(c->minimumSizeHint().width()).arg(c->minimumSizeHint().height());
                for (QDockWidget* d : window.findChildren<QDockWidget*>())
                    report += QStringLiteral(" %1-min %2x%3").arg(d->objectName()).arg(d->widget() ? d->widget()->minimumSizeHint().width() : -1).arg(d->widget() ? d->widget()->minimumSizeHint().height() : -1);
                qInfo("%s", qPrintable(report));
                window.grab().save(path);
                app.quit();
            });
        };
        if (parser.isSet(runOpt)) {
            QObject::connect(&bridge, &sirius::app::WorkbenchBridge::runFinished, &window,
                             [finish](bool, const QString&) { QTimer::singleShot(300, finish); });
            QTimer::singleShot(600000, &window, finish);   // never hang a headless run
        } else {
            QTimer::singleShot(1200, &window, finish);
        }
    }
    if (parser.isSet(quitAfterOpt)) QTimer::singleShot(parser.value(quitAfterOpt).toInt(), &app, &QApplication::quit);
    const int rc = QApplication::exec();
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(sirius::app::toStd(scratch)), ec);
    return rc;
}
