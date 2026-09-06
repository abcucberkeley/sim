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
#include <QDir>
#include <QStandardPaths>
#include <QTimer>

#include "core/operation.hpp"
#include "core/workbench.hpp"
#include "qt/dialogs/preferences_dialog.hpp"
#include "qt/main_window.hpp"
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
    parser.addOptions({datasetOpt, pipelineOpt, runOpt});
    parser.process(app);

    sirius::app::registerBuiltinOperations();

    // Per-process scratch for the disk cache and worker files.
    const QString tmp = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    const QString scratch = QDir(tmp).filePath(QStringLiteral("sirius-%1").arg(QCoreApplication::applicationPid()));
    QDir().mkpath(scratch);

    sirius::app::Workbench workbench(std::filesystem::path(sirius::app::toStd(scratch)));
    sirius::app::PreferencesDialog::applyStored(workbench);
    sirius::app::WorkbenchBridge bridge(workbench);
    sirius::app::MainWindow window(bridge);
    if (parser.isSet(pipelineOpt)) window.openPipelinePath(parser.value(pipelineOpt));
    if (parser.isSet(datasetOpt)) window.openDatasetPath(parser.value(datasetOpt));
    window.show();
    if (parser.isSet(runOpt)) QTimer::singleShot(0, &window, &sirius::app::MainWindow::runAll);
    const int rc = QApplication::exec();
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(sirius::app::toStd(scratch)), ec);
    return rc;
}
