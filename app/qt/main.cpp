// sirius-app: Qt front end for the SIM reconstruction library.
//
//   sirius-app [--raw stack.tif] [--otf otf.tif] [--params config.toml] [--reconstruct]
//
// Any of the inputs may also be opened from the File menu; --reconstruct
// starts a run as soon as the window is up (with the default device).

#include <QApplication>
#include <QCommandLineParser>
#include <QTimer>

#include "qt/main_window.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("sirius-app"));
    QCoreApplication::setOrganizationName(QStringLiteral("sirius"));

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("Structured illumination microscopy reconstruction"));
    parser.addHelpOption();
    const QCommandLineOption rawOpt(QStringLiteral("raw"), QStringLiteral("Raw SIM stack (TIFF)"),
                                    QStringLiteral("file"));
    const QCommandLineOption otfOpt(QStringLiteral("otf"), QStringLiteral("Radially averaged OTF (TIFF)"),
                                    QStringLiteral("file"));
    const QCommandLineOption paramsOpt(QStringLiteral("params"),
                                       QStringLiteral("Parameter file (TOML or legacy cudasirecon)"),
                                       QStringLiteral("file"));
    const QCommandLineOption runOpt(QStringLiteral("reconstruct"),
                                    QStringLiteral("Start a reconstruction once the inputs are loaded"));
    parser.addOptions({rawOpt, otfOpt, paramsOpt, runOpt});
    parser.process(app);

    sirius::app::MainWindow window;
    // parameters first so the raw stack is validated against them
    if (parser.isSet(paramsOpt)) window.openParameters(parser.value(paramsOpt));
    if (parser.isSet(otfOpt)) window.openOtf(parser.value(otfOpt));
    if (parser.isSet(rawOpt)) window.openRaw(parser.value(rawOpt));
    window.show();
    // queued so the window is painted before the (possibly long) run starts
    if (parser.isSet(runOpt)) QTimer::singleShot(0, &window, &sirius::app::MainWindow::startReconstruction);
    return QApplication::exec();
}
