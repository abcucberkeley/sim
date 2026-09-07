#ifndef SIRIUS_APP_MAIN_WINDOW_HPP
#define SIRIUS_APP_MAIN_WINDOW_HPP

// The window shell of docs/design: title/menu bar (brand, seven menus,
// dataset · GPU readout, ✦ Assistant), the Operations / Parameters /
// Diagnostics / Assistant docks around the viewer, the status bar, every
// menu action and keyboard shortcut, layout persistence, and the file
// dialogs (open, export, preferences). All state lives in the Workbench;
// this class only routes.

#include <memory>

#include <QMainWindow>
#include <QString>
#include <QStringList>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class ViewerWidget;

    class MainWindow : public QMainWindow {
        Q_OBJECT
    public:
        explicit MainWindow(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~MainWindow() override;

        // Command-line conveniences; failures land in the log.
        void openDatasetPath(const QString& path);
        ViewerWidget& viewer();   // scripting hooks (--stroke, --wheel)
        void openPipelinePath(const QString& path);
        void runAll();
        // Shows the assistant dock and submits `text` (scripting).
        void askAssistant(const QString& text);

    protected:
        void closeEvent(QCloseEvent* event) override;

    private:
        void refreshAllLater();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_MAIN_WINDOW_HPP
