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

class QMimeData;

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
        // Scripting: act as though these paths were dropped on the window.
        void dropPaths(const QStringList& paths);
        void openPipelinePath(const QString& path);
        void runAll();
        // Shows the assistant dock and submits `text` (scripting).
        void askAssistant(const QString& text);
        // Unattended mode: --screenshot, --quit-after and the scripting flags
        // run with nobody at the keyboard, so closing must not raise the
        // "a run is in progress" confirmation -- it would wait for an answer
        // that never comes. The run is cancelled and the window closes.
        void setUnattended(bool on);

    protected:
        void closeEvent(QCloseEvent* event) override;
        // Files dropped anywhere on the window: a dataset opens, a folder goes
        // through the manifest dialog, a pipeline loads, a Python file is added
        // as a user operation.
        void dragEnterEvent(QDragEnterEvent* event) override;
        void dragMoveEvent(QDragMoveEvent* event) override;
        void dropEvent(QDropEvent* event) override;
        // The drop itself only accepts and defers: opening reads files and can
        // raise dialogs, neither of which belongs inside the drop callback.
        bool canAcceptDrop(const QMimeData* mime) const;
        void openDroppedPaths(const QStringList& paths);
        // The status bar's log line opens the log dock.
        bool eventFilter(QObject* watched, QEvent* event) override;

    private:
        void refreshAllLater();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_MAIN_WINDOW_HPP
