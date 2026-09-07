#ifndef SIRIUS_APP_PANELS_LOG_PANEL_HPP
#define SIRIUS_APP_PANELS_LOG_PANEL_HPP

// The session log as a dock: everything Workbench::logLine records -- refused
// edits during a run, plugin errors, worker output, "no flagged labels" --
// where before only the last line flashed in the status bar for four seconds.
//
// Monospace, selectable, copy and clear, and an auto-scroll that stops
// following as soon as the reader scrolls up (a running step logs steadily,
// and a log that jumps away mid-read is unreadable).

#include <memory>

#include <QWidget>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class LogPanel : public QWidget {
        Q_OBJECT
    public:
        explicit LogPanel(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~LogPanel() override;

        // Scrolls to the newest line and resumes following (the status bar's
        // log line opens the dock this way).
        void showLatest();
        int lineCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_PANELS_LOG_PANEL_HPP
