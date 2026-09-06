#ifndef SIRIUS_APP_DIAGNOSTICS_PANEL_HPP
#define SIRIUS_APP_DIAGNOSTICS_PANEL_HPP

// Bottom dock content: header (▼/▶ toggle, "DIAGNOSTICS · <step>", tab row,
// hint, ▁ ❐ ⛶ controls) and the per-kind body built from the selected
// step's Diagnostics (image cells, table, curves, histograms, facts) plus
// the dedicated segmentation-cleanup and volume panels.

#include <QWidget>

#include "qt/workbench_bridge.hpp"

class QDockWidget;

namespace sirius::app {

    class DiagnosticsPanel : public QWidget {
        Q_OBJECT
    public:
        explicit DiagnosticsPanel(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~DiagnosticsPanel() override;

        // The dock hosting this panel, for the ▁ ❐ ⛶ controls and collapse.
        void setDock(QDockWidget* dock);
        bool isCollapsed() const;
        void setCollapsed(bool collapsed);
        void setMaximized(bool on);       // covers the viewer
        bool isMaximized() const;
        void setTab(int index);
        int tabCount() const;

    signals:
        void maximizedChanged(bool on);
        void collapsedChanged(bool collapsed);

    protected:
        bool eventFilter(QObject* watched, QEvent* event) override;
        void paintEvent(QPaintEvent* event) override;
        void resizeEvent(QResizeEvent* event) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_DIAGNOSTICS_PANEL_HPP
