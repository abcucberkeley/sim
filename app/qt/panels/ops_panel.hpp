#ifndef SIRIUS_APP_OPS_PANEL_HPP
#define SIRIUS_APP_OPS_PANEL_HPP

// Left dock: "OPERATIONS · ANY ORDER" header, the step rows (enable box /
// pin, name + kind label, cache glyph + summary, ▲▼, ◉), the "Add a
// processing step" row with its grouped dropdown, the legend and the
// Run all / Export footer.

#include <QWidget>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class OpsPanel : public QWidget {
        Q_OBJECT
    public:
        explicit OpsPanel(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~OpsPanel() override;

        void openAddMenu();                 // Process ▸ Add operation…
        void refresh();                     // rebuild the rows from the workbench
        // The design's 290 px, as a preference rather than a floor: the dock
        // can be narrowed well past it (the rows elide) so the window can be
        // rebalanced.
        QSize sizeHint() const override;

    signals:
        void exportRequested();
        void managePluginsRequested();      // "Manage user operations…" in the add menu
        // The window owns removal: it warns about a discarded cache and points
        // at Undo afterwards.
        void removeStepRequested(int index);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_PANEL_HPP
