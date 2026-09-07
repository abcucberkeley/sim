#ifndef SIRIUS_APP_PARAMS_PANEL_HPP
#define SIRIUS_APP_PARAMS_PANEL_HPP

// Right dock: "STEP 05 · INTENSITY" kicker + state, step name + ? help
// button, the per-kind parameter body (generic form from ParamSpecs plus
// bespoke editors for Load, SIM, Einsum, Segmentation and Merge),
// the BACKEND and CACHE OUTPUT tile rows and the Run step / View / Remove footer.

#include <QWidget>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class ParamsPanel : public QWidget {
        Q_OBJECT
    public:
        explicit ParamsPanel(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~ParamsPanel() override;

        void setHelpOpen(bool open);        // accent-fills the ? button
        // The design's 320 px as a preference, not a floor (see OpsPanel).
        QSize sizeHint() const override;

    signals:
        void helpRequested(bool open);      // ? clicked (toggle)
        void removeStepRequested(int index);
        void runStepRequested(int index);   // "Run step": the window checks the input first

    protected:
        bool eventFilter(QObject* watched, QEvent* event) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_PARAMS_PANEL_HPP
