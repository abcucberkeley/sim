#ifndef SIRIUS_APP_HELP_WINDOW_HPP
#define SIRIUS_APP_HELP_WINDOW_HPP

// Floating help page (520 x <= 760): "HELP · <step>", Edit page, ✕; the
// operation's Markdown + LaTeX page rendered into a QTextBrowser.

#include <QWidget>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class HelpWindow : public QWidget {
        Q_OBJECT
    public:
        explicit HelpWindow(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~HelpWindow() override;

        void showKind(const std::string& kind);   // page for an operation kind
        void showManual();                        // the general manual page
        void showShortcuts();                     // keyboard shortcuts page
        std::string currentKind() const;

    signals:
        void visibilityChanged(bool visible);

    protected:
        void showEvent(QShowEvent*) override;
        void hideEvent(QHideEvent*) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_HELP_WINDOW_HPP
