#ifndef SIRIUS_APP_HELP_WINDOW_HPP
#define SIRIUS_APP_HELP_WINDOW_HPP

// Floating help page (520 x <= 760): "HELP · <step>", Edit page, ✕; the
// operation's Markdown + LaTeX page rendered into a QTextBrowser.

#include <string>

#include <QTextBrowser>
#include <QWidget>

#include "core/help_pages.hpp"
#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    // The rendered page alone (title, intro, display formula, figure slot,
    // parameter table, note, any further sections). Knows no workbench, so
    // it can be embedded elsewhere and exercised offscreen.
    class HelpView : public QTextBrowser {
        Q_OBJECT
    public:
        explicit HelpView(QWidget* parent = nullptr);
        void setPage(const HelpPage& page);
        const HelpPage& page() const noexcept { return page_; }

    private:
        HelpPage page_;
    };

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
        bool eventFilter(QObject* watched, QEvent* event) override;
        void paintEvent(QPaintEvent* event) override;
        void dragEnterEvent(QDragEnterEvent* event) override;
        void dropEvent(QDropEvent* event) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_HELP_WINDOW_HPP
