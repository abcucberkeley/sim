#include "qt/panels/log_panel.hpp"

#include <QApplication>
#include <QBoxLayout>
#include <QClipboard>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QTimer>

#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::Rule;

    struct LogPanel::Impl {
        WorkbenchBridge& bridge;
        QPlainTextEdit* view = nullptr;
        QLabel* count = nullptr;
        QPushButton* copy = nullptr;
        QPushButton* clear = nullptr;
        QPushButton* follow = nullptr;   // shown while auto-scroll is paused
        QTimer copiedTimer;
        bool following = true;

        explicit Impl(WorkbenchBridge& b) : bridge(b) {}

        bool atBottom() const {
            const QScrollBar* sb = view->verticalScrollBar();
            return sb->value() >= sb->maximum() - 2;
        }

        void toBottom() {
            QScrollBar* sb = view->verticalScrollBar();
            sb->setValue(sb->maximum());
        }

        void append(const QString& line) {
            const bool stick = following && atBottom();
            view->appendPlainText(line);
            if (stick) toBottom();
            updateCount();
        }

        void updateCount() {
            const int n = view->blockCount();
            count->setText(QStringLiteral("%1 %2").arg(n).arg(n == 1 ? QStringLiteral("line") : QStringLiteral("lines")));
            follow->setVisible(!following);
        }

        void reload() {
            view->clear();
            QString all;
            for (const std::string& line : bridge.wb().log()) {
                if (!all.isEmpty()) all += QLatin1Char('\n');
                all += fromStd(line);
            }
            view->setPlainText(all);
            following = true;
            toBottom();
            updateCount();
        }
    };

    LogPanel::LogPanel(WorkbenchBridge& bridge, QWidget* parent) : QWidget(parent), impl_(std::make_unique<Impl>(bridge)) {
        Impl& d = *impl_;
        setObjectName(QStringLiteral("Panel"));
        setAccessibleName(QStringLiteral("Log"));
        setAccessibleDescription(QStringLiteral("Everything the session has reported, newest at the bottom"));
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);

        auto* header = new QWidget(this);
        header->setFixedHeight(theme::kDiagnosticsHeaderH);
        auto* h = new QHBoxLayout(header);
        h->setContentsMargins(14, 0, 14, 0);
        h->setSpacing(10);
        h->addWidget(new CaptionLabel(QStringLiteral("Log · this session"), header));
        d.count = widgets::label(QString(), 11, theme::kNeutral600, -1, header);
        h->addWidget(d.count);
        h->addStretch(1);
        d.follow = new QPushButton(QStringLiteral("Jump to latest"), header);
        widgets::setButtonClass(d.follow, "link");
        d.follow->setCursor(Qt::PointingHandCursor);
        d.follow->setToolTip(QStringLiteral("Scroll to the newest line and keep following it"));
        d.follow->hide();
        h->addWidget(d.follow);
        d.copy = new QPushButton(QStringLiteral("Copy"), header);
        widgets::setButtonClass(d.copy, "secondary small");
        d.copy->setToolTip(QStringLiteral("Copy the whole log (or the selection) to the clipboard"));
        d.clear = new QPushButton(QStringLiteral("Clear"), header);
        widgets::setButtonClass(d.clear, "ghost small");
        d.clear->setToolTip(QStringLiteral("Empty this view; the session's own log is kept"));
        h->addWidget(d.copy);
        h->addWidget(d.clear);
        root->addWidget(header);
        root->addWidget(new Rule(theme::kHairline, Qt::Horizontal, this));

        d.view = new QPlainTextEdit(this);
        d.view->setReadOnly(true);
        d.view->setLineWrapMode(QPlainTextEdit::NoWrap);
        d.view->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
        d.view->setFont(theme::mono(12));
        d.view->setMaximumBlockCount(5000);   // the workbench keeps the same number of lines
        d.view->setAccessibleName(QStringLiteral("Session log"));
        widgets::setWidgetClass(d.view, "log");
        root->addWidget(d.view, 1);

        connect(d.copy, &QPushButton::clicked, this, [this] {
            QString text = impl_->view->textCursor().hasSelection() ? impl_->view->textCursor().selectedText()
                                                                    : impl_->view->toPlainText();
            text.replace(QChar(0x2029), QLatin1Char('\n'));   // QTextCursor's paragraph separator
            QApplication::clipboard()->setText(text);
            impl_->copy->setText(QStringLiteral("Copied"));
            impl_->copiedTimer.start(1200);
        });
        d.copiedTimer.setSingleShot(true);
        connect(&d.copiedTimer, &QTimer::timeout, this, [this] { impl_->copy->setText(QStringLiteral("Copy")); });
        connect(d.clear, &QPushButton::clicked, this, [this] {
            impl_->view->clear();
            impl_->following = true;
            impl_->updateCount();
        });
        connect(d.follow, &QPushButton::clicked, this, [this] { showLatest(); });
        // Following stops the moment the reader scrolls up, and resumes when
        // they come back to the bottom themselves.
        connect(d.view->verticalScrollBar(), &QScrollBar::valueChanged, this, [this] {
            const bool bottom = impl_->atBottom();
            if (bottom == impl_->following) return;
            impl_->following = bottom;
            impl_->updateCount();
        });
        connect(&bridge, &WorkbenchBridge::logged, this, [this](const QString& line) { impl_->append(line); });
        d.reload();
    }

    LogPanel::~LogPanel() = default;

    void LogPanel::showLatest() {
        impl_->following = true;
        impl_->toBottom();
        impl_->updateCount();
    }

    int LogPanel::lineCount() const { return impl_->view->blockCount(); }

} // namespace sirius::app
