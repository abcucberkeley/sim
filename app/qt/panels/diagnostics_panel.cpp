#include "qt/panels/diagnostics_panel.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <QAbstractButton>
#include <QEvent>
#include <QFrame>
#include <QGridLayout>
#include <QCheckBox>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPainter>
#include <QPushButton>
#include <QSlider>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>

#include "qt/panels/diagnostic_cells.hpp"
#include "qt/theme.hpp"

namespace sirius::app {

    namespace {

        // A square glyph button: 1.5 px border, accent fill when active.
        class GlyphButton : public QAbstractButton {
        public:
            GlyphButton(const QString& glyph, const QString& tip, QSize size, QWidget* parent = nullptr)
                : QAbstractButton(parent), glyph_(glyph) {
                setFixedSize(size);
                setToolTip(tip);
                setCursor(Qt::PointingHandCursor);
                setCheckable(true);
                setFocusPolicy(Qt::NoFocus);
            }
            void setDimmed(bool dim) {
                dimmed_ = dim;
                update();
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                p.setRenderHint(QPainter::Antialiasing, true);
                const bool on = isChecked();
                const QColor border = on ? theme::kAccent : (underMouse() ? theme::kAccent : (dimmed_ ? theme::kNeutral400 : theme::kDivider));
                p.setPen(QPen(border, 1.5));
                p.setBrush(on ? QBrush(theme::kAccent) : Qt::NoBrush);
                p.drawRect(QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75));
                p.setPen(on ? theme::kBg : (dimmed_ ? theme::kNeutral600 : theme::kText));
                p.setFont(theme::font(glyphPx_));
                p.drawText(rect(), Qt::AlignCenter, glyph_);
            }
            void enterEvent(QEnterEvent*) override { update(); }
            void leaveEvent(QEvent*) override { update(); }

        private:
            QString glyph_;
            bool dimmed_ = false;
            int glyphPx_ = 12;
        };

        // Tab of the diagnostics header: 12 px text, 2 px accent underline + 800 when active.
        class TabButton : public QAbstractButton {
        public:
            explicit TabButton(const QString& text, QWidget* parent = nullptr) : QAbstractButton(parent) {
                setText(text);
                setCheckable(true);
                setCursor(Qt::PointingHandCursor);
                setFocusPolicy(Qt::NoFocus);
                const int w = QFontMetrics(theme::heading(12)).horizontalAdvance(text) + 20;
                setFixedSize(w, 26);
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                const bool on = isChecked();
                p.setFont(on ? theme::heading(12) : theme::font(12));
                p.setPen(underMouse() && !on ? theme::kAccent : theme::kText);
                p.drawText(rect().adjusted(0, 0, 0, -2), Qt::AlignCenter, text());
                if (on) p.fillRect(QRect(0, height() - 2, width(), 2), theme::kAccent);
            }
            void enterEvent(QEnterEvent*) override { update(); }
            void leaveEvent(QEvent*) override { update(); }
        };

        QLabel* caption(const QString& text, QWidget* parent) {
            auto* l = new QLabel(text, parent);
            l->setFont(theme::caption());
            QPalette pal = l->palette();
            pal.setColor(QPalette::WindowText, theme::kNeutral600);
            l->setPalette(pal);
            return l;
        }

        QString groupThousands(Index n) {
            QString s = QString::number(n);
            for (int i = s.size() - 3; i > 0; i -= 3) s.insert(i, QChar(0x2009));   // thin space
            return s;
        }

        QString bytesText(std::size_t bytes) {
            const double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
            if (gb >= 0.1) return QStringLiteral("%1 GB").arg(gb, 0, 'f', 1);
            return QStringLiteral("%1 MB").arg(static_cast<double>(bytes) / (1024.0 * 1024.0), 0, 'f', 0);
        }

        QPushButton* styledButton(const QString& text, bool secondary, QWidget* parent) {
            auto* b = new QPushButton(text, parent);
            b->setCursor(Qt::PointingHandCursor);
            b->setFocusPolicy(Qt::NoFocus);
            b->setFont(theme::font(theme::kSmallPx));
            if (secondary)
                b->setStyleSheet(QStringLiteral("QPushButton { border: 1.5px solid %1; border-radius: 0; padding: 5px 9px;"
                                                " background: transparent; color: %1; text-align: left; }"
                                                "QPushButton:hover { border-color: %2; color: %2; }")
                                     .arg(theme::hex(theme::kText), theme::hex(theme::kAccent)));
            else
                b->setStyleSheet(QStringLiteral("QPushButton { border: none; padding: 5px 6px; background: transparent; color: %1; }"
                                                "QPushButton:hover { color: %2; }")
                                     .arg(theme::hex(theme::kText), theme::hex(theme::kAccent)));
            return b;
        }

        struct SegTool {
            PaintTool tool;
            const char* glyph;
            const char* name;
        };
        constexpr std::array<SegTool, 8> kSegTools{{
            {PaintTool::Brush, "●", "Brush"},
            {PaintTool::Erase, "◌", "Erase"},
            {PaintTool::Fill, "▣", "Fill region"},
            {PaintTool::Pick, "⌖", "Pick label"},
            {PaintTool::Merge, "⊕", "Merge labels"},
            {PaintTool::Split, "⊘", "Split (watershed seed)"},
            {PaintTool::Delete, "✕", "Delete label"},
            {PaintTool::Lasso, "◠", "Lasso"},
        }};

        // --- segmentation cleanup ---------------------------------------------------

        class SegmentCleanupView : public QWidget {
        public:
            explicit SegmentCleanupView(WorkbenchBridge& bridge, QWidget* parent = nullptr)
                : QWidget(parent), bridge_(bridge) {
                setAutoFillBackground(true);
                QPalette pal = palette();
                pal.setColor(QPalette::Window, theme::kDivider);
                setPalette(pal);
                auto* row = new QHBoxLayout(this);
                row->setContentsMargins(0, 0, 0, 0);
                row->setSpacing(2);
                row->addWidget(buildTools(), 0);
                row->addWidget(buildTable(), 1);
                row->addWidget(buildQueue(), 0);
            }

            void refresh() {
                const ViewState& vs = bridge_.wb().viewState();
                updating_ = true;
                for (std::size_t i = 0; i < tools_.size(); ++i) tools_[i]->setChecked(kSegTools[i].tool == vs.paintTool);
                brush_->setValue(vs.brushPx);
                brushLabel_->setText(QStringLiteral("%1 px").arg(vs.brushPx));
                toolName_->setText(QString::fromUtf8(kSegTools[toolIndex(vs.paintTool)].name));
                paint3d_->setChecked(vs.paint3d);
                paint3d_->setText(QStringLiteral("Paint in 3D (±%1 z)").arg(std::max(1, static_cast<int>(std::lround(vs.brushPx / 6.0)))));
                updating_ = false;
                refreshTable();
            }

        private:
            static std::size_t toolIndex(PaintTool t) {
                for (std::size_t i = 0; i < kSegTools.size(); ++i)
                    if (kSegTools[i].tool == t) return i;
                return 0;
            }

            QWidget* cell(QWidget* content = nullptr) {
                auto* w = content ? content : new QWidget(this);
                w->setAutoFillBackground(true);
                QPalette pal = w->palette();
                pal.setColor(QPalette::Window, theme::kBg);
                w->setPalette(pal);
                return w;
            }

            QWidget* buildTools() {
                QWidget* w = cell();
                w->setFixedWidth(230);
                auto* v = new QVBoxLayout(w);
                v->setContentsMargins(14, 10, 14, 10);
                v->setSpacing(8);
                v->addWidget(caption(QStringLiteral("CLEANUP TOOLS · PAINT IN VIEWER"), w));
                auto* gridHost = new QWidget(w);
                auto* grid = new QGridLayout(gridHost);
                grid->setContentsMargins(0, 0, 0, 0);
                grid->setSpacing(2);
                for (std::size_t i = 0; i < kSegTools.size(); ++i) {
                    auto* b = new GlyphButton(QString::fromUtf8(kSegTools[i].glyph), QString::fromUtf8(kSegTools[i].name), QSize(48, 34), gridHost);
                    b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
                    b->setMinimumWidth(30);
                    b->setMaximumWidth(1000);
                    grid->addWidget(b, static_cast<int>(i / 4), static_cast<int>(i % 4));
                    QObject::connect(b, &QAbstractButton::clicked, this, [this, i] {
                        if (updating_) return;
                        bridge_.wb().setPaintTool(kSegTools[i].tool);
                        bridge_.wb().setTool(ViewerTool::Paint);
                    });
                    tools_.push_back(b);
                }
                v->addWidget(gridHost);
                auto* nameRow = new QHBoxLayout;
                toolName_ = new QLabel(w);
                toolName_->setFont(theme::font(theme::kSmallPx));
                QPalette np = toolName_->palette();
                np.setColor(QPalette::WindowText, theme::kNeutral600);
                toolName_->setPalette(np);
                brushLabel_ = new QLabel(w);
                brushLabel_->setFont(theme::font(theme::kSmallPx));
                nameRow->addWidget(toolName_);
                nameRow->addStretch(1);
                nameRow->addWidget(brushLabel_);
                v->addLayout(nameRow);
                brush_ = new QSlider(Qt::Horizontal, w);
                brush_->setRange(2, 60);
                QObject::connect(brush_, &QSlider::valueChanged, this, [this](int value) {
                    if (updating_) return;
                    ViewState vs = bridge_.wb().viewState();
                    vs.brushPx = value;
                    bridge_.wb().setViewState(vs);
                });
                v->addWidget(brush_);
                paint3d_ = new QCheckBox(QStringLiteral("Paint in 3D"), w);
                paint3d_->setFont(theme::font(theme::kSmallPx));
                QObject::connect(paint3d_, &QCheckBox::toggled, this, [this](bool on) {
                    if (updating_) return;
                    ViewState vs = bridge_.wb().viewState();
                    vs.paint3d = on;
                    bridge_.wb().setViewState(vs);
                });
                v->addWidget(paint3d_);
                v->addStretch(1);
                return w;
            }

            QWidget* buildTable() {
                table_ = new DiagnosticTableView(this);
                cell(table_);
                table_->setSelectionMode(QAbstractItemView::ExtendedSelection);
                table_->setSelectionBehavior(QAbstractItemView::SelectRows);
                table_->setColumnCount(6);
                table_->setHorizontalHeaderLabels({QStringLiteral("ID"), QStringLiteral("CLASS"), QStringLiteral("VOXELS"),
                                                   QStringLiteral("CONF."), QStringLiteral("FLAG"), QString()});
                table_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
                table_->horizontalHeader()->setStretchLastSection(true);
                QObject::connect(table_, &QTableWidget::itemSelectionChanged, this, [this] {
                    if (updating_) return;
                    const auto rows = table_->selectionModel()->selectedRows();
                    if (rows.isEmpty()) return;
                    const std::uint32_t id = table_->item(rows.first().row(), 0)->data(Qt::UserRole).toUInt();
                    ViewState vs = bridge_.wb().viewState();
                    if (vs.selectedLabel == id) return;
                    vs.selectedLabel = id;
                    bridge_.wb().setViewState(vs);
                });
                return table_;
            }

            QWidget* buildQueue() {
                QWidget* w = cell();
                w->setFixedWidth(280);
                auto* v = new QVBoxLayout(w);
                v->setContentsMargins(14, 10, 14, 10);
                v->setSpacing(6);
                v->addWidget(caption(QStringLiteral("REVIEW QUEUE"), w));
                queue_ = new FactsView(w);
                v->addWidget(queue_, 1);
                auto* buttons = new QHBoxLayout;
                buttons->setSpacing(6);
                auto* next = styledButton(QStringLiteral("Next flagged →"), true, w);
                QObject::connect(next, &QPushButton::clicked, this, [this] { bridge_.wb().nextFlaggedLabel(true); });
                auto* undo = styledButton(QStringLiteral("Undo"), false, w);
                QObject::connect(undo, &QPushButton::clicked, this, [this] { bridge_.wb().undo(); });
                auto* accept = styledButton(QStringLiteral("Accept all reviewed"), false, w);
                QObject::connect(accept, &QPushButton::clicked, this, [this] { bridge_.wb().acceptAllReviewed(); });
                buttons->addWidget(next);
                buttons->addWidget(undo);
                buttons->addWidget(accept);
                buttons->addStretch(1);
                v->addLayout(buttons);
                return w;
            }

            void refreshTable() {
                updating_ = true;
                std::shared_ptr<LabelVolume> labels = bridge_.wb().viewedLabels();
                const ViewState& vs = bridge_.wb().viewState();
                table_->setRowCount(0);
                std::vector<DiagnosticFact> facts;
                if (!labels || labels->stats().empty()) {
                    facts.push_back({"Labels", "none yet"});
                    queue_->setFacts(facts, {}, QStringLiteral("Run the segmentation step to fill the review queue."));
                    updating_ = false;
                    return;
                }
                const auto& stats = labels->stats();
                table_->setRowCount(static_cast<int>(stats.size()));
                const QFont bold = theme::heading(theme::kSmallPx);
                int selectedRow = -1;
                for (int r = 0; r < static_cast<int>(stats.size()); ++r) {
                    const LabelStats& s = stats[static_cast<std::size_t>(r)];
                    auto* id = new QTableWidgetItem(QStringLiteral("%1").arg(s.id, 4, 10, QChar('0')));
                    id->setData(Qt::UserRole, static_cast<uint>(s.id));
                    QPixmap chip(10, 10);
                    const auto c = labelColor(s.id);
                    chip.fill(QColor::fromRgbF(c[0], c[1], c[2]));
                    id->setIcon(QIcon(chip));
                    table_->setItem(r, 0, id);
                    table_->setItem(r, 1, new QTableWidgetItem(QString::fromStdString(s.cls)));
                    table_->setItem(r, 2, new QTableWidgetItem(groupThousands(s.voxels)));
                    auto* conf = new QTableWidgetItem(QString::number(s.confidence, 'f', 2));
                    if (s.confidence < 0.6) conf->setForeground(theme::kAccent);
                    table_->setItem(r, 3, conf);
                    auto* flag = new QTableWidgetItem(QString::fromStdString(s.flagText()));
                    if (!s.flags.empty()) {
                        flag->setForeground(theme::kAccent);
                        flag->setFont(bold);
                    }
                    table_->setItem(r, 4, flag);
                    auto* actions = new QLabel(QStringLiteral("<a href=\"merge\" style=\"color:%1;text-decoration:none\">merge</a> · "
                                                              "<a href=\"split\" style=\"color:%1;text-decoration:none\">split</a> · "
                                                              "<a href=\"delete\" style=\"color:%1;text-decoration:none\">✕</a>")
                                                   .arg(theme::hex(theme::kAccent)));
                    actions->setFont(theme::font(theme::kSmallPx));
                    actions->setContentsMargins(6, 0, 6, 0);
                    const std::uint32_t labelId = s.id;
                    QObject::connect(actions, &QLabel::linkActivated, this, [this, labelId](const QString& link) { act(link, labelId); });
                    table_->setCellWidget(r, 5, actions);
                    if (s.id == vs.selectedLabel) selectedRow = r;
                }
                if (selectedRow >= 0) table_->selectRow(selectedRow);
                // review queue
                facts.push_back({"Low confidence (< 0.6)", std::to_string(labels->flaggedCount("low conf"))});
                facts.push_back({"Touching border", std::to_string(labels->flaggedCount("touching border"))});
                facts.push_back({"Size outliers", std::to_string(labels->flaggedCount("small") + labels->flaggedCount("merged?"))});
                facts.push_back({"Reviewed", std::to_string(labels->reviewedCount()) + " / " + std::to_string(stats.size())});
                queue_->setFacts(facts);
                updating_ = false;
            }

            void act(const QString& link, std::uint32_t id) {
                Workbench& wb = bridge_.wb();
                if (link == QLatin1String("delete")) {
                    wb.deleteLabel(id);
                } else if (link == QLatin1String("merge")) {
                    std::vector<std::uint32_t> ids{id};
                    for (const QModelIndex& row : table_->selectionModel()->selectedRows()) {
                        const std::uint32_t other = table_->item(row.row(), 0)->data(Qt::UserRole).toUInt();
                        if (other != id) ids.push_back(other);
                    }
                    if (ids.size() < 2 && wb.viewState().selectedLabel != 0 && wb.viewState().selectedLabel != id)
                        ids.push_back(wb.viewState().selectedLabel);
                    if (ids.size() >= 2) wb.mergeLabels(ids);
                    else bridge_.wb().logLine("Merge: select another label row first.");
                } else if (link == QLatin1String("split")) {
                    std::shared_ptr<LabelVolume> labels = wb.viewedLabels();
                    const LabelStats* s = labels ? labels->statsOf(id) : nullptr;
                    if (!s) return;
                    // two seeds at a quarter and three quarters of the longest bbox axis
                    const std::array<Index, 3> extent{s->bbox[1] - s->bbox[0], s->bbox[3] - s->bbox[2], s->bbox[5] - s->bbox[4]};
                    const int axis = static_cast<int>(std::max_element(extent.begin(), extent.end()) - extent.begin());
                    std::array<Index, 3> centre{(s->bbox[0] + s->bbox[1]) / 2, (s->bbox[2] + s->bbox[3]) / 2, (s->bbox[4] + s->bbox[5]) / 2};
                    std::array<Index, 3> a = centre, b = centre;
                    const std::size_t ax = static_cast<std::size_t>(axis);
                    a[ax] = s->bbox[2 * ax] + extent[ax] / 4;
                    b[ax] = s->bbox[2 * ax] + (3 * extent[ax]) / 4;
                    wb.splitLabel(id, a, b);
                }
            }

            WorkbenchBridge& bridge_;
            std::vector<GlyphButton*> tools_;
            QLabel* toolName_ = nullptr;
            QLabel* brushLabel_ = nullptr;
            QSlider* brush_ = nullptr;
            QCheckBox* paint3d_ = nullptr;
            DiagnosticTableView* table_ = nullptr;
            FactsView* queue_ = nullptr;
            bool updating_ = false;
        };

    } // namespace

    // --- DiagnosticsPanel -----------------------------------------------------------

    struct DiagnosticsPanel::Impl {
        DiagnosticsPanel* self = nullptr;
        WorkbenchBridge& bridge;
        QDockWidget* dock = nullptr;
        bool collapsed = false;
        bool maximized = false;
        int tab = 0;
        int expandedHeight = theme::kDiagnosticsH;

        QWidget* header = nullptr;
        QLabel* toggle = nullptr;
        QLabel* captionLabel = nullptr;
        QWidget* tabRow = nullptr;
        QHBoxLayout* tabLayout = nullptr;
        std::vector<TabButton*> tabs;
        QLabel* hint = nullptr;
        GlyphButton* dockBtn = nullptr;
        GlyphButton* floatBtn = nullptr;
        GlyphButton* maxBtn = nullptr;
        QStackedWidget* stack = nullptr;
        DiagnosticsBody* body = nullptr;
        SegmentCleanupView* segment = nullptr;
        QTimer refreshTimer;
        QFrame* rule = nullptr;

        explicit Impl(WorkbenchBridge& b) : bridge(b) {}

        void buildHeader() {
            header = new QWidget(self);
            header->setFixedHeight(theme::kDiagnosticsHeaderH);
            auto* h = new QHBoxLayout(header);
            h->setContentsMargins(14, 0, 14, 0);
            h->setSpacing(16);
            auto* left = new QWidget(header);
            left->setCursor(Qt::PointingHandCursor);
            auto* lh = new QHBoxLayout(left);
            lh->setContentsMargins(0, 0, 0, 0);
            lh->setSpacing(8);
            toggle = new QLabel(QStringLiteral("▼"), left);
            toggle->setFont(theme::font(theme::kCaptionPx));
            QPalette tp = toggle->palette();
            tp.setColor(QPalette::WindowText, theme::kAccent);
            toggle->setPalette(tp);
            captionLabel = caption(QStringLiteral("DIAGNOSTICS"), left);
            lh->addWidget(toggle);
            lh->addWidget(captionLabel);
            left->installEventFilter(self);
            h->addWidget(left);
            tabRow = new QWidget(header);
            tabLayout = new QHBoxLayout(tabRow);
            tabLayout->setContentsMargins(0, 0, 0, 0);
            tabLayout->setSpacing(2);
            h->addWidget(tabRow);
            // The header never dictates the dock's minimum width: a long tab
            // row or hint clips instead of widening the whole window.
            header->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
            hint = new QLabel(header);
            hint->setFont(theme::font(theme::kSmallPx));
            QPalette hp = hint->palette();
            hp.setColor(QPalette::WindowText, theme::kNeutral600);
            hint->setPalette(hp);
            hint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
            hint->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
            h->addWidget(hint, 1);
            auto* modes = new QHBoxLayout;
            modes->setSpacing(2);
            dockBtn = new GlyphButton(QStringLiteral("▁"), QStringLiteral("Dock to bottom"), QSize(24, 22), header);
            floatBtn = new GlyphButton(QStringLiteral("❐"), QStringLiteral("Undock as floating window"), QSize(24, 22), header);
            maxBtn = new GlyphButton(QStringLiteral("⛶"), QStringLiteral("Maximize over viewer"), QSize(24, 22), header);
            for (GlyphButton* b : {dockBtn, floatBtn, maxBtn}) {
                b->setDimmed(true);
                modes->addWidget(b);
            }
            h->addLayout(modes);
            QObject::connect(dockBtn, &QAbstractButton::clicked, self, [this] {
                self->setMaximized(false);
                if (dock) dock->setFloating(false);
                self->setCollapsed(false);
                updateModes();
            });
            QObject::connect(floatBtn, &QAbstractButton::clicked, self, [this] {
                self->setMaximized(false);
                self->setCollapsed(false);
                if (dock) dock->setFloating(true);
                updateModes();
            });
            QObject::connect(maxBtn, &QAbstractButton::clicked, self, [this] {
                self->setCollapsed(false);
                if (dock && dock->isFloating()) dock->setFloating(false);
                self->setMaximized(!maximized);
            });
        }

        void updateModes() {
            const bool floating = dock && dock->isFloating();
            dockBtn->setChecked(!floating && !maximized);
            floatBtn->setChecked(floating);
            maxBtn->setChecked(maximized);
            self->update();
        }

        void rebuildTabs(const QStringList& names) {
            // Detach the old buttons now: deleteLater() alone leaves them in
            // the layout until the event loop runs, and several refreshes
            // before the first paint would stack every generation of tabs
            // into the header's minimum width (which the window then adopts).
            for (TabButton* t : tabs) {
                tabLayout->removeWidget(t);
                t->hide();
                t->deleteLater();
            }
            tabs.clear();
            for (int i = 0; i < names.size(); ++i) {
                auto* t = new TabButton(names[i], tabRow);
                t->setChecked(i == tab);
                QObject::connect(t, &QAbstractButton::clicked, self, [this, i] { self->setTab(i); });
                tabLayout->addWidget(t);
                tabs.push_back(t);
            }
            tabRow->setVisible(!collapsed && names.size() > 1);
        }

        void refresh() {
            const Workbench& wb = bridge.wb();
            const int sel = wb.selectedIndex();
            const Pipeline& p = wb.pipeline();
            if (sel < 0 || sel >= p.size()) {
                captionLabel->setText(QStringLiteral("DIAGNOSTICS"));
                rebuildTabs({});
                body->setDiagnostics(Diagnostics{}, DiagnosticsKind::Generic, 0, {});
                stack->setCurrentWidget(body);
                return;
            }
            const Step& step = p.at(sel);
            captionLabel->setText(QStringLiteral("DIAGNOSTICS · %1").arg(QString::fromStdString(step.name).toUpper()));
            const Operation* op = findOperation(step.kind);
            const DiagnosticsKind kind = op ? op->info().diagnostics : DiagnosticsKind::Generic;
            Diagnostics d;
            try {
                d = wb.selectedDiagnostics();
            } catch (const std::exception&) {
            }
            const QStringList names = DiagnosticsBody::tabNames(d, kind);
            tab = std::clamp(tab, 0, std::max(0, static_cast<int>(names.size()) - 1));
            rebuildTabs(names);
            hint->setText(collapsed ? QStringLiteral("Click to expand") : QStringLiteral("Updates live as parameters change"));
            if (collapsed) return;
            if (kind == DiagnosticsKind::Segment) {
                segment->refresh();
                stack->setCurrentWidget(segment);
                return;
            }
            DiagnosticsBody::Context ctx;
            try {
                ctx.stepSummary = QString::fromStdString(wb.stepSummary(sel));
                ctx.inputShape = QString::fromStdString(wb.inputMetaOf(sel).shapeString());
                ctx.outputShape = QString::fromStdString(wb.outputMetaOf(sel).shapeString());
                ctx.estimate = QStringLiteral("≈ %1 output · cache %2")
                                   .arg(bytesText(wb.estimatedBytesOf(sel)), QString::fromUtf8(toString(step.cache)));
            } catch (const std::exception& e) {
                ctx.stepSummary = QString::fromUtf8(e.what());
            }
            body->setDiagnostics(d, kind, tab, ctx);
            stack->setCurrentWidget(body);
        }

        void scheduleRefresh() { refreshTimer.start(); }
    };

    DiagnosticsPanel::DiagnosticsPanel(WorkbenchBridge& bridge, QWidget* parent)
        : QWidget(parent), impl_(std::make_unique<Impl>(bridge)) {
        Impl& d = *impl_;
        d.self = this;
        setAutoFillBackground(true);
        QPalette pal = palette();
        pal.setColor(QPalette::Window, theme::kBg);
        setPalette(pal);
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(0, 0, 0, 0);
        v->setSpacing(0);
        d.buildHeader();
        v->addWidget(d.header, 0);
        d.rule = new QFrame(this);
        d.rule->setFixedHeight(theme::kHairline);
        d.rule->setAutoFillBackground(true);
        QPalette rp = d.rule->palette();
        rp.setColor(QPalette::Window, theme::kDivider);
        d.rule->setPalette(rp);
        v->addWidget(d.rule, 0);
        d.stack = new QStackedWidget(this);
        d.body = new DiagnosticsBody(d.stack);
        d.segment = new SegmentCleanupView(bridge, d.stack);
        d.stack->addWidget(d.body);
        d.stack->addWidget(d.segment);
        v->addWidget(d.stack, 1);
        setMinimumHeight(theme::kDiagnosticsHeaderH);

        d.refreshTimer.setSingleShot(true);
        d.refreshTimer.setInterval(150);
        connect(&d.refreshTimer, &QTimer::timeout, this, [this] { impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::selectionChanged, this, [this] { impl_->tab = 0; impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::outputsChanged, this, [this] { impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, [this] { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::stepChanged, this, [this](int) { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::datasetChanged, this, [this] { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::labelsChanged, this, [this](quint64) { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::viewStateChanged, this, [this] {
            if (impl_->stack->currentWidget() == impl_->segment) impl_->segment->refresh();
        });
        connect(&bridge, &WorkbenchBridge::runFinished, this, [this](bool, const QString&) { impl_->refresh(); });
        d.updateModes();
        d.refresh();
    }

    DiagnosticsPanel::~DiagnosticsPanel() = default;

    void DiagnosticsPanel::setDock(QDockWidget* dock) {
        impl_->dock = dock;
        if (dock) {
            connect(dock, &QDockWidget::topLevelChanged, this, [this](bool) {
                impl_->updateModes();
                update();
            });
        }
        impl_->updateModes();
    }

    bool DiagnosticsPanel::isCollapsed() const { return impl_->collapsed; }

    void DiagnosticsPanel::setCollapsed(bool collapsed) {
        Impl& d = *impl_;
        if (d.collapsed == collapsed) return;
        d.collapsed = collapsed;
        d.toggle->setText(collapsed ? QStringLiteral("▶") : QStringLiteral("▼"));
        if (collapsed) {
            d.expandedHeight = std::max(height(), theme::kDiagnosticsHeaderH + 60);
            d.stack->hide();
            d.rule->hide();
            d.tabRow->hide();
            setMaximumHeight(theme::kDiagnosticsHeaderH);
            setMinimumHeight(theme::kDiagnosticsHeaderH);
        } else {
            setMaximumHeight(QWIDGETSIZE_MAX);
            setMinimumHeight(theme::kDiagnosticsHeaderH + 40);
            d.stack->show();
            d.rule->show();
            resize(width(), d.expandedHeight);
        }
        d.refresh();
        emit collapsedChanged(collapsed);
    }

    void DiagnosticsPanel::setMaximized(bool on) {
        Impl& d = *impl_;
        if (d.maximized == on) return;
        d.maximized = on;
        d.updateModes();
        emit maximizedChanged(on);
    }

    bool DiagnosticsPanel::isMaximized() const { return impl_->maximized; }

    void DiagnosticsPanel::setTab(int index) {
        Impl& d = *impl_;
        d.tab = std::max(0, index);
        for (std::size_t i = 0; i < d.tabs.size(); ++i) d.tabs[i]->setChecked(static_cast<int>(i) == d.tab);
        d.refresh();
    }

    int DiagnosticsPanel::tabCount() const { return static_cast<int>(impl_->tabs.size()); }

    bool DiagnosticsPanel::eventFilter(QObject* watched, QEvent* event) {
        if (event->type() == QEvent::MouseButtonRelease && watched == impl_->toggle->parentWidget()) {
            setCollapsed(!impl_->collapsed);
            return true;
        }
        return QWidget::eventFilter(watched, event);
    }

    void DiagnosticsPanel::paintEvent(QPaintEvent* event) {
        QWidget::paintEvent(event);
        QPainter p(this);
        // 2 px rule on top (the region divider); an ink border when floating
        p.fillRect(QRect(0, 0, width(), theme::kRule), theme::kDivider);
        if (impl_->dock && impl_->dock->isFloating()) {
            p.setPen(QPen(theme::kText, 2));
            p.drawRect(rect().adjusted(1, 1, -1, -1));
        }
    }

} // namespace sirius::app
