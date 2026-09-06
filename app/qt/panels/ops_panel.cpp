#include "qt/panels/ops_panel.hpp"

#include <QApplication>
#include <QBoxLayout>
#include <QCheckBox>
#include <QEvent>
#include <QFrame>
#include <QGridLayout>
#include <algorithm>

#include <QLabel>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QStyle>
#include <QVector>

#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::ClickRow;
    using widgets::GlyphButton;
    using widgets::Rule;

    namespace {

        // A 14 × 14 enable box that swallows the click so the row does not select.
        class EnableBox : public QAbstractButton {
        public:
            explicit EnableBox(QWidget* parent = nullptr) : QAbstractButton(parent) {
                setCheckable(true);
                setFixedSize(14, 14);
                setCursor(Qt::PointingHandCursor);
                setToolTip(QStringLiteral("Enable / skip"));
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                p.setPen(QPen(theme::kNeutral700, 1.5));
                p.setBrush(isChecked() ? QBrush(theme::kAccent) : Qt::NoBrush);
                p.drawRect(QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75));
            }
            void mouseReleaseEvent(QMouseEvent* e) override {
                QAbstractButton::mouseReleaseEvent(e);
                e->accept();
            }
        };

        struct CacheLook {
            QString glyph;
            QColor color;
            QString title;
        };

        CacheLook cacheLook(CachePolicy c) {
            switch (c) {
                case CachePolicy::Memory: return {QStringLiteral("M"), theme::kAccent, QStringLiteral("Cached in GPU/RAM")};
                case CachePolicy::Disk: return {QStringLiteral("D"), theme::kText, QStringLiteral("Cached on disk (zarr scratch)")};
                case CachePolicy::Recompute: break;
            }
            return {QStringLiteral("↻"), theme::kNeutral500, QStringLiteral("Recomputed on demand")};
        }

        // One step row: grid 22 | 1fr | auto.
        class StepRow : public ClickRow {
        public:
            StepRow(WorkbenchBridge& bridge, QWidget* parent) : ClickRow(parent), bridge_(bridge) {
                setEdge(true);
                auto* grid = new QGridLayout(this);
                grid->setContentsMargins(10, 9, 14, 9);
                grid->setHorizontalSpacing(10);
                grid->setVerticalSpacing(0);
                grid->setColumnMinimumWidth(0, 22);
                grid->setColumnStretch(1, 1);

                auto* leftCell = new QWidget(this);
                auto* leftLayout = new QHBoxLayout(leftCell);
                leftLayout->setContentsMargins(0, 0, 0, 0);
                leftLayout->setAlignment(Qt::AlignCenter);
                enable_ = new EnableBox(leftCell);
                pin_ = widgets::label(QStringLiteral("⬢"), 10, theme::kNeutral500, -1, leftCell);
                pin_->setToolTip(QStringLiteral("Always first, always enabled"));
                pin_->setAlignment(Qt::AlignCenter);
                leftLayout->addWidget(enable_);
                leftLayout->addWidget(pin_);
                grid->addWidget(leftCell, 0, 0, 2, 1, Qt::AlignCenter);

                body_ = new QWidget(this);
                auto* bodyLayout = new QVBoxLayout(body_);
                bodyLayout->setContentsMargins(0, 0, 0, 0);
                bodyLayout->setSpacing(0);
                auto* titleRow = new QHBoxLayout();
                titleRow->setContentsMargins(0, 0, 0, 0);
                titleRow->setSpacing(8);
                name_ = widgets::heading(QString(), 13, body_);
                name_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
                kind_ = new QLabel(body_);
                QFont kf = theme::font(10);
                kf.setLetterSpacing(QFont::PercentageSpacing, 106.0);
                kf.setCapitalization(QFont::AllUppercase);
                kind_->setFont(kf);
                QPalette kp = kind_->palette();
                kp.setColor(QPalette::WindowText, theme::kNeutral500);
                kind_->setPalette(kp);
                titleRow->addWidget(name_, 1);
                titleRow->addWidget(kind_, 0, Qt::AlignBaseline);
                bodyLayout->addLayout(titleRow);
                auto* sumRow = new QHBoxLayout();
                sumRow->setContentsMargins(0, 0, 0, 0);
                sumRow->setSpacing(6);
                cache_ = widgets::label(QString(), 11, theme::kText, QFont::ExtraBold, body_);
                summary_ = widgets::label(QString(), 11, theme::kNeutral600, -1, body_);
                summary_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
                sumRow->addWidget(cache_);
                sumRow->addWidget(summary_, 1);
                bodyLayout->addLayout(sumRow);
                grid->addWidget(body_, 0, 1, 2, 1);

                auto* right = new QWidget(this);
                auto* rightLayout = new QHBoxLayout(right);
                rightLayout->setContentsMargins(0, 0, 0, 0);
                rightLayout->setSpacing(4);
                up_ = new GlyphButton(QStringLiteral("▲"), 16, right);
                up_->setBorderless(true);
                up_->setGlyphPx(10);
                up_->setIdleColor(theme::kNeutral500);
                up_->setToolTip(QStringLiteral("Move up (⌥↑)"));
                down_ = new GlyphButton(QStringLiteral("▼"), 16, right);
                down_->setBorderless(true);
                down_->setGlyphPx(10);
                down_->setIdleColor(theme::kNeutral500);
                down_->setToolTip(QStringLiteral("Move down (⌥↓)"));
                view_ = new GlyphButton(QStringLiteral("◉"), 20, right);
                view_->setGlyphPx(10);
                view_->setIdleColor(theme::kNeutral400);
                view_->setToolTip(QStringLiteral("Show this step's output in the viewer"));
                rightLayout->addWidget(up_);
                rightLayout->addWidget(down_);
                rightLayout->addSpacing(4);
                rightLayout->addWidget(view_);
                grid->addWidget(right, 0, 2, 2, 1, Qt::AlignVCenter);

                connect(this, &ClickRow::clicked, this, [this] { bridge_.wb().select(index_); });
                connect(enable_, &QAbstractButton::clicked, this, [this](bool on) {
                    bridge_.wb().setStepEnabled(index_, on);
                });
                connect(up_, &QAbstractButton::clicked, this, [this] { bridge_.wb().moveStep(index_, -1); });
                connect(down_, &QAbstractButton::clicked, this, [this] { bridge_.wb().moveStep(index_, +1); });
                connect(view_, &QAbstractButton::clicked, this, [this] { bridge_.wb().view(index_); });
            }

            void refresh(int index) {
                index_ = index;
                const Workbench& wb = bridge_.wb();
                const Step& step = wb.pipeline().at(index);
                const bool selected = wb.selectedIndex() == index;
                const bool viewed = wb.viewedIndex() == index;
                setSelected(selected);
                enable_->setVisible(!step.pinned);
                pin_->setVisible(step.pinned);
                enable_->setChecked(step.enabled);
                fullName_ = fromStd(step.name);
                name_->setToolTip(fullName_);
                kind_->setText(fromStd(step.op().info().kindLabel));
                const CacheLook look = cacheLook(step.cache);
                cache_->setText(look.glyph);
                cache_->setToolTip(look.title);
                QPalette cp = cache_->palette();
                cp.setColor(QPalette::WindowText, look.color);
                cache_->setPalette(cp);
                QString sum = fromStd(wb.stepSummary(index));
                const Validation v = wb.stepValidation(index);
                if (!v.ok()) sum = fromStd(v.firstError());
                else if (!step.enabled) sum += QStringLiteral(" — skipped");
                fullSummary_ = sum;
                summary_->setToolTip(sum);
                applyElide();
                QPalette sp = summary_->palette();
                sp.setColor(QPalette::WindowText, v.ok() ? theme::kNeutral600 : theme::kAccent);
                summary_->setPalette(sp);
                body_->setEnabled(true);
                setBodyOpacity(step.enabled ? 1.0 : 0.45);
                up_->setVisible(!step.pinned);
                down_->setVisible(!step.pinned);
                up_->setEnabled(index > 1);
                down_->setEnabled(index < wb.pipeline().size() - 1);
                view_->setActive(viewed);
            }

        private:
            void setBodyOpacity(double op) {
                // QSS has no opacity: fade the labels' colours instead.
                auto fade = [op](QLabel* l, const QColor& base) {
                    QPalette p = l->palette();
                    QColor c = base;
                    c.setAlphaF(static_cast<float>(op));
                    p.setColor(QPalette::WindowText, c);
                    l->setPalette(p);
                };
                fade(name_, theme::kText);
                fade(kind_, theme::kNeutral500);
                if (op < 1.0) {
                    fade(summary_, theme::kNeutral600);
                    fade(cache_, cache_->palette().color(QPalette::WindowText));
                }
            }

            WorkbenchBridge& bridge_;
            int index_ = 0;
            EnableBox* enable_ = nullptr;
            QLabel* pin_ = nullptr;
            QWidget* body_ = nullptr;
            // Labels do not elide by themselves: the full texts are kept and
            // cut to the row's width on every resize (the design ellipsises both lines).
            void applyElide() {
                const int nameW = std::max(20, name_->width());
                name_->setText(widgets::elide(name_, fullName_, nameW));
                summary_->setText(widgets::elide(summary_, fullSummary_, std::max(20, summary_->width())));
            }
            void resizeEvent(QResizeEvent* e) override {
                ClickRow::resizeEvent(e);
                applyElide();
            }
            QString fullName_, fullSummary_;
            QLabel* name_ = nullptr;
            QLabel* kind_ = nullptr;
            QLabel* cache_ = nullptr;
            QLabel* summary_ = nullptr;
            GlyphButton* up_ = nullptr;
            GlyphButton* down_ = nullptr;
            GlyphButton* view_ = nullptr;
        };

        // Grouped dropdown: 82 px group caption column | items.
        class AddMenu : public QFrame {
        public:
            AddMenu(WorkbenchBridge& bridge, QWidget* parent) : QFrame(parent), bridge_(bridge) {
                setObjectName(QStringLiteral("AddMenu"));
                setAutoFillBackground(true);
                QPalette p = palette();
                p.setColor(QPalette::Window, theme::kBg);
                setPalette(p);
                auto* layout = new QVBoxLayout(this);
                layout->setContentsMargins(2, 2, 2, 2);
                layout->setSpacing(0);
                for (const auto& [group, ops] : operationGroups()) {
                    auto* row = new QWidget(this);
                    auto* grid = new QGridLayout(row);
                    grid->setContentsMargins(0, 0, 0, 0);
                    grid->setSpacing(0);
                    grid->setColumnMinimumWidth(0, 82);
                    auto* cap = new CaptionLabel(fromStd(group), row);
                    cap->setContentsMargins(10, 8, 10, 8);
                    cap->setAlignment(Qt::AlignTop | Qt::AlignLeft);
                    grid->addWidget(cap, 0, 0, Qt::AlignTop);
                    auto* items = new QWidget(row);
                    auto* itemsLayout = new QVBoxLayout(items);
                    itemsLayout->setContentsMargins(0, 0, 0, 0);
                    itemsLayout->setSpacing(0);
                    for (const Operation* op : ops) {
                        auto* item = new ClickRow(items);
                        item->setTopRule(0);
                        auto* il = new QHBoxLayout(item);
                        il->setContentsMargins(10, 6, 10, 6);
                        auto* name = widgets::heading(fromStd(op->info().name), 12, item);
                        il->addWidget(name);
                        const std::string kind = op->kind();
                        connect(item, &ClickRow::clicked, this, [this, kind] { add(kind); });
                        itemsLayout->addWidget(item);
                    }
                    grid->addWidget(items, 0, 1);
                    layout->addWidget(row);
                    layout->addWidget(new Rule(1, Qt::Horizontal, this));
                }
                auto* example = new ClickRow(this);
                example->setTopRule(0);
                auto* el = new QHBoxLayout(example);
                el->setContentsMargins(10, 8, 10, 8);
                auto* text = widgets::label(
                    QStringLiteral("Load example pipeline (SIM → einsum → contrast → merge → segment → volume)"), 11,
                    theme::kAccent, -1, example);
                text->setWordWrap(true);
                el->addWidget(text);
                connect(example, &ClickRow::clicked, this, [this] {
                    bridge_.wb().loadExamplePipeline();
                    hide();
                });
                layout->addWidget(example);
            }

        protected:
            void paintEvent(QPaintEvent* e) override {
                QFrame::paintEvent(e);
                QPainter p(this);
                p.setPen(QPen(theme::kText, 2));
                p.drawRect(rect().adjusted(1, 1, -1, -1));
            }

        private:
            void add(const std::string& kind) {
                bridge_.wb().addStep(kind);
                hide();
            }
            WorkbenchBridge& bridge_;
        };

    } // namespace

    struct OpsPanel::Impl {
        WorkbenchBridge& bridge;
        QLabel* count = nullptr;
        QWidget* rowsHost = nullptr;
        QVBoxLayout* rowsLayout = nullptr;
        QVector<StepRow*> rows;
        ClickRow* addRow = nullptr;
        GlyphButton* addGlyph = nullptr;
        QLabel* addTitle = nullptr;
        QLabel* addHint = nullptr;
        AddMenu* addMenu = nullptr;
        QPushButton* runAll = nullptr;
        QPushButton* exportBtn = nullptr;

        explicit Impl(WorkbenchBridge& b) : bridge(b) {}
    };

    OpsPanel::OpsPanel(WorkbenchBridge& bridge, QWidget* parent) : QWidget(parent), impl_(std::make_unique<Impl>(bridge)) {
        setObjectName(QStringLiteral("Panel"));
        setMinimumWidth(theme::kOpsDockW);
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);

        // header
        auto* header = new QWidget(this);
        auto* hl = new QHBoxLayout(header);
        hl->setContentsMargins(14, 12, 14, 8);
        hl->addWidget(new CaptionLabel(QStringLiteral("Operations · any order"), header));
        hl->addStretch(1);
        impl_->count = widgets::label(QStringLiteral("0 steps"), 11, theme::kNeutral600, -1, header);
        hl->addWidget(impl_->count);
        root->addWidget(header);

        // scrolling rows + add + legend
        auto* scroll = new QScrollArea(this);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* content = new QWidget(scroll);
        auto* cl = new QVBoxLayout(content);
        cl->setContentsMargins(0, 0, 0, 0);
        cl->setSpacing(0);

        impl_->rowsHost = new QWidget(content);
        impl_->rowsLayout = new QVBoxLayout(impl_->rowsHost);
        impl_->rowsLayout->setContentsMargins(0, 0, 0, 0);
        impl_->rowsLayout->setSpacing(0);
        cl->addWidget(impl_->rowsHost);

        impl_->addRow = new ClickRow(content);
        {
            auto* grid = new QGridLayout(impl_->addRow);
            grid->setContentsMargins(10, 12, 14, 12);
            grid->setHorizontalSpacing(10);
            grid->setColumnMinimumWidth(0, 22);
            grid->setColumnStretch(1, 1);
            impl_->addGlyph = new GlyphButton(QStringLiteral("+"), 14, impl_->addRow);
            impl_->addGlyph->setDashed(true);
            impl_->addGlyph->setGlyphPx(11);
            impl_->addGlyph->setIdleColor(theme::kNeutral600);
            impl_->addGlyph->setAttribute(Qt::WA_TransparentForMouseEvents, true);
            grid->addWidget(impl_->addGlyph, 0, 0, 2, 1, Qt::AlignCenter);
            impl_->addTitle = widgets::heading(QStringLiteral("Add a processing step"), 13, impl_->addRow);
            QPalette tp = impl_->addTitle->palette();
            tp.setColor(QPalette::WindowText, theme::kNeutral600);
            impl_->addTitle->setPalette(tp);
            impl_->addHint = widgets::label(QString(), 11, theme::kNeutral600, -1, impl_->addRow);
            grid->addWidget(impl_->addTitle, 0, 1);
            grid->addWidget(impl_->addHint, 1, 1);
        }
        connect(impl_->addRow, &ClickRow::clicked, this, [this] {
            impl_->addMenu->setVisible(!impl_->addMenu->isVisible());
            impl_->addRow->setSelected(impl_->addMenu->isVisible());
        });
        cl->addWidget(impl_->addRow);

        impl_->addMenu = new AddMenu(bridge, content);
        impl_->addMenu->hide();
        auto* menuHost = new QWidget(content);
        auto* ml = new QHBoxLayout(menuHost);
        ml->setContentsMargins(10, 0, 14, 10);
        ml->addWidget(impl_->addMenu);
        cl->addWidget(menuHost);

        // legend
        cl->addWidget(new Rule(2, Qt::Horizontal, content));
        auto* legend = new QWidget(content);
        auto* lg = new QGridLayout(legend);
        lg->setContentsMargins(14, 12, 14, 12);
        lg->setHorizontalSpacing(8);
        lg->setVerticalSpacing(8);
        lg->setColumnMinimumWidth(0, 36);
        lg->setColumnStretch(1, 1);
        auto addLegend = [&](int row, QWidget* icon, const QString& text) {
            lg->addWidget(icon, row, 0, Qt::AlignCenter | Qt::AlignTop);
            auto* t = widgets::label(text, 12, theme::kNeutral700, -1, legend);
            t->setWordWrap(true);
            lg->addWidget(t, row, 1);
        };
        auto* onBox = new EnableBox(legend);
        onBox->setChecked(true);
        onBox->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        addLegend(0, onBox, QStringLiteral("Enabled — runs and passes its output on"));
        auto* offBox = new EnableBox(legend);
        offBox->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        addLegend(1, offBox, QStringLiteral("Skipped — data passes through unchanged"));
        auto* eye = new GlyphButton(QStringLiteral("◉"), 20, legend);
        eye->setActive(true);
        eye->setGlyphPx(10);
        eye->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        addLegend(2, eye, QStringLiteral("Shown in the viewer (click any step's ◉)"));
        auto* arrows = widgets::label(QStringLiteral("▲▼"), 10, theme::kNeutral500, -1, legend);
        addLegend(3, arrows, QStringLiteral("Reorder — steps run top to bottom"));
        auto* cacheIcons = new QWidget(legend);
        auto* cil = new QHBoxLayout(cacheIcons);
        cil->setContentsMargins(0, 0, 0, 0);
        cil->setSpacing(3);
        cil->addWidget(widgets::label(QStringLiteral("M"), 11, theme::kAccent, QFont::ExtraBold, cacheIcons));
        cil->addWidget(widgets::label(QStringLiteral("D"), 11, theme::kText, QFont::ExtraBold, cacheIcons));
        cil->addWidget(widgets::label(QStringLiteral("↻"), 11, theme::kNeutral500, QFont::ExtraBold, cacheIcons));
        addLegend(4, cacheIcons, QStringLiteral("Output cached in memory · on disk · recomputed"));
        cl->addWidget(legend);
        cl->addStretch(1);
        scroll->setWidget(content);
        root->addWidget(scroll, 1);

        // footer
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        auto* footer = new QWidget(this);
        auto* fl = new QVBoxLayout(footer);
        fl->setContentsMargins(14, 12, 14, 12);
        fl->setSpacing(8);
        impl_->runAll = new QPushButton(QStringLiteral("Run all enabled"), footer);
        widgets::setButtonClass(impl_->runAll, "primary");
        impl_->runAll->setToolTip(QStringLiteral("Run every enabled step top to bottom (⌘R)"));
        impl_->exportBtn = new QPushButton(QStringLiteral("Export result…"), footer);
        widgets::setButtonClass(impl_->exportBtn, "secondary");
        fl->addWidget(impl_->runAll);
        fl->addWidget(impl_->exportBtn);
        root->addWidget(footer);

        connect(impl_->runAll, &QPushButton::clicked, this, [this] { impl_->bridge.startRun(-1); });
        connect(impl_->exportBtn, &QPushButton::clicked, this, &OpsPanel::exportRequested);

        auto refreshAll = [this] { refresh(); };
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::datasetChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::selectionChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::viewedStepChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::outputsChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::stepChanged, this, [this](int index) {
            if (index >= 0 && index < impl_->rows.size()) impl_->rows[index]->refresh(index);
        });
        connect(&bridge, &WorkbenchBridge::runStarted, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::runFinished, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::runProgress, this, [this](double f, int, const QString&) {
            if (impl_->bridge.running())
                impl_->runAll->setText(QStringLiteral("Running · %1 %").arg(static_cast<int>(f * 100.0 + 0.5)));
        });
        refresh();
    }

    OpsPanel::~OpsPanel() = default;

    void OpsPanel::openAddMenu() {
        impl_->addMenu->show();
        impl_->addRow->setSelected(true);
    }

    void OpsPanel::refresh() {
        const Workbench& wb = impl_->bridge.wb();
        const Pipeline& p = wb.pipeline();
        const int n = p.size();
        while (impl_->rows.size() < n) {
            auto* row = new StepRow(impl_->bridge, impl_->rowsHost);
            impl_->rowsLayout->addWidget(row);
            impl_->rows.push_back(row);
        }
        while (impl_->rows.size() > n) {
            StepRow* row = impl_->rows.takeLast();
            impl_->rowsLayout->removeWidget(row);
            row->deleteLater();
        }
        for (int i = 0; i < n; ++i) impl_->rows[i]->refresh(i);
        impl_->count->setText(QStringLiteral("%1 %2").arg(n).arg(n == 1 ? QStringLiteral("step") : QStringLiteral("steps")));
        impl_->addHint->setText(n < 2 ? QStringLiteral("Reconstruct, reduce, adjust, combine, segment…")
                                      : QStringLiteral("Runs after step %1").arg(fromStd(Step::number(n - 1))));
        const bool running = impl_->bridge.running();
        impl_->runAll->setEnabled(!running && wb.hasDataset());
        if (!running) impl_->runAll->setText(QStringLiteral("Run all enabled"));
        impl_->exportBtn->setEnabled(wb.hasDataset() && !running);
    }

} // namespace sirius::app
