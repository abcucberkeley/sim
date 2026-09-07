#include "qt/viewer/viewer_widget.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

#include <QGridLayout>
#include <QHBoxLayout>
#include <QGuiApplication>
#include <QLabel>
#include <QPushButton>
#include <QPainter>
#include <QShortcut>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#include "core/array_source.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"
#include "qt/viewer/dims_strip.hpp"
#include "qt/viewer/display_model.hpp"
#include "qt/viewer/slice_pane.hpp"
#include "qt/viewer/viewer_widgets.hpp"
#include "qt/viewer/volume_view.hpp"

namespace sirius::app {

    namespace {
        constexpr double kMinZoom = 0.5, kMaxZoom = 16.0;
        constexpr int kPlayIntervalMs = 125;   // ~8 fps

        // The zoom readout of the tool strip: 10 px text running upwards.
        class VerticalLabel : public QWidget {
        public:
            explicit VerticalLabel(QWidget* parent) : QWidget(parent) { setFixedWidth(14); }
            void setText(const QString& t) {
                text_ = t;
                update();
            }
            QSize sizeHint() const override { return {14, 60}; }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                QFont f(theme::kFontFamily);
                f.setPixelSize(10);
                p.setFont(f);
                p.setPen(theme::kNeutral600);
                p.translate(width(), height());
                p.rotate(-90);
                p.drawText(QRect(0, 0, height(), width()), Qt::AlignVCenter | Qt::AlignLeft, text_);
            }

        private:
            QString text_;
        };

        QLabel* richLabel(QWidget* parent) {
            auto* l = new QLabel(parent);
            l->setTextFormat(Qt::RichText);
            l->setFont(theme::font(12));
            return l;
        }

        QString num2(int index) {
            return QString::fromStdString(Step::number(index));
        }

        Index clampIndex(double v, Index n) {
            return std::clamp<Index>(static_cast<Index>(std::floor(v)), 0, std::max<Index>(n - 1, 0));
        }
    } // namespace

    // -------------------------------------------------------------------------
    // Impl
    // -------------------------------------------------------------------------

    struct ViewerWidget::Impl {
        ViewerWidget* q;
        WorkbenchBridge& bridge;
        Workbench& wb;

        // chrome
        SegmentedControl* modeSeg = nullptr;
        QLabel* viewing = nullptr;
        QString viewingHtml;
        TokenCheck* labelsCheck = nullptr;
        TokenCheck* crossCheck = nullptr;
        TokenCheck* boxCheck = nullptr;
        QWidget* swatchHost = nullptr;
        QHBoxLayout* swatchLayout = nullptr;
        std::vector<ChannelSwatch*> swatches;
        QString swatchSignature;

        // tool strip
        std::array<GlyphButton*, 5> tools{};
        VerticalLabel* zoomReadout = nullptr;

        // views
        QStackedWidget* stack = nullptr;
        QWidget* orthoPage = nullptr;
        SlicePane* xy = nullptr;
        SlicePane* yz = nullptr;
        SlicePane* xz = nullptr;
        SlicePane* mip = nullptr;
        VolumeView* volume = nullptr;      // null when the platform cannot host a QOpenGLWidget
        QWidget* volumePage = nullptr;     // the VolumeView, or a notice
        QWidget* comparePage = nullptr;
        SlicePane* cmpLeft = nullptr;
        SlicePane* cmpRight = nullptr;
        DimsStrip* dims = nullptr;

        // rendering state
        DisplayModel model, rawModel;
        int displayIndex = -1;
        QImage xyImg, xzImg, yzImg, mipImg, cmpLeftImg, cmpRightImg;
        int xyFactor = 1, mipFactor = 1, cmpFactor = 1;
        struct Dirty {
            bool xy = true, xz = true, yz = true, mip = true, cmp = true, vol = true;
        } dirty;
        bool updateQueued = false;
        ViewState prev;
        bool havePrev = false;
        bool rebuildQueued = false;

        // interaction
        QVector<QPointF> measure;
        QRectF roi;
        QPointF roiStart;
        std::array<Index, 3> splitA{};
        bool splitPending = false;
        std::uint32_t mergeFirst = 0;
        QPointF lastPaint;
        bool painting = false;
        QTimer playTimer;
        QString cursorText = QStringLiteral("cursor —");
        QString zoomText = QStringLiteral("100 %");

        Impl(ViewerWidget* q, WorkbenchBridge& b) : q(q), bridge(b), wb(b.wb()) {}

        // --- helpers ---------------------------------------------------------
        const ViewState& vs() const { return wb.viewState(); }
        Index nx() const { return model.dims().x; }
        Index ny() const { return model.dims().y; }
        Index nz() const { return model.dims().z; }
        Index nt() const { return model.dims().t; }
        double zAspect() const {
            const auto& v = model.meta().voxelUm;
            return v[0] > 0.0 && v[2] > 0.0 ? v[2] / v[0] : 1.0;
        }
        Index curZ() const { return std::clamp<Index>(vs().z, 0, std::max<Index>(nz() - 1, 0)); }
        Index curT() const { return std::clamp<Index>(vs().t, 0, std::max<Index>(nt() - 1, 0)); }
        Index curX() const { return std::clamp<Index>(vs().cx, 0, std::max<Index>(nx() - 1, 0)); }
        Index curY() const { return std::clamp<Index>(vs().cy, 0, std::max<Index>(ny() - 1, 0)); }
        bool probe() const { return vs().tool == ViewerTool::Probe; }

        bool paintAvailable() const {
            if (model.hasLabels()) return true;
            for (const Step& s : wb.pipeline().steps()) {
                if (!s.enabled) continue;
                if (const Operation* op = findOperation(s.kind))
                    if (op->info().producesLabels || op->info().needsLabels) return true;
            }
            return false;
        }

        void build();
        void connectSignals();
        void rebuildOutput();          // display output changed
        void refreshChrome();
        void refreshSwatches();
        void refreshDims();
        void refreshHints();
        void applyViewStateDiff(const ViewState& s);
        void scheduleUpdate();
        void applyDirty();
        void layoutPanes();
        void renderVolume();
        void setZoomPan(double zoom, double panX, double panY);
        void zoomAround(double factor, const QPointF& xyScreen);
        void fit();
        void setCursorFor(ViewerTool t);

        // tools
        void onXYPressed(const QPointF& v, Qt::MouseButton b, Qt::KeyboardModifiers m);
        void onXYDragged(const QPointF& v, const QPointF& delta, Qt::MouseButton b, Qt::KeyboardModifiers m);
        void onXYReleased(const QPointF& v, Qt::MouseButton b, Qt::KeyboardModifiers m, bool moved);
        void paintAt(const QPointF& v, bool erase);
        std::uint32_t labelAt(Index z, Index y, Index x) const;
        void updateMeasureText();
        void hover(SlicePane::Kind kind, const QPointF& v);
    };

    // --- building ----------------------------------------------------------------

    void ViewerWidget::Impl::build() {
        auto* root = new QVBoxLayout(q);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);

        // toolbar
        auto* bar = new QWidget(q);
        bar->setFixedHeight(theme::kViewerToolbarH);
        bar->setObjectName(QStringLiteral("viewerToolbar"));
        auto* bl = new QHBoxLayout(bar);
        bl->setContentsMargins(14, 0, 14, 0);
        bl->setSpacing(14);
        modeSeg = new SegmentedControl({QStringLiteral("Ortho"), QStringLiteral("3D"), QStringLiteral("Compare")}, bar);
        bl->addWidget(modeSeg);
        viewing = richLabel(bar);
        bl->addWidget(viewing);
        bl->addStretch(1);
        labelsCheck = new TokenCheck(QStringLiteral("Labels"), bar);
        crossCheck = new TokenCheck(QStringLiteral("Crosshair"), bar);
        boxCheck = new TokenCheck(QStringLiteral("Bounding box"), bar);
        bl->addWidget(labelsCheck);
        bl->addWidget(crossCheck);
        bl->addWidget(boxCheck);
        swatchHost = new QWidget(bar);
        swatchLayout = new QHBoxLayout(swatchHost);
        swatchLayout->setContentsMargins(0, 0, 0, 0);
        swatchLayout->setSpacing(4);
        bl->addWidget(swatchHost);
        // display contrast: the auto percentile window or the full data range
        auto* autoBtn = new QPushButton(QStringLiteral("Auto"), bar);
        auto* resetBtn = new QPushButton(QStringLiteral("Reset"), bar);
        for (QPushButton* b : {autoBtn, resetBtn}) {
            widgets::setButtonClass(b, "ghost small");
            b->setCursor(Qt::PointingHandCursor);
            b->setFocusPolicy(Qt::NoFocus);
        }
        autoBtn->setToolTip(QStringLiteral("Auto contrast (display): window on the 0.1–99.9 percentiles (⇧A)"));
        resetBtn->setToolTip(QStringLiteral("Reset the display window to the full data range (⇧R)"));
        QObject::connect(autoBtn, &QPushButton::clicked, q, [this] { q->autoContrast(); });
        QObject::connect(resetBtn, &QPushButton::clicked, q, [this] { q->resetContrast(); });
        bl->addWidget(autoBtn);
        bl->addWidget(resetBtn);
        root->addWidget(bar);
        auto* rule1 = new QFrame(q);
        rule1->setFixedHeight(theme::kRule);
        rule1->setStyleSheet(QStringLiteral("background:%1;").arg(theme::kDivider.name()));
        root->addWidget(rule1);

        // canvas area: tool strip + views on the neutral-900 ground
        auto* canvas = new QWidget(q);
        canvas->setAutoFillBackground(true);
        {
            QPalette pal = canvas->palette();
            pal.setColor(QPalette::Window, theme::kNeutral900);
            canvas->setPalette(pal);
        }
        auto* cl = new QHBoxLayout(canvas);
        cl->setContentsMargins(2, 2, 2, 2);
        cl->setSpacing(2);

        auto* strip = new QWidget(canvas);
        strip->setFixedWidth(theme::kToolStripW);
        strip->setAutoFillBackground(true);
        {
            QPalette pal = strip->palette();
            pal.setColor(QPalette::Window, theme::kBg);
            strip->setPalette(pal);
        }
        auto* sl = new QVBoxLayout(strip);
        sl->setContentsMargins(0, 4, 0, 4);
        sl->setSpacing(2);
        sl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
        static const struct { const char* glyph; const char* tip; } toolDefs[] = {
            {"✥", "Navigate — drag to pan, wheel to zoom, double-click to fit (V)"},
            {"+", "Probe — click to place the crosshair and read values (P)"},
            {"↔", "Measure distance / angle (M)"},
            {"▢", "ROI — drag a box (R)"},
            {"●", "Paint labels — needs a segmentation step (B)"}};
        for (std::size_t i = 0; i < tools.size(); ++i) {
            auto* b = new GlyphButton(QString::fromUtf8(toolDefs[i].glyph), strip);
            b->setToolTip(QString::fromUtf8(toolDefs[i].tip));
            b->setCheckable(true);
            tools[i] = b;
            sl->addWidget(b, 0, Qt::AlignHCenter);
        }
        auto* stripRule = new QFrame(strip);
        stripRule->setFixedSize(20, theme::kRule);
        stripRule->setStyleSheet(QStringLiteral("background:%1;").arg(theme::kDivider.name()));
        sl->addSpacing(6);
        sl->addWidget(stripRule, 0, Qt::AlignHCenter);
        sl->addSpacing(6);
        auto* zin = new GlyphButton(QStringLiteral("+"), strip);
        zin->setGlyphPx(14);
        zin->setToolTip(QStringLiteral("Zoom in (+)"));
        auto* zout = new GlyphButton(QStringLiteral("−"), strip);
        zout->setGlyphPx(14);
        zout->setToolTip(QStringLiteral("Zoom out (−)"));
        auto* zfit = new GlyphButton(QStringLiteral("⤢"), strip);
        zfit->setGlyphPx(11);
        zfit->setToolTip(QStringLiteral("Fit to view (0)"));
        for (GlyphButton* b : {zin, zout, zfit}) sl->addWidget(b, 0, Qt::AlignHCenter);
        sl->addStretch(1);
        zoomReadout = new VerticalLabel(strip);
        sl->addWidget(zoomReadout, 0, Qt::AlignHCenter);
        cl->addWidget(strip);

        stack = new QStackedWidget(canvas);
        // ortho grid: 1fr 220 / 1fr 170, 2 px gaps on neutral-900
        orthoPage = new QWidget(stack);
        auto* grid = new QGridLayout(orthoPage);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setSpacing(2);
        xy = new SlicePane(SlicePane::Kind::XY, orthoPage);
        yz = new SlicePane(SlicePane::Kind::YZ, orthoPage);
        xz = new SlicePane(SlicePane::Kind::XZ, orthoPage);
        mip = new SlicePane(SlicePane::Kind::MIP, orthoPage);
        xy->setObjectName(QStringLiteral("xyPane"));
        yz->setObjectName(QStringLiteral("yzPane"));
        xz->setObjectName(QStringLiteral("xzPane"));
        mip->setObjectName(QStringLiteral("mipPane"));
        yz->setFixedWidth(220);
        mip->setFixedSize(220, 170);
        xz->setFixedHeight(170);
        grid->addWidget(xy, 0, 0);
        grid->addWidget(yz, 0, 1);
        grid->addWidget(xz, 1, 0);
        grid->addWidget(mip, 1, 1);
        grid->setColumnStretch(0, 1);
        grid->setRowStretch(0, 1);
        stack->addWidget(orthoPage);

        // QOpenGLWidget is unsupported (and crashes at flush) on the
        // offscreen / minimal platforms used by headless runs and tests.
        const QString platform = QGuiApplication::platformName();
        if (platform != QLatin1String("offscreen") && platform != QLatin1String("minimal") &&
            platform != QLatin1String("vnc")) {
            volume = new VolumeView(stack);
            volumePage = volume;
        } else {
            auto* notice = new QLabel(QStringLiteral("3D rendering needs OpenGL, which the %1 platform does not provide").arg(platform), stack);
            notice->setAlignment(Qt::AlignCenter);
            notice->setStyleSheet(QStringLiteral("background:%1;color:%2;").arg(theme::hex(theme::kViewerGround), theme::hex(theme::kViewerText)));
            volumePage = notice;
        }
        stack->addWidget(volumePage);

        comparePage = new QWidget(stack);
        auto* cgrid = new QHBoxLayout(comparePage);
        cgrid->setContentsMargins(0, 0, 0, 0);
        cgrid->setSpacing(2);
        cmpLeft = new SlicePane(SlicePane::Kind::Compare, comparePage);
        cmpRight = new SlicePane(SlicePane::Kind::Compare, comparePage);
        cmpLeft->setObjectName(QStringLiteral("compareRawPane"));
        cmpRight->setObjectName(QStringLiteral("compareStepPane"));
        cgrid->addWidget(cmpLeft, 1);
        cgrid->addWidget(cmpRight, 1);
        stack->addWidget(comparePage);
        cl->addWidget(stack, 1);
        root->addWidget(canvas, 1);

        auto* rule2 = new QFrame(q);
        rule2->setFixedHeight(theme::kRule);
        rule2->setStyleSheet(QStringLiteral("background:%1;").arg(theme::kDivider.name()));
        root->addWidget(rule2);
        dims = new DimsStrip(q);
        root->addWidget(dims);

        // chrome wiring
        QObject::connect(modeSeg, &SegmentedControl::changed, q, [this](int i) {
            wb.setViewMode(i == 0 ? ViewMode::Ortho : i == 1 ? ViewMode::Volume : ViewMode::Compare);
        });
        QObject::connect(labelsCheck, &QAbstractButton::clicked, q, [this] { wb.toggleLabels(); });
        QObject::connect(crossCheck, &QAbstractButton::clicked, q, [this] { wb.toggleCrosshair(); });
        QObject::connect(boxCheck, &QAbstractButton::clicked, q, [this] {
            ViewState s = vs();
            s.boundingBox = !s.boundingBox;
            wb.setViewState(s);
        });
        static const ViewerTool toolIds[] = {ViewerTool::Navigate, ViewerTool::Probe, ViewerTool::Measure,
                                             ViewerTool::Roi, ViewerTool::Paint};
        for (std::size_t i = 0; i < tools.size(); ++i) {
            const ViewerTool t = toolIds[i];
            QObject::connect(tools[i], &QAbstractButton::clicked, q, [this, t] {
                if (t == ViewerTool::Paint && !paintAvailable()) {
                    refreshChrome();
                    return;
                }
                wb.setTool(t);
            });
        }
        QObject::connect(zin, &QAbstractButton::clicked, q, [this] { q->zoomIn(); });
        QObject::connect(zout, &QAbstractButton::clicked, q, [this] { q->zoomOut(); });
        QObject::connect(zfit, &QAbstractButton::clicked, q, [this] { q->fitToWindow(); });

        // keyboard: tool letters and brush size (the menus own the rest)
        auto key = [this](const QKeySequence& ks, std::function<void()> fn) {
            auto* sc = new QShortcut(ks, q);
            sc->setContext(Qt::WidgetWithChildrenShortcut);
            QObject::connect(sc, &QShortcut::activated, q, std::move(fn));
        };
        key(QKeySequence(Qt::Key_V), [this] { wb.setTool(ViewerTool::Navigate); });
        key(QKeySequence(Qt::Key_P), [this] { wb.setTool(ViewerTool::Probe); });
        key(QKeySequence(Qt::Key_M), [this] { wb.setTool(ViewerTool::Measure); });
        key(QKeySequence(Qt::Key_R), [this] { wb.setTool(ViewerTool::Roi); });
        key(QKeySequence(Qt::Key_BracketLeft), [this] {
            ViewState s = vs();
            s.brushPx = std::max(2, s.brushPx - 2);
            wb.setViewState(s);
        });
        key(QKeySequence(Qt::Key_BracketRight), [this] {
            ViewState s = vs();
            s.brushPx = std::min(60, s.brushPx + 2);
            wb.setViewState(s);
        });

        playTimer.setInterval(kPlayIntervalMs);
        QObject::connect(&playTimer, &QTimer::timeout, q, [this] {
            if (nt() <= 1) {
                q->setPlaying(false);
                return;
            }
            wb.setT((curT() + 1) % nt());
        });
        QObject::connect(dims, &DimsStrip::zRequested, q, [this](Index z) { wb.setZ(z); });
        QObject::connect(dims, &DimsStrip::tRequested, q, [this](Index t) { wb.setT(t); });
        QObject::connect(dims, &DimsStrip::playToggled, q, [this](bool on) { q->setPlaying(on); });
    }

    void ViewerWidget::Impl::connectSignals() {
        // pane events
        QObject::connect(xy, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers m) { onXYPressed(v, b, m); });
        QObject::connect(xy, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers m) { onXYDragged(v, d, b, m); });
        QObject::connect(xy, &SlicePane::released, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers m, bool moved) { onXYReleased(v, b, m, moved); });
        QObject::connect(xy, &SlicePane::doubleClicked, q, [this](QPointF, Qt::KeyboardModifiers) {
            if (vs().tool == ViewerTool::Navigate || vs().tool == ViewerTool::Probe) fit();
        });
        QObject::connect(xy, &SlicePane::wheeled, q, [this](QPointF s, double steps) { zoomAround(std::pow(1.15, steps), s); });
        QObject::connect(xy, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::XY, v); });
        QObject::connect(xy, &SlicePane::resized, q, [this] { layoutPanes(); dirty.xy = true; scheduleUpdate(); });

        // YZ / XZ: probe moves the crosshair (and z); navigate pans along the shared axis
        QObject::connect(yz, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(curX(), clampIndex(v.y(), ny()), clampIndex(v.x(), nz()));
        });
        QObject::connect(yz, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::MiddleButton || (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate))
                setZoomPan(vs().zoom, vs().panX, vs().panY + d.y());
            else if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(curX(), clampIndex(v.y(), ny()), clampIndex(v.x(), nz()));
        });
        QObject::connect(yz, &SlicePane::wheeled, q, [this](QPointF s, double steps) {
            zoomAround(std::pow(1.15, steps), QPointF(xy->width() / 2.0, s.y()));
        });
        QObject::connect(yz, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::YZ, v); });
        QObject::connect(yz, &SlicePane::resized, q, [this] { layoutPanes(); });

        QObject::connect(xz, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(clampIndex(v.x(), nx()), curY(), clampIndex(v.y(), nz()));
        });
        QObject::connect(xz, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::MiddleButton || (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate))
                setZoomPan(vs().zoom, vs().panX + d.x(), vs().panY);
            else if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(clampIndex(v.x(), nx()), curY(), clampIndex(v.y(), nz()));
        });
        QObject::connect(xz, &SlicePane::wheeled, q, [this](QPointF s, double steps) {
            zoomAround(std::pow(1.15, steps), QPointF(s.x(), xy->height() / 2.0));
        });
        QObject::connect(xz, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::XZ, v); });
        QObject::connect(xz, &SlicePane::resized, q, [this] { layoutPanes(); });
        QObject::connect(mip, &SlicePane::resized, q, [this] { layoutPanes(); dirty.mip = true; scheduleUpdate(); });
        QObject::connect(mip, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
        });
        for (SlicePane* p : {xy, yz, xz, mip, cmpLeft, cmpRight})
            QObject::connect(p, &SlicePane::exited, q, [this] {
                cursorText = QStringLiteral("cursor —");
                emit q->cursorChanged(cursorText);
            });

        // compare panes share the XY transform
        for (SlicePane* p : {cmpLeft, cmpRight}) {
            QObject::connect(p, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers) {
                if (b == Qt::MiddleButton || (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate))
                    setZoomPan(vs().zoom, vs().panX + d.x(), vs().panY + d.y());
                else if (b == Qt::LeftButton && probe() && model.valid())
                    wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
            });
            QObject::connect(p, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
                if (b == Qt::LeftButton && probe() && model.valid())
                    wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
            });
            QObject::connect(p, &SlicePane::wheeled, q, [this](QPointF s, double steps) { zoomAround(std::pow(1.15, steps), s); });
            QObject::connect(p, &SlicePane::doubleClicked, q, [this](QPointF, Qt::KeyboardModifiers) { fit(); });
            QObject::connect(p, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::Compare, v); });
            QObject::connect(p, &SlicePane::resized, q, [this] { layoutPanes(); dirty.cmp = true; scheduleUpdate(); });
        }

        // volume view
        if (volume) {
            QObject::connect(volume, &VolumeView::orientationChanged, q, [this](double yaw, double pitch) {
                ViewState s = vs();
                s.yaw = yaw;
                s.pitch = pitch;
                wb.setViewState(s);
            });
            QObject::connect(volume, &VolumeView::clipChanged, q, [this](double lo, double hi) {
                ViewState s = vs();
                s.clipZ = {lo, hi};
                wb.setViewState(s);
            });
        }

        // workbench
        QObject::connect(&bridge, &WorkbenchBridge::datasetChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::viewedStepChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::outputsChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::pipelineChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::runFinished, q, [this](bool, const QString&) { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::stepChanged, q, [this](int index) {
            if (index == wb.viewedIndex()) {
                refreshChrome();
                dirty.vol = true;
                scheduleUpdate();
            }
        });
        QObject::connect(&bridge, &WorkbenchBridge::labelsChanged, q, [this](quint64) {
            dirty.xy = dirty.xz = dirty.yz = dirty.cmp = true;
            scheduleUpdate();
        });
        QObject::connect(&bridge, &WorkbenchBridge::viewStateChanged, q, [this] { applyViewStateDiff(vs()); });
    }

    // --- output --------------------------------------------------------------------

    void ViewerWidget::Impl::rebuildOutput() {
        // coalesce bursts (a run finishing emits several notifications)
        if (rebuildQueued) return;
        rebuildQueued = true;
        QTimer::singleShot(0, q, [this] {
            rebuildQueued = false;
            int actual = -1;
            std::shared_ptr<const StepOutput> out = wb.hasDataset() ? wb.displayOutput(&actual) : nullptr;
            const bool changed = out != model.output();
            model.setOutput(out);
            displayIndex = actual;
            // the raw side of Compare: the Load step's output, or the bare source
            std::shared_ptr<const StepOutput> raw = wb.hasDataset() ? wb.output(0) : nullptr;
            if (!raw && wb.hasDataset() && wb.source()) {
                auto so = std::make_shared<StepOutput>();
                so->meta = wb.dataset();
                so->source = wb.source();
                raw = so;
            }
            if (raw != rawModel.output()) {
                rawModel.setOutput(raw);
                dirty.cmp = true;
            }
            if (changed) {
                dirty = Dirty{};
                measure.clear();
                roi = QRectF();
                splitPending = false;
                mergeFirst = 0;
            }
            refreshSwatches();
            refreshChrome();
            refreshDims();
            layoutPanes();
            scheduleUpdate();
        });
    }

    void ViewerWidget::Impl::refreshSwatches() {
        const DatasetMeta& m = model.meta();
        QString sig;
        std::vector<std::pair<QString, QColor>> items;
        if (model.valid()) {
            if (m.rgb) {
                items = {{QStringLiteral("R"), QColor(255, 80, 80)}, {QStringLiteral("G"), QColor(80, 255, 80)},
                         {QStringLiteral("B"), QColor(110, 110, 255)}};
            } else {
                for (Index c = 0; c < m.dims.c; ++c) {
                    if (static_cast<std::size_t>(c) < m.channels.size()) {
                        const ChannelInfo& ch = m.channels[static_cast<std::size_t>(c)];
                        items.emplace_back(QString::fromStdString(ch.shortName()), QColor(QString::fromStdString(ch.hexColor())));
                    } else {
                        items.emplace_back(QString::number(c), QColor(255, 255, 255));
                    }
                }
            }
        }
        for (const auto& it : items) sig += it.first + it.second.name() + QLatin1Char(';');
        if (sig != swatchSignature) {
            swatchSignature = sig;
            for (ChannelSwatch* s : swatches) delete s;
            swatches.clear();
            for (std::size_t c = 0; c < items.size(); ++c) {
                auto* s = new ChannelSwatch(items[c].first, items[c].second, swatchHost);
                const Index ci = static_cast<Index>(c);
                QObject::connect(s, &QAbstractButton::clicked, q, [this, ci] { wb.setChannelVisible(ci, !vs().channelOn(ci)); });
                swatchLayout->addWidget(s);
                swatches.push_back(s);
            }
        }
        for (std::size_t c = 0; c < swatches.size(); ++c) swatches[c]->setChecked(vs().channelOn(static_cast<Index>(c)));
    }

    void ViewerWidget::Impl::refreshChrome() {
        const ViewState& s = vs();
        modeSeg->setCurrent(s.mode == ViewMode::Ortho ? 0 : s.mode == ViewMode::Volume ? 1 : 2);
        stack->setCurrentWidget(s.mode == ViewMode::Ortho ? orthoPage : s.mode == ViewMode::Volume ? volumePage : comparePage);

        // "Viewing 05 Contrast rgb z48 y4096 x4096"
        const int viewed = wb.viewedIndex();
        QString html = QStringLiteral("<span style='color:%1'>Viewing</span>").arg(theme::kNeutral600.name());
        if (viewed >= 0 && viewed < wb.pipeline().size()) {
            const Step& st = wb.pipeline().at(viewed);
            html += QStringLiteral(" <span style='color:%1;font-weight:800'>%2 %3</span>")
                        .arg(theme::kAccent.name(), num2(viewed), QString::fromStdString(st.name).toHtmlEscaped());
            QString shape = QString::fromStdString(wb.outputMetaOf(viewed).shapeString());
            html += QStringLiteral(" <span style='color:%1'>%2</span>").arg(theme::kNeutral500.name(), shape);
            if (displayIndex >= 0 && displayIndex != viewed)
                html += QStringLiteral(" <span style='color:%1'>· not run yet, showing %2</span>")
                            .arg(theme::kNeutral500.name(), num2(displayIndex));
        } else if (!wb.hasDataset()) {
            html += QStringLiteral(" <span style='color:%1'>no dataset</span>").arg(theme::kNeutral500.name());
        }
        if (html != viewingHtml) {   // a pan drag must not relayout the toolbar
            viewingHtml = html;
            viewing->setText(html);
        }

        labelsCheck->setChecked(s.labels);
        labelsCheck->setEnabled(model.hasLabels());
        crossCheck->setChecked(s.crosshair);
        crossCheck->setCaption(s.tool == ViewerTool::Probe ? QString() : QStringLiteral("locked"));
        crossCheck->setVisible(s.mode != ViewMode::Volume);
        boxCheck->setChecked(s.boundingBox);
        boxCheck->setVisible(s.mode == ViewMode::Volume);

        static const ViewerTool toolIds[] = {ViewerTool::Navigate, ViewerTool::Probe, ViewerTool::Measure,
                                             ViewerTool::Roi, ViewerTool::Paint};
        const bool paintOk = paintAvailable();
        for (std::size_t i = 0; i < tools.size(); ++i) {
            tools[i]->setChecked(s.tool == toolIds[i]);
            if (toolIds[i] == ViewerTool::Paint) tools[i]->setEnabled(paintOk);
        }
        zoomText = QStringLiteral("%1 %").arg(std::lround(s.zoom * 100.0));
        zoomReadout->setText(zoomText);
        emit q->zoomChanged(zoomText);
        refreshHints();
        setCursorFor(s.tool);
        if (!volume) return;
        volume->setBoundingBox(s.boundingBox);
        volume->setOrientation(s.yaw, s.pitch);
        volume->setClip(s.clipZ[0], s.clipZ[1]);
        // transfer function from a volume reconstruction step's parameters
        if (viewed >= 0 && viewed < wb.pipeline().size() && wb.pipeline().at(viewed).kind == "volrec") {
            const ParamSet& p = wb.pipeline().at(viewed).params;
            const std::string method = p.getString("method", "Ray casting");
            const bool isMip = method.find("ax") != std::string::npos || method.find("MIP") != std::string::npos;
            volume->setTransfer(static_cast<float>(p.getDouble("opacity_lo", 0.05)),
                                static_cast<float>(p.getDouble("opacity_hi", 0.6)),
                                static_cast<float>(p.getDouble("opacity", 0.9)),
                                static_cast<float>(p.getDouble("step", 0.5)), isMip);
            volume->setMethodText(QString::fromStdString(method));
        } else {
            volume->setTransfer(0.05f, 0.6f, 0.9f, 0.5f, false);
            volume->setMethodText(QStringLiteral("Ray casting"));
        }
    }

    void ViewerWidget::Impl::refreshHints() {
        const ViewState& s = vs();
        QString hint;
        switch (s.tool) {
            case ViewerTool::Navigate: hint = QStringLiteral("drag · pan   wheel · zoom   double-click · fit"); break;
            case ViewerTool::Probe: hint = QStringLiteral("click · move crosshair   drag · follow   wheel · zoom"); break;
            case ViewerTool::Measure: hint = QStringLiteral("click twice · distance   third click · angle   fourth · restart"); break;
            case ViewerTool::Roi: hint = QStringLiteral("drag · box   wheel · zoom"); break;
            case ViewerTool::Paint: {
                QString what;
                switch (s.paintTool) {
                    case PaintTool::Brush: what = QStringLiteral("brush %1 px · alt · erase · [ ] · size").arg(s.brushPx); break;
                    case PaintTool::Erase: what = QStringLiteral("erase %1 px · [ ] · size").arg(s.brushPx); break;
                    case PaintTool::Fill: what = QStringLiteral("click · fill region with the selected label"); break;
                    case PaintTool::Pick: what = QStringLiteral("click · pick label"); break;
                    case PaintTool::Merge: what = mergeFirst ? QStringLiteral("click the label to merge into %1").arg(mergeFirst) : QStringLiteral("click first label · then the second"); break;
                    case PaintTool::Split: what = splitPending ? QStringLiteral("click the second seed") : QStringLiteral("click two seeds inside one label"); break;
                    case PaintTool::Delete: what = QStringLiteral("click · delete label"); break;
                    case PaintTool::Lasso: what = QStringLiteral("lasso not available · painting with the brush"); break;
                }
                hint = what + QStringLiteral(" · crosshair locked");
                break;
            }
        }
        if (model.volumeTooLarge()) {
            yz->setMessage(QStringLiteral("volume too large for re-slicing"));
            xz->setMessage(QStringLiteral("volume too large for re-slicing"));
            mip->setMessage(QStringLiteral("volume too large"));
        } else {
            yz->setMessage({});
            xz->setMessage({});
            mip->setMessage({});
        }
        xy->setHint(hint);
        cmpRight->setHint(hint);
        const bool brush = s.tool == ViewerTool::Paint && (s.paintTool == PaintTool::Brush || s.paintTool == PaintTool::Erase || s.paintTool == PaintTool::Lasso);
        xy->setBrushCursor(brush, s.brushPx / 2.0);
        cmpRight->setBrushCursor(brush, s.brushPx / 2.0);
    }

    void ViewerWidget::Impl::setCursorFor(ViewerTool t) {
        Qt::CursorShape shape = Qt::CrossCursor;
        if (t == ViewerTool::Navigate) shape = Qt::OpenHandCursor;
        else if (t == ViewerTool::Paint) shape = Qt::BlankCursor;
        xy->setCursor(shape);
        cmpLeft->setCursor(t == ViewerTool::Navigate ? Qt::OpenHandCursor : Qt::CrossCursor);
        cmpRight->setCursor(shape);
        for (SlicePane* p : {yz, xz, mip}) p->setCursor(t == ViewerTool::Navigate ? Qt::OpenHandCursor : Qt::CrossCursor);
    }

    void ViewerWidget::Impl::refreshDims() {
        const DatasetMeta& m = model.meta();
        dims->setExtents(model.valid() ? nz() : 1, model.valid() ? nt() : 1, m.dz(), m.frameIntervalS);
        dims->setPosition(vs().z, vs().t);
        dims->setPlaying(playTimer.isActive());
    }

    // --- view state ------------------------------------------------------------------

    void ViewerWidget::Impl::applyViewStateDiff(const ViewState& s) {
        if (!havePrev) {
            prev = s;
            havePrev = true;
            dirty = Dirty{};
        } else {
            if (s.t != prev.t) dirty = Dirty{};
            if (s.z != prev.z) dirty.xy = dirty.cmp = true;
            if (s.cx != prev.cx || s.cy != prev.cy) dirty.xz = dirty.yz = true;
            if (s.channelVisible != prev.channelVisible) dirty = Dirty{};
            if (s.labels != prev.labels || s.labelOpacity != prev.labelOpacity || s.selectedLabel != prev.selectedLabel)
                dirty.xy = dirty.xz = dirty.yz = dirty.cmp = true;
            if (s.mode != prev.mode) {
                if (s.mode == ViewMode::Volume) dirty.vol = true;
                if (s.mode == ViewMode::Compare) dirty.cmp = true;
            }
            if (s.yaw != prev.yaw || s.pitch != prev.pitch || s.clipZ != prev.clipZ || s.boundingBox != prev.boundingBox)
                dirty.vol = dirty.vol || false;   // the volume view keeps its own orientation state
            prev = s;
        }
        refreshChrome();
        refreshSwatches();
        dims->setPosition(s.z, s.t);
        layoutPanes();
        scheduleUpdate();
    }

    void ViewerWidget::Impl::scheduleUpdate() {
        if (updateQueued) return;
        updateQueued = true;
        QTimer::singleShot(0, q, [this] {
            updateQueued = false;
            applyDirty();
        });
    }

    void ViewerWidget::Impl::layoutPanes() {
        if (!model.valid()) return;
        const ViewState& s = vs();
        const double za = zAspect();
        // XY: fit x state.zoom, offset by the pan
        xy->setContent(xyImg, xyFactor, nx(), ny());   // keeps cols/rows current before fitView
        const SlicePane::View fitV = xy->fitView(1.0, 1.0);
        SlicePane::View v;
        v.zx = v.zy = fitV.zx * s.zoom;
        v.ox = (xy->width() - nx() * v.zx) / 2.0 + s.panX;
        v.oy = (xy->height() - ny() * v.zy) / 2.0 + s.panY;
        xy->setView(v);
        const Index cz = curZ();
        // YZ: rows y follow XY, cols z at the physical aspect, centred on z when wider than the pane
        {
            SlicePane::View w;
            w.zy = v.zy;
            w.oy = v.oy;
            w.zx = v.zx * za;
            const double ez = nz() * w.zx;
            w.ox = ez <= yz->width() ? (yz->width() - ez) / 2.0 : yz->width() / 2.0 - (cz + 0.5) * w.zx;
            yz->setView(w);
        }
        {
            SlicePane::View w;
            w.zx = v.zx;
            w.ox = v.ox;
            w.zy = v.zy * za;
            const double ez = nz() * w.zy;
            w.oy = ez <= xz->height() ? (xz->height() - ez) / 2.0 : xz->height() / 2.0 - (cz + 0.5) * w.zy;
            xz->setView(w);
        }
        mip->setView(mip->fitView(1.0, 1.0));
        {
            // Both compare panes show the same physical field: the raw pane
            // scales its (coarser or finer) voxels by the voxel-size ratio so
            // a 64-pixel raw frame overlays a 128-pixel reconstruction.
            const SlicePane::View f = cmpRight->fitView(1.0, 1.0);
            SlicePane::View w;
            w.zx = w.zy = f.zx * s.zoom;
            w.ox = (cmpRight->width() - nx() * w.zx) / 2.0 + s.panX;
            w.oy = (cmpRight->height() - ny() * w.zy) / 2.0 + s.panY;
            cmpRight->setView(w);
            SlicePane::View l = w;
            if (rawModel.valid() && rawModel.meta().dx() > 0.0 && rawModel.meta().dy() > 0.0) {
                // screen pixels per raw voxel: a raw voxel twice as large as
                // the reconstruction's covers twice the screen
                l.zx = w.zx * rawModel.meta().dx() / model.meta().dx();
                l.zy = w.zy * rawModel.meta().dy() / model.meta().dy();
            }
            cmpLeft->setView(l);
        }
        // a coarser render is enough when the image is smaller than the pane
        const int wantFactor = std::max(1, static_cast<int>(std::floor(1.0 / std::max(v.zx, 1e-6))));
        if (wantFactor != xyFactor) {
            xyFactor = wantFactor;
            dirty.xy = true;
        }
        const int cmpWant = std::max(1, static_cast<int>(std::floor(1.0 / std::max(cmpRight->view().zx, 1e-6))));
        if (cmpWant != cmpFactor) {
            cmpFactor = cmpWant;
            dirty.cmp = true;
        }
        const int mipWant = std::max(1, static_cast<int>(std::floor(1.0 / std::max(mip->view().zx, 1e-6))));
        if (mipWant != mipFactor) {
            mipFactor = mipWant;
            dirty.mip = true;
        }
        xy->setSmooth(v.zx * xyFactor < 1.0);
        // overlays that depend on the view
        const ViewState& st = vs();
        const bool locked = st.tool != ViewerTool::Probe;
        const QPointF cross(static_cast<double>(curX()), static_cast<double>(curY()));
        xy->setCrosshair(cross, st.crosshair, locked);
        yz->setCrosshair(QPointF(static_cast<double>(cz), static_cast<double>(curY())), st.crosshair, locked);
        xz->setCrosshair(QPointF(static_cast<double>(curX()), static_cast<double>(cz)), st.crosshair, locked);
        mip->setCrosshair(cross, st.crosshair, locked);
        QPointF rawCross = cross;
        double rawDx = model.meta().dx();
        if (rawModel.valid() && rawModel.meta().dx() > 0.0 && rawModel.meta().dy() > 0.0) {
            rawCross = QPointF(cross.x() * model.meta().dx() / rawModel.meta().dx(),
                               cross.y() * model.meta().dy() / rawModel.meta().dy());
            rawDx = rawModel.meta().dx();
        }
        cmpLeft->setCrosshair(rawCross, st.crosshair, locked);
        cmpRight->setCrosshair(cross, st.crosshair, locked);
        const double dx = model.meta().dx();
        xy->setScaleBar(dx);
        cmpLeft->setScaleBar(rawDx);
        cmpRight->setScaleBar(dx);
        const QString zt = QStringLiteral("z %1 / %2  t %3 / %4  %5 %")
                               .arg(cz).arg(nz() - 1).arg(curT()).arg(nt() - 1).arg(std::lround(st.zoom * 100.0));
        xy->setTitle(QStringLiteral("XY  ") + zt);
        yz->setTitle(QStringLiteral("YZ"));
        xz->setTitle(QStringLiteral("XZ"));
        mip->setTitle(QStringLiteral("MIP · Z"));
        cmpLeft->setTitle(QStringLiteral("01 Load · raw  ") + zt);
        const int viewed = wb.viewedIndex();
        const QString name = viewed >= 0 && viewed < wb.pipeline().size()
                                 ? num2(viewed) + QLatin1Char(' ') + QString::fromStdString(wb.pipeline().at(viewed).name)
                                 : QString();
        cmpRight->setTitle(name);
        xy->setMeasure(measure, QString());
        updateMeasureText();
        xy->setRoi(roi);
    }

    void ViewerWidget::Impl::applyDirty() {
        if (!model.valid()) {
            for (SlicePane* p : {xy, yz, xz, mip, cmpLeft, cmpRight}) p->clearContent();
            xy->setMessage(wb.hasDataset() ? QStringLiteral("Nothing to display") : QStringLiteral("Open a dataset (File ▸ Open dataset…)"));
            if (volume) volume->clearVolumes();
            dirty = Dirty{};
            return;
        }
        xy->setMessage({});
        const ViewState& s = vs();
        const Index t = curT(), z = curZ();
        if (s.mode == ViewMode::Ortho) {
            if (dirty.xy) {
                model.renderXY(t, z, s, xyFactor, xyImg);
                if (s.labels) model.overlayLabelsXY(t, z, xyFactor, s, xyImg);
                xy->setContent(xyImg, xyFactor, nx(), ny());
                dirty.xy = false;
            }
            if (dirty.xz) {
                model.renderXZ(t, curY(), s, xzImg);
                if (s.labels) model.overlayLabelsXZ(t, curY(), s, xzImg);
                xz->setContent(xzImg, 1, nx(), nz());
                dirty.xz = false;
            }
            if (dirty.yz) {
                model.renderYZ(t, curX(), s, yzImg);
                if (s.labels) model.overlayLabelsYZ(t, curX(), s, yzImg);
                yz->setContent(yzImg, 1, nz(), ny());
                dirty.yz = false;
            }
            if (dirty.mip) {
                model.renderMIP(t, s, mipFactor, mipImg);
                mip->setContent(mipImg, mipFactor, nx(), ny());
                dirty.mip = false;
            }
            if (model.volumeTooLarge()) refreshHints();
        } else if (s.mode == ViewMode::Compare) {
            if (dirty.cmp) {
                if (rawModel.valid()) {
                    const Index rt = std::clamp<Index>(t, 0, rawModel.dims().t - 1);
                    const Index rz = std::clamp<Index>(z, 0, rawModel.dims().z - 1);
                    rawModel.renderXY(rt, rz, s, cmpFactor, cmpLeftImg);
                    cmpLeft->setContent(cmpLeftImg, cmpFactor, rawModel.dims().x, rawModel.dims().y);
                } else {
                    cmpLeft->clearContent();
                }
                model.renderXY(t, z, s, cmpFactor, cmpRightImg);
                if (s.labels) model.overlayLabelsXY(t, z, cmpFactor, s, cmpRightImg);
                cmpRight->setContent(cmpRightImg, cmpFactor, nx(), ny());
                dirty.cmp = false;
            }
        } else {
            if (dirty.vol) {
                renderVolume();
                dirty.vol = false;
            }
        }
        layoutPanes();
    }

    void ViewerWidget::Impl::renderVolume() {
        const ViewState& s = vs();
        const Index t = curT();
        std::vector<VolumeView::Channel> chans;
        quint64 key = static_cast<quint64>(reinterpret_cast<std::uintptr_t>(model.output().get())) ^ (static_cast<quint64>(t) << 48);
        for (Index c = 0; c < model.dims().c; ++c) {
            if (!s.channelOn(c)) continue;
            const float* v = model.volume(c, t);
            if (!v) continue;
            VolumeView::Channel ch;
            ch.data = v;
            ch.z = nz();
            ch.y = ny();
            ch.x = nx();
            const DisplayWindow w = model.window(c, t);
            ch.lo = w.lo;
            ch.hi = w.hi;
            if (model.meta().rgb) {
                ch.color = {c == 0 ? 1.f : 0.f, c == 1 ? 1.f : 0.f, c == 2 ? 1.f : 0.f};
            } else if (static_cast<std::size_t>(c) < model.meta().channels.size()) {
                ch.color = model.meta().channels[static_cast<std::size_t>(c)].color;
            }
            key ^= (static_cast<quint64>(c + 1) * 0x9e3779b97f4a7c15ull) ^ static_cast<quint64>(std::hash<float>{}(w.lo) * 31 + std::hash<float>{}(w.hi));
            chans.push_back(ch);
        }
        if (volume) volume->setVolumes(key, chans, model.meta().voxelUm);
    }

    // --- zoom / pan --------------------------------------------------------------------

    void ViewerWidget::Impl::setZoomPan(double zoom, double panX, double panY) {
        ViewState s = vs();
        s.zoom = std::clamp(zoom, kMinZoom, kMaxZoom);
        s.panX = panX;
        s.panY = panY;
        if (s.zoom == 1.0 && zoom == 1.0 && panX == 0.0 && panY == 0.0) {
            s.panX = s.panY = 0.0;
        }
        wb.setViewState(s);
    }

    void ViewerWidget::Impl::zoomAround(double factor, const QPointF& anchor) {
        if (!model.valid()) return;
        const ViewState& s = vs();
        const double newZoom = std::clamp(s.zoom * factor, kMinZoom, kMaxZoom);
        if (newZoom == s.zoom) return;
        const SlicePane::View v = xy->view();
        const double fz = v.zx / s.zoom;   // px per voxel at fit
        const QPointF voxel = xy->toVoxel(anchor);
        const double z2 = fz * newZoom;
        const double ox = anchor.x() - voxel.x() * z2, oy = anchor.y() - voxel.y() * z2;
        const double panX = ox - (xy->width() - nx() * z2) / 2.0;
        const double panY = oy - (xy->height() - ny() * z2) / 2.0;
        setZoomPan(newZoom, panX, panY);
    }

    void ViewerWidget::Impl::fit() { setZoomPan(1.0, 0.0, 0.0); }

    // --- tools ---------------------------------------------------------------------------

    std::uint32_t ViewerWidget::Impl::labelAt(Index z, Index y, Index x) const {
        const LabelVolume* L = model.labels();
        if (!L) return 0;
        const Index t = curT();
        if (t >= L->t() || z < 0 || z >= L->z() || y < 0 || y >= L->y() || x < 0 || x >= L->x()) return 0;
        return L->at(t, z, y, x);
    }

    void ViewerWidget::Impl::paintAt(const QPointF& v, bool erase) {
        if (!model.valid()) return;
        const Index x = clampIndex(v.x(), nx()), y = clampIndex(v.y(), ny());
        wb.paintLabels(curZ(), y, x, erase);
    }

    void ViewerWidget::Impl::onXYPressed(const QPointF& v, Qt::MouseButton b, Qt::KeyboardModifiers m) {
        if (!model.valid() || b != Qt::LeftButton) return;
        const ViewState& s = vs();
        const Index x = clampIndex(v.x(), nx()), y = clampIndex(v.y(), ny()), z = curZ();
        switch (s.tool) {
            case ViewerTool::Navigate:
                xy->setCursor(Qt::ClosedHandCursor);
                break;
            case ViewerTool::Probe:
                if (xy->inside(v)) wb.setCrosshair(x, y, z);
                break;
            case ViewerTool::Measure:
                if (measure.size() >= 3) measure.clear();
                measure.push_back(v);
                layoutPanes();
                break;
            case ViewerTool::Roi:
                roiStart = v;
                roi = QRectF();
                layoutPanes();
                break;
            case ViewerTool::Paint: {
                const bool erase = s.paintTool == PaintTool::Erase || m.testFlag(Qt::AltModifier);
                switch (s.paintTool) {
                    case PaintTool::Brush:
                    case PaintTool::Erase:
                    case PaintTool::Lasso:
                        painting = true;
                        lastPaint = v;
                        paintAt(v, erase);
                        break;
                    case PaintTool::Fill:
                        wb.fillLabel(z, y, x);
                        break;
                    case PaintTool::Pick: {
                        ViewState ns = s;
                        ns.selectedLabel = labelAt(z, y, x);
                        wb.setViewState(ns);
                        break;
                    }
                    case PaintTool::Merge: {
                        const std::uint32_t id = labelAt(z, y, x);
                        if (id == 0) break;
                        if (mergeFirst == 0 || mergeFirst == id) {
                            mergeFirst = id;
                            ViewState ns = s;
                            ns.selectedLabel = id;
                            wb.setViewState(ns);
                        } else {
                            wb.mergeLabels({mergeFirst, id});
                            mergeFirst = 0;
                        }
                        refreshHints();
                        break;
                    }
                    case PaintTool::Split: {
                        if (!splitPending) {
                            splitA = {z, y, x};
                            splitPending = labelAt(z, y, x) != 0;
                        } else {
                            const std::uint32_t id = labelAt(splitA[0], splitA[1], splitA[2]);
                            if (id != 0) wb.splitLabel(id, splitA, {z, y, x});
                            splitPending = false;
                        }
                        refreshHints();
                        break;
                    }
                    case PaintTool::Delete: {
                        const std::uint32_t id = labelAt(z, y, x);
                        if (id != 0) wb.deleteLabel(id);
                        break;
                    }
                }
                break;
            }
        }
    }

    void ViewerWidget::Impl::onXYDragged(const QPointF& v, const QPointF& delta, Qt::MouseButton b, Qt::KeyboardModifiers m) {
        if (!model.valid()) return;
        const ViewState& s = vs();
        if (b == Qt::MiddleButton || (b == Qt::LeftButton && s.tool == ViewerTool::Navigate)) {
            setZoomPan(s.zoom, s.panX + delta.x(), s.panY + delta.y());
            return;
        }
        if (b != Qt::LeftButton) return;
        switch (s.tool) {
            case ViewerTool::Probe:
                if (xy->inside(v)) wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
                break;
            case ViewerTool::Roi:
                roi = QRectF(roiStart, v).normalized();
                xy->setRoi(roi);
                break;
            case ViewerTool::Paint:
                if (painting) {
                    const bool erase = s.paintTool == PaintTool::Erase || m.testFlag(Qt::AltModifier);
                    // stamp along the path so fast strokes stay continuous
                    const double spacing = std::max(1.0, s.brushPx / 4.0);
                    const QPointF d = v - lastPaint;
                    const double len = std::hypot(d.x(), d.y());
                    const int n = std::max(1, static_cast<int>(len / spacing));
                    for (int i = 1; i <= n; ++i) paintAt(lastPaint + d * (static_cast<double>(i) / n), erase);
                    lastPaint = v;
                }
                break;
            default:
                break;
        }
    }

    void ViewerWidget::Impl::onXYReleased(const QPointF&, Qt::MouseButton b, Qt::KeyboardModifiers, bool) {
        if (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate) xy->setCursor(Qt::OpenHandCursor);
        painting = false;
    }

    void ViewerWidget::Impl::updateMeasureText() {
        if (measure.isEmpty()) {
            xy->setMeasure({}, QString());
            return;
        }
        const double dx = model.meta().dx(), dy = model.meta().dy();
        QString text;
        if (measure.size() >= 2) {
            const QPointF a = measure[0], b = measure[1];
            const double um = std::hypot((b.x() - a.x()) * dx, (b.y() - a.y()) * dy);
            text = QStringLiteral("%1 µm").arg(um, 0, 'f', 2);
        }
        if (measure.size() >= 3) {
            const QPointF a = measure[0], b = measure[1], c = measure[2];
            const double ux = (a.x() - b.x()) * dx, uy = (a.y() - b.y()) * dy;
            const double vx = (c.x() - b.x()) * dx, vy = (c.y() - b.y()) * dy;
            const double ang = std::acos(std::clamp((ux * vx + uy * vy) / std::max(1e-12, std::hypot(ux, uy) * std::hypot(vx, vy)), -1.0, 1.0));
            text += QStringLiteral("  ∠ %1°").arg(ang * 180.0 / M_PI, 0, 'f', 1);
        }
        xy->setMeasure(measure, text);
    }

    void ViewerWidget::Impl::hover(SlicePane::Kind kind, const QPointF& v) {
        if (!model.valid()) return;
        Index x = curX(), y = curY(), z = curZ();
        bool inside = true;
        switch (kind) {
            case SlicePane::Kind::XY:
            case SlicePane::Kind::MIP:
            case SlicePane::Kind::Compare:
                x = static_cast<Index>(std::floor(v.x()));
                y = static_cast<Index>(std::floor(v.y()));
                inside = x >= 0 && y >= 0 && x < nx() && y < ny();
                break;
            case SlicePane::Kind::XZ:
                x = static_cast<Index>(std::floor(v.x()));
                z = static_cast<Index>(std::floor(v.y()));
                inside = x >= 0 && z >= 0 && x < nx() && z < nz();
                break;
            case SlicePane::Kind::YZ:
                z = static_cast<Index>(std::floor(v.x()));
                y = static_cast<Index>(std::floor(v.y()));
                inside = z >= 0 && y >= 0 && z < nz() && y < ny();
                break;
        }
        if (!inside) {
            cursorText = QStringLiteral("cursor —");
        } else {
            Index c = 0;
            for (Index i = 0; i < model.dims().c; ++i)
                if (vs().channelOn(i)) {
                    c = i;
                    break;
                }
            std::optional<float> val;
            if (kind == SlicePane::Kind::XY || kind == SlicePane::Kind::Compare) val = model.valueAt(c, curT(), z, y, x);
            else if (const float* vol = model.volume(c, curT())) val = vol[(z * ny() + y) * nx() + x];
            QString vtext = val ? QString::number(*val, 'g', 5) : QStringLiteral("—");
            if (const LabelVolume* L = model.labels(); L && vs().labels) {
                const std::uint32_t id = labelAt(z, y, x);
                if (id) vtext += QStringLiteral(" · label %1").arg(id);
            }
            cursorText = QStringLiteral("cursor %1, %2, %3 · %4").arg(x).arg(y).arg(z).arg(vtext);
        }
        emit q->cursorChanged(cursorText);
    }

    // -------------------------------------------------------------------------
    // ViewerWidget
    // -------------------------------------------------------------------------

    ViewerWidget::ViewerWidget(WorkbenchBridge& bridge, QWidget* parent)
        : QWidget(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        setFocusPolicy(Qt::StrongFocus);
        impl_->build();
        impl_->connectSignals();
        impl_->rebuildOutput();
        impl_->applyViewStateDiff(impl_->vs());
    }

    ViewerWidget::~ViewerWidget() = default;

    void ViewerWidget::zoomIn() {
        if (impl_->vs().mode == ViewMode::Volume) {
            if (impl_->volume) impl_->volume->setZoom(impl_->volume->zoom() * 1.5);
            return;
        }
        impl_->zoomAround(1.5, QPointF(impl_->xy->width() / 2.0, impl_->xy->height() / 2.0));
    }

    void ViewerWidget::zoomOut() {
        if (impl_->vs().mode == ViewMode::Volume) {
            if (impl_->volume) impl_->volume->setZoom(impl_->volume->zoom() / 1.5);
            return;
        }
        impl_->zoomAround(1.0 / 1.5, QPointF(impl_->xy->width() / 2.0, impl_->xy->height() / 2.0));
    }

    void ViewerWidget::fitToWindow() {
        if (impl_->vs().mode == ViewMode::Volume && impl_->volume) impl_->volume->setZoom(1.0);
        impl_->fit();
    }

    void ViewerWidget::setPlaying(bool on) {
        if (on && impl_->nt() > 1) impl_->playTimer.start();
        else impl_->playTimer.stop();
        impl_->dims->setPlaying(impl_->playTimer.isActive());
    }

    bool ViewerWidget::playing() const { return impl_->playTimer.isActive(); }

    void ViewerWidget::autoContrast() {
        impl_->model.setWindowMode(DisplayModel::WindowMode::Auto);
        impl_->rawModel.setWindowMode(DisplayModel::WindowMode::Auto);
        impl_->dirty = Impl::Dirty{};
        impl_->scheduleUpdate();
    }

    void ViewerWidget::resetContrast() {
        impl_->model.setWindowMode(DisplayModel::WindowMode::Full);
        impl_->rawModel.setWindowMode(DisplayModel::WindowMode::Full);
        impl_->dirty = Impl::Dirty{};
        impl_->scheduleUpdate();
    }

    QImage ViewerWidget::grabView() const {
        const ViewState& s = impl_->vs();
        if (s.mode == ViewMode::Volume && impl_->volume) return impl_->volume->grabImage();
        QWidget* page = s.mode == ViewMode::Ortho ? impl_->orthoPage : impl_->comparePage;
        return page->grab().toImage();
    }

    QString ViewerWidget::cursorText() const { return impl_->cursorText; }
    QString ViewerWidget::zoomText() const { return impl_->zoomText; }

} // namespace sirius::app
