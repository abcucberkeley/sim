#include "qt/stack_view.hpp"
#include "qt/image_canvas.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QToolButton>
#include <QVBoxLayout>

namespace sirius::app {

    namespace {
        constexpr int kSliderSteps = 1000;

        // log10 with a floor six decades below the plane's peak, so zeros and
        // the noise floor stay finite and the window covers the useful range
        void logTransform(std::vector<double>& v) {
            double peak = 0.0;
            for (double x : v) peak = std::max(peak, x);
            if (!(peak > 0.0)) {
                std::fill(v.begin(), v.end(), 0.0);
                return;
            }
            const double floor = peak * 1e-6;
            for (double& x : v) x = std::log10(std::max(x, 0.0) + floor);
        }

        QToolButton* toolButton(QWidget* parent, const QString& text, const QString& tip) {
            auto* b = new QToolButton(parent);
            b->setText(text);
            b->setToolTip(tip);
            b->setAutoRaise(true);
            return b;
        }
    } // namespace

    StackView::StackView(QWidget* parent) : QWidget(parent) { buildUi(); }

    void StackView::buildUi() {
        xyCanvas_ = new ImageCanvas(this);
        xzCanvas_ = new ImageCanvas(this);
        yzCanvas_ = new ImageCanvas(this);
        xzCanvas_->hide();
        yzCanvas_->hide();

        // --- tool row ---
        auto* tools = new QHBoxLayout;
        auto* zoomOut = toolButton(this, QStringLiteral("-"), tr("Zoom out (mouse wheel)"));
        auto* zoomIn = toolButton(this, QStringLiteral("+"), tr("Zoom in (mouse wheel)"));
        auto* fit = toolButton(this, tr("Fit"), tr("Fit the image to the window"));
        auto* one = toolButton(this, tr("1:1"), tr("One screen pixel per image pixel"));
        zoomLabel_ = new QLabel(QStringLiteral("100%"), this);
        zoomLabel_->setMinimumWidth(48);
        selectTool_ = toolButton(this, tr("Select"), tr("Left drag draws a rectangle (otherwise it pans)"));
        selectTool_->setCheckable(true);
        crop_ = new QPushButton(tr("Crop"), this);
        crop_->setEnabled(false);
        crop_->setToolTip(tr("Open the selected rectangle (every slice) in a new tab"));
        ortho_ = new QCheckBox(tr("Ortho"), this);
        ortho_->setToolTip(tr("Show XZ and YZ views through the crosshair; click to move it"));
        physicalZ_ = new QCheckBox(tr("Physical z"), this);
        physicalZ_->setToolTip(tr("Scale the orthogonal views by dz / dx"));
        spectrumBox_ = new QCheckBox(tr("Spectrum"), this);
        spectrumBox_->setToolTip(tr("Show the centered |FFT| of the displayed planes"));
        logBox_ = new QCheckBox(tr("Log"), this);
        logBox_->setToolTip(tr("Display log10 of the intensity"));
        tools->addWidget(zoomOut);
        tools->addWidget(zoomIn);
        tools->addWidget(fit);
        tools->addWidget(one);
        tools->addWidget(zoomLabel_);
        tools->addSpacing(12);
        tools->addWidget(selectTool_);
        tools->addWidget(crop_);
        tools->addSpacing(12);
        tools->addWidget(ortho_);
        tools->addWidget(physicalZ_);
        tools->addSpacing(12);
        tools->addWidget(spectrumBox_);
        tools->addWidget(logBox_);
        tools->addStretch(1);

        // --- canvases: XY with YZ to the right and XZ below ---
        auto* grid = new QGridLayout;
        grid->setContentsMargins(0, 0, 0, 0);
        grid->addWidget(xyCanvas_, 0, 0);
        grid->addWidget(yzCanvas_, 0, 1);
        grid->addWidget(xzCanvas_, 1, 0);
        grid->setColumnStretch(0, 3);
        grid->setColumnStretch(1, 1);
        grid->setRowStretch(0, 3);
        grid->setRowStretch(1, 1);

        // --- slice + window ---
        slice_ = new QSlider(Qt::Horizontal, this);
        slice_->setEnabled(false);
        sliceLabel_ = new QLabel(tr("-"), this);
        sliceLabel_->setMinimumWidth(80);
        auto* sliceRow = new QHBoxLayout;
        sliceRow->addWidget(new QLabel(tr("Slice"), this));
        sliceRow->addWidget(slice_, 1);
        sliceRow->addWidget(sliceLabel_);

        minSpin_ = new QDoubleSpinBox(this);
        maxSpin_ = new QDoubleSpinBox(this);
        for (QDoubleSpinBox* s : {minSpin_, maxSpin_}) {
            s->setRange(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
            s->setDecimals(4);
            s->setKeyboardTracking(false);
            s->setMinimumWidth(110);
        }
        minSlider_ = new QSlider(Qt::Horizontal, this);
        maxSlider_ = new QSlider(Qt::Horizontal, this);
        minSlider_->setRange(0, kSliderSteps);
        maxSlider_->setRange(0, kSliderSteps);
        auto* autoBtn = new QPushButton(tr("Auto"), this);
        autoBtn->setToolTip(tr("Window from the 0.1% and 99.9% percentiles of the displayed plane"));
        auto* resetBtn = new QPushButton(tr("Reset"), this);
        resetBtn->setToolTip(tr("Window from the minimum and maximum of the displayed plane"));
        auto* windowRow = new QHBoxLayout;
        windowRow->addWidget(new QLabel(tr("Min"), this));
        windowRow->addWidget(minSlider_, 1);
        windowRow->addWidget(minSpin_);
        windowRow->addSpacing(8);
        windowRow->addWidget(new QLabel(tr("Max"), this));
        windowRow->addWidget(maxSlider_, 1);
        windowRow->addWidget(maxSpin_);
        windowRow->addWidget(autoBtn);
        windowRow->addWidget(resetBtn);

        status_ = new QLabel(this);
        status_->setTextInteractionFlags(Qt::TextSelectableByMouse);

        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addLayout(tools);
        layout->addLayout(grid, 1);
        layout->addLayout(sliceRow);
        layout->addLayout(windowRow);
        layout->addWidget(status_);

        // --- wiring ---
        connect(zoomOut, &QToolButton::clicked, this, [this] { xyCanvas_->zoomBy(0.8); });
        connect(zoomIn, &QToolButton::clicked, this, [this] { xyCanvas_->zoomBy(1.25); });
        connect(fit, &QToolButton::clicked, this, [this] { xyCanvas_->fitToWindow(); });
        connect(one, &QToolButton::clicked, this, [this] { xyCanvas_->setZoom(1.0); });
        connect(xyCanvas_, &ImageCanvas::zoomChanged, this, [this](double z) {
            zoomLabel_->setText(QStringLiteral("%1%").arg(std::lround(z * 100.0)));
        });
        connect(selectTool_, &QToolButton::toggled, this, [this](bool on) { xyCanvas_->setSelectionMode(on); });
        connect(xyCanvas_, &ImageCanvas::selectionChanged, this,
                [this](QRect r) { crop_->setEnabled(volume_ && !r.isNull()); });
        connect(crop_, &QPushButton::clicked, this, [this] {
            const QRect r = xyCanvas_->selection();
            if (!r.isNull()) emit cropRequested(r);
        });
        connect(ortho_, &QCheckBox::toggled, this, [this](bool on) {
            xzCanvas_->setVisible(on);
            yzCanvas_->setVisible(on);
            updateCrosshairs();
            renderOrtho();
        });
        connect(physicalZ_, &QCheckBox::toggled, this, [this] { renderOrtho(); });
        connect(spectrumBox_, &QCheckBox::toggled, this, [this] {
            renderAll();
            autoWindow(true);
        });
        connect(logBox_, &QCheckBox::toggled, this, [this] {
            renderAll();
            autoWindow(true);
        });
        connect(slice_, &QSlider::valueChanged, this, [this](int) {
            renderXY();
            updateCrosshairs();
            renderOrtho();
        });
        connect(minSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
            if (!updatingControls_) setWindow({v, std::max(v, maxSpin_->value())});
        });
        connect(maxSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) {
            if (!updatingControls_) setWindow({std::min(v, minSpin_->value()), v});
        });
        auto fromSlider = [this](int step) { return spanLo_ + (spanHi_ - spanLo_) * step / kSliderSteps; };
        connect(minSlider_, &QSlider::valueChanged, this, [this, fromSlider](int s) {
            if (updatingControls_) return;
            const double v = fromSlider(s);
            setWindow({v, std::max(v, window_.hi)});
        });
        connect(maxSlider_, &QSlider::valueChanged, this, [this, fromSlider](int s) {
            if (updatingControls_) return;
            const double v = fromSlider(s);
            setWindow({std::min(v, window_.lo), v});
        });
        connect(autoBtn, &QPushButton::clicked, this, [this] { autoWindow(true); });
        connect(resetBtn, &QPushButton::clicked, this, [this] { autoWindow(false); });

        connect(xyCanvas_, &ImageCanvas::hovered, this, &StackView::hoverXY);
        connect(xzCanvas_, &ImageCanvas::hovered, this, &StackView::hoverXZ);
        connect(yzCanvas_, &ImageCanvas::hovered, this, &StackView::hoverYZ);
        connect(xyCanvas_, &ImageCanvas::clicked, this, [this](int x, int y) { setCrosshair(x, y, slice_->value()); });
        connect(xzCanvas_, &ImageCanvas::clicked, this, [this](int x, int z) { setCrosshair(x, crossY_, z); });
        connect(yzCanvas_, &ImageCanvas::clicked, this, [this](int z, int y) { setCrosshair(crossX_, y, z); });
    }

    // --- volume ----------------------------------------------------------

    void StackView::setVolume(std::shared_ptr<const Buffer<double>> volume) {
        if (!volume || volume->rank() != 3 || volume->empty() || !volume->device().isCpu()) {
            clear();
            return;
        }
        const bool sameShape = volume_ && volume_->shape() == volume->shape();
        volume_ = std::move(volume);
        const Index nz = volume_->dim(0), ny = volume_->dim(1), nx = volume_->dim(2);
        {
            const QSignalBlocker block(slice_);
            slice_->setRange(0, static_cast<int>(nz) - 1);
            slice_->setValue(sameShape ? std::min<int>(slice_->value(), static_cast<int>(nz) - 1)
                                       : static_cast<int>(nz / 2));
        }
        slice_->setEnabled(nz > 1);
        ortho_->setEnabled(nz > 1);
        physicalZ_->setEnabled(nz > 1);
        if (!sameShape) {
            crossX_ = nx / 2;
            crossY_ = ny / 2;
        }
        renderAll();
        if (!sameShape || !window_.valid()) autoWindow(true);
        updateCrosshairs();
        crop_->setEnabled(!xyCanvas_->selection().isNull());
    }

    void StackView::clear() {
        volume_.reset();
        for (Plane* p : {&xy_, &xz_, &yz_}) {
            p->values.clear();
            p->gray.clear();
            p->rows = p->cols = 0;
        }
        for (ImageCanvas* c : {xyCanvas_, xzCanvas_, yzCanvas_}) c->clearImage();
        slice_->setEnabled(false);
        sliceLabel_->setText(tr("-"));
        crop_->setEnabled(false);
        status_->clear();
    }

    int StackView::currentSlice() const { return slice_->value(); }

    void StackView::setPixelSize(double dx, double dy, double dz) {
        dx_ = dx > 0 ? dx : 1.0;
        dy_ = dy > 0 ? dy : 1.0;
        dz_ = dz > 0 ? dz : 1.0;
        updateOverlays();
        renderOrtho();
    }

    void StackView::setVolumeIsSpectrum(bool isSpectrum) {
        volumeIsSpectrum_ = isSpectrum;
        spectrumBox_->setVisible(!isSpectrum);
        if (isSpectrum) {
            const QSignalBlocker block(spectrumBox_);
            spectrumBox_->setChecked(false);
        }
        updateOverlays();
    }

    void StackView::setOverlay(std::optional<SpectrumOverlay> overlay) {
        overlay_ = std::move(overlay);
        updateOverlays();
    }

    void StackView::setLogScale(bool on) { logBox_->setChecked(on); }
    bool StackView::logScale() const { return logBox_->isChecked(); }
    void StackView::setSpectrumMode(bool on) { spectrumBox_->setChecked(on); }
    bool StackView::spectrumMode() const { return spectrumBox_->isChecked() && !volumeIsSpectrum_; }

    // --- rendering -------------------------------------------------------

    void StackView::preparePlane(const double* src, Index rows, Index cols, Plane& plane) {
        plane.rows = rows;
        plane.cols = cols;
        plane.values.resize(static_cast<std::size_t>(rows * cols));
        plane.gray.resize(plane.values.size());
        if (spectrumMode()) spectrum_.magnitude(src, rows, cols, plane.values.data());
        else std::copy_n(src, rows * cols, plane.values.data());
        if (logBox_->isChecked()) logTransform(plane.values);
    }

    void StackView::showPlane(ImageCanvas* canvas, Plane& plane) {
        if (plane.rows == 0) {
            canvas->clearImage();
            return;
        }
        mapToGray8(plane.values.data(), plane.rows * plane.cols, window_, plane.gray.data());
        // QImage wraps the gray buffer without copying; it stays valid because
        // the buffer is only resized in preparePlane before the next setImage
        canvas->setImage(QImage(plane.gray.data(), static_cast<int>(plane.cols), static_cast<int>(plane.rows),
                                static_cast<int>(plane.cols), QImage::Format_Grayscale8));
    }

    void StackView::renderXY() {
        if (!volume_) return;
        const Index ny = volume_->dim(1), nx = volume_->dim(2);
        const Index z = slice_->value();
        preparePlane(volume_->data() + z * ny * nx, ny, nx, xy_);
        showPlane(xyCanvas_, xy_);
        sliceLabel_->setText(QStringLiteral("%1 / %2").arg(z + 1).arg(volume_->dim(0)));
        updateOverlays();
        updateStatus();
    }

    void StackView::renderOrtho() {
        if (!volume_ || !ortho_->isChecked()) return;
        const Index nz = volume_->dim(0), ny = volume_->dim(1), nx = volume_->dim(2);
        std::vector<double> tmp(static_cast<std::size_t>(std::max(nz * nx, nz * ny)));
        sliceXZ(volume_->view(), crossY_, tmp.data());
        preparePlane(tmp.data(), nz, nx, xz_);
        sliceYZ(volume_->view(), crossX_, tmp.data());
        preparePlane(tmp.data(), ny, nz, yz_);
        const double zScale = physicalZ_->isChecked() ? dz_ / dx_ : 1.0;
        xzCanvas_->setPixelAspect(1.0, zScale);
        yzCanvas_->setPixelAspect(zScale * dx_ / dy_, 1.0);
        showPlane(xzCanvas_, xz_);
        showPlane(yzCanvas_, yz_);
    }

    void StackView::renderAll() {
        renderXY();
        renderOrtho();
    }

    // --- window ----------------------------------------------------------

    void StackView::setWindow(DisplayRange r) {
        if (!(r.hi > r.lo)) r.hi = r.lo + 1e-12;
        window_ = r;
        spanLo_ = std::min(spanLo_, r.lo);
        spanHi_ = std::max(spanHi_, r.hi);
        syncWindowControls();
        for (auto [canvas, plane] : {std::pair{xyCanvas_, &xy_}, std::pair{xzCanvas_, &xz_}, std::pair{yzCanvas_, &yz_}})
            if (canvas->isVisible() || canvas == xyCanvas_) showPlane(canvas, *plane);
        updateStatus();
    }

    void StackView::syncWindowControls() {
        updatingControls_ = true;
        const double span = spanHi_ - spanLo_;
        auto toStep = [&](double v) {
            return span > 0 ? static_cast<int>(std::lround((v - spanLo_) / span * kSliderSteps)) : 0;
        };
        minSpin_->setValue(window_.lo);
        maxSpin_->setValue(window_.hi);
        minSlider_->setValue(std::clamp(toStep(window_.lo), 0, kSliderSteps));
        maxSlider_->setValue(std::clamp(toStep(window_.hi), 0, kSliderSteps));
        updatingControls_ = false;
    }

    void StackView::autoWindow(bool percentile) {
        if (!volume_ || xy_.values.empty()) return;
        const Index n = static_cast<Index>(xy_.values.size());
        const DisplayRange full = minMaxRange(xy_.values.data(), n);
        spanLo_ = full.lo;
        spanHi_ = full.hi > full.lo ? full.hi : full.lo + 1.0;
        setWindow(percentile ? percentileRange(xy_.values.data(), n, 0.001, 0.999) : full);
    }

    // --- overlays / crosshair -------------------------------------------

    bool StackView::frequencyReadout() const { return volumeIsSpectrum_ || spectrumMode(); }

    std::array<double, 2> StackView::frequencyOf(Index cols, Index rows, int x, int y) const {
        SpectrumGeometry g{rows, cols, 1.0 / (static_cast<double>(cols) * dx_), 1.0 / (static_cast<double>(rows) * dy_)};
        return {(x - static_cast<double>(cols / 2)) * g.dkx, (y - static_cast<double>(rows / 2)) * g.dky};
    }

    void StackView::updateOverlays() {
        std::vector<CanvasOverlay> items;
        if (volume_ && overlay_ && frequencyReadout()) {
            const Index ny = volume_->dim(1), nx = volume_->dim(2);
            const SpectrumGeometry g{ny, nx, 1.0 / (static_cast<double>(nx) * dx_), 1.0 / (static_cast<double>(ny) * dy_)};
            const double cx = static_cast<double>(nx / 2) + 0.5, cy = static_cast<double>(ny / 2) + 0.5;
            const SpectrumOverlay& o = *overlay_;
            if (o.supportRadius > 0.0) {
                const auto r = g.radiusPixels(o.supportRadius);
                items.push_back({CanvasOverlay::Kind::Circle, cx, cy, r[0], r[1], QColor(255, 255, 255, 180),
                                 tr("2NA/λ")});
            }
            if (o.showOrders) {
                auto marks = [&](const std::vector<std::array<double, 2>>& k0s, QColor color, bool withAmps,
                                 CanvasOverlay::Kind kind) {
                    for (std::size_t d = 0; d < k0s.size(); ++d)
                        for (int order = 1; order < o.norders; ++order)
                            for (int sign : {1, -1}) {
                                const auto p = g.pixelOf(sign * order * k0s[d][0], sign * order * k0s[d][1]);
                                CanvasOverlay item{kind, p[0] + 0.5, p[1] + 0.5, 3.0, 3.0, color, {}};
                                if (sign > 0) {
                                    item.text = tr("d%1 o%2").arg(d).arg(order);
                                    if (withAmps && d < o.ampMagnitude.size() &&
                                        static_cast<std::size_t>(order) < o.ampMagnitude[d].size())
                                        item.text += tr(" |a|=%1").arg(o.ampMagnitude[d][static_cast<std::size_t>(order)], 0, 'f', 2);
                                }
                                items.push_back(item);
                            }
                };
                marks(o.predictedK0, QColor(255, 220, 0), false, CanvasOverlay::Kind::Cross);
                marks(o.fittedK0, QColor(0, 255, 255), true, CanvasOverlay::Kind::Circle);
            }
        }
        xyCanvas_->setOverlays(std::move(items));
    }

    void StackView::updateCrosshairs() {
        const bool on = volume_ && ortho_->isChecked();
        const double z = slice_->value() + 0.5;
        xyCanvas_->setCrosshair(QPointF(crossX_ + 0.5, crossY_ + 0.5), on);
        xzCanvas_->setCrosshair(QPointF(crossX_ + 0.5, z), on);
        yzCanvas_->setCrosshair(QPointF(z, crossY_ + 0.5), on);
    }

    void StackView::setCrosshair(Index x, Index y, Index z) {
        if (!volume_) return;
        crossX_ = std::clamp<Index>(x, 0, volume_->dim(2) - 1);
        crossY_ = std::clamp<Index>(y, 0, volume_->dim(1) - 1);
        z = std::clamp<Index>(z, 0, volume_->dim(0) - 1);
        if (z != slice_->value()) slice_->setValue(static_cast<int>(z));   // renders XY + ortho
        else {
            updateCrosshairs();
            renderOrtho();
        }
    }

    // --- status ----------------------------------------------------------

    void StackView::updateStatus(const QString& text) {
        if (!volume_) {
            status_->clear();
            return;
        }
        if (!text.isEmpty()) {
            status_->setText(text);
            return;
        }
        status_->setText(tr("%1 x %2 x %3   window [%4, %5]   zoom %6%")
                             .arg(volume_->dim(0)).arg(volume_->dim(1)).arg(volume_->dim(2))
                             .arg(window_.lo, 0, 'g', 5).arg(window_.hi, 0, 'g', 5)
                             .arg(std::lround(xyCanvas_->zoom() * 100.0)));
    }

    void StackView::hoverXY(int x, int y) {
        if (!volume_ || x < 0 || xy_.rows == 0) {
            updateStatus();
            return;
        }
        const Index z = slice_->value();
        const double shown = xy_.values[static_cast<std::size_t>(y * xy_.cols + x)];
        QString s;
        if (frequencyReadout()) {
            const auto k = frequencyOf(xy_.cols, xy_.rows, x, y);
            s = tr("(z %1, ky %2, kx %3) = %4    |k| = %5 1/um  (%6 um)")
                    .arg(z).arg(k[1], 0, 'f', 3).arg(k[0], 0, 'f', 3).arg(shown, 0, 'g', 6)
                    .arg(std::hypot(k[0], k[1]), 0, 'f', 3)
                    .arg(std::hypot(k[0], k[1]) > 0 ? 1.0 / std::hypot(k[0], k[1]) : 0.0, 0, 'f', 3);
        } else {
            const double v = volume_->data()[(z * xy_.rows + y) * xy_.cols + x];
            s = tr("(z %1, y %2, x %3) = %4").arg(z).arg(y).arg(x).arg(v, 0, 'g', 7);
            if (logBox_->isChecked()) s += tr("   (log %1)").arg(shown, 0, 'g', 5);
        }
        updateStatus(s);
    }

    void StackView::hoverXZ(int x, int z) {
        if (!volume_ || x < 0 || xz_.rows == 0) {
            updateStatus();
            return;
        }
        const double v = xz_.values[static_cast<std::size_t>(z * xz_.cols + x)];
        updateStatus(tr("XZ at y %1: (z %2, x %3) = %4").arg(crossY_).arg(z).arg(x).arg(v, 0, 'g', 7));
    }

    void StackView::hoverYZ(int z, int y) {
        if (!volume_ || z < 0 || yz_.rows == 0) {
            updateStatus();
            return;
        }
        const double v = yz_.values[static_cast<std::size_t>(y * yz_.cols + z)];
        updateStatus(tr("YZ at x %1: (z %2, y %3) = %4").arg(crossX_).arg(z).arg(y).arg(v, 0, 'g', 7));
    }

} // namespace sirius::app
