#include "qt/band_view.hpp"
#include "qt/image_canvas.hpp"
#include "qt/stack_view.hpp"

#include <algorithm>
#include <cmath>
#include <exception>

#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QVBoxLayout>

#include "core/volume_ops.hpp"

namespace sirius::app {

    namespace {
        SpectrumOverlay overlayFor(const SIMParameters& p, Index nz, const SimFit* fit, bool showOrders) {
            SpectrumOverlay o;
            o.supportRadius = otfSupportRadius(p);
            o.norders = p.norders > 0 ? p.norders : p.nphases / 2 + 1;
            o.predictedK0 = predictedK0(p, nz);
            if (fit) {
                o.fittedK0 = fit->k0;
                for (const auto& amps : fit->amps) {
                    std::vector<double> mags;
                    for (const auto& a : amps) mags.push_back(std::abs(a));
                    o.ampMagnitude.push_back(std::move(mags));
                }
            }
            o.showOrders = showOrders;
            return o;
        }

        QString bandItemName(int item) {
            if (item == 0) return QObject::tr("order 0");
            const int order = (item + 1) / 2;
            return item % 2 == 1 ? QObject::tr("order +%1").arg(order) : QObject::tr("order -%1").arg(order);
        }

        // Storage band index and side of a band item.
        void decodeItem(int item, int& band, BandSide& side) {
            const int order = item == 0 ? 0 : (item + 1) / 2;
            band = order == 0 ? 0 : 2 * order - 1;
            side = item == 0 ? BandSide::ReOnly : (item % 2 == 1 ? BandSide::Plus : BandSide::Minus);
        }

        const QString kStageTip = QObject::tr("Separated: the bands right after unmixing the phases and the band "
                                              "FFT, before any filtering. Wiener filtered: the same bands after the "
                                              "generalized Wiener filter, apodization and singularity suppression, "
                                              "i.e. what gets shifted and summed into the result.");
    } // namespace

    // --- BandGridView ----------------------------------------------------

    BandGridView::BandGridView(QWidget* parent) : QWidget(parent) {
        stage_ = new QComboBox(this);
        stage_->addItem(tr("Separated (before filtering)"));
        stage_->addItem(tr("Wiener filtered (as assembled)"));
        stage_->setToolTip(kStageTip);
        slice_ = new QSlider(Qt::Horizontal, this);
        slice_->setToolTip(tr("kz plane shown in every cell (centered: the middle is kz = 0)"));
        sliceLabel_ = new QLabel(tr("-"), this);
        log_ = new QCheckBox(tr("Log"), this);
        log_->setChecked(true);
        log_->setToolTip(tr("Display log10 of the band magnitudes"));
        auto* autoBtn = new QPushButton(tr("Auto window"), this);
        autoBtn->setToolTip(tr("Shared display window from the percentiles of all cells at this kz"));
        info_ = new QLabel(tr("Enable \"Capture intermediate spectra\" in the dock and reconstruct to see the "
                              "separated and filtered band spectra here."), this);
        info_->setWordWrap(true);

        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel(tr("Stage"), this));
        row->addWidget(stage_);
        row->addSpacing(12);
        row->addWidget(new QLabel(tr("kz"), this));
        row->addWidget(slice_, 1);
        row->addWidget(sliceLabel_);
        row->addSpacing(12);
        row->addWidget(log_);
        row->addWidget(autoBtn);

        gridHost_ = new QWidget(this);
        grid_ = new QGridLayout(gridHost_);
        grid_->setSpacing(6);
        auto* scroll = new QScrollArea(this);
        scroll->setWidget(gridHost_);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);

        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addLayout(row);
        layout->addWidget(info_);
        layout->addWidget(scroll, 1);

        connect(stage_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] {
            renderCells();
            autoWindow();
        });
        connect(slice_, &QSlider::valueChanged, this, [this] { renderCells(); });
        connect(log_, &QCheckBox::toggled, this, [this] {
            renderCells();
            autoWindow();
        });
        connect(autoBtn, &QPushButton::clicked, this, [this] { autoWindow(); });
    }

    void BandGridView::setResult(std::shared_ptr<const SimDiagnostics> diagnostics, SimFit fit,
                                 SIMParameters params) {
        if (!diagnostics || !diagnostics->captured) {
            clear();
            return;
        }
        diag_ = std::move(diagnostics);
        fit_ = std::move(fit);
        params_ = std::move(params);
        {
            const QSignalBlocker block(slice_);
            slice_->setRange(0, static_cast<int>(diag_->nz) - 1);
            slice_->setValue(static_cast<int>(diag_->nz / 2));
        }
        slice_->setEnabled(diag_->nz > 1);
        rebuildGrid();
        renderCells();
        autoWindow();
        info_->setText(tr("Rows: directions, columns: bands. Each cell is the centered |spectrum| of that band on "
                          "the %1 x %2 x %3 data grid; the white circle is the OTF support 2NA/λ and the cyan "
                          "circles on order 0 are the fitted pattern vectors. Order o is the object spectrum "
                          "shifted by -o k0, so its center corresponds to frequency o k0. Double-click a cell "
                          "to open it in a full viewer.")
                           .arg(diag_->nz).arg(diag_->ny).arg(diag_->nx));
    }

    void BandGridView::clear() {
        diag_.reset();
        for (Cell& c : cells_) {
            c.title->deleteLater();
            c.canvas->deleteLater();
        }
        cells_.clear();
        info_->setText(tr("Enable \"Capture intermediate spectra\" in the dock and reconstruct to see the "
                          "separated and filtered band spectra here."));
    }

    void BandGridView::rebuildGrid() {
        for (Cell& c : cells_) {
            c.title->deleteLater();
            c.canvas->deleteLater();
        }
        cells_.clear();
        const int norders = (diag_->nbands + 1) / 2;
        const int items = 2 * norders - 1;
        for (int d = 0; d < diag_->ndirs; ++d)
            for (int item = 0; item < items; ++item) {
                Cell c;
                c.direction = d;
                c.item = item;
                c.title = new QLabel(tr("direction %1 · %2").arg(d).arg(bandItemName(item)), gridHost_);
                c.title->setAlignment(Qt::AlignCenter);
                c.canvas = new ImageCanvas(gridHost_);
                c.canvas->setNavigationLocked(true);
                c.canvas->setMinimumSize(140, 140);
                c.canvas->setToolTip(tr("Double-click to open this band in a full viewer"));
                connect(c.canvas, &ImageCanvas::doubleClicked, this,
                        [this, d, item] { emit openRequested(d, item, stage_->currentIndex()); });
                grid_->addWidget(c.title, 2 * d, item);
                grid_->addWidget(c.canvas, 2 * d + 1, item);
                grid_->setRowStretch(2 * d + 1, 1);
                grid_->setColumnStretch(item, 1);
                cells_.push_back(std::move(c));
            }
    }

    void BandGridView::renderCells() {
        if (!diag_) return;
        const SimDiagnostics& d = *diag_;
        const Buffer<std::complex<double>>& bands = stage_->currentIndex() == 1 ? d.filtered : d.separated;
        const Index planeElems = d.ny * (d.nx / 2 + 1);
        const Index bandElems = d.nz * planeElems;
        // slider is centered kz; the storage is in FFT order
        const Index zc = slice_->value();
        const Index z = (zc - d.nz / 2 + d.nz) % d.nz;
        sliceLabel_->setText(tr("%1 / %2").arg(zc + 1).arg(d.nz));

        for (Cell& c : cells_) {
            int band = 0;
            BandSide side = BandSide::ReOnly;
            decodeItem(c.item, band, side);
            const int order = c.item == 0 ? 0 : (c.item + 1) / 2;
            const std::complex<double>* re = bands.data() + (static_cast<Index>(c.direction) * d.nbands + band) * bandElems;
            const std::complex<double>* im = order == 0 ? nullptr : re + bandElems;
            c.values.resize(static_cast<std::size_t>(d.ny * d.nx));
            c.gray.resize(c.values.size());
            bandPlaneMagnitude(re, im, d.nz, d.ny, d.nx, z, side, c.values.data());
            if (log_->isChecked()) logDisplayTransform(c.values);
            mapToGray8(c.values.data(), d.ny * d.nx, window_, c.gray.data());
            c.canvas->setImage(QImage(c.gray.data(), static_cast<int>(d.nx), static_cast<int>(d.ny),
                                      static_cast<int>(d.nx), QImage::Format_Grayscale8));
            c.canvas->setOverlays(spectrumOverlayItems(overlayFor(params_, d.nz, &fit_, order == 0), d.ny, d.nx,
                                                       params_.dx, params_.dy));
        }
    }

    void BandGridView::autoWindow() {
        if (!diag_ || cells_.empty()) return;
        // one window for all cells: percentiles of everything on screen
        std::vector<double> all;
        for (const Cell& c : cells_) all.insert(all.end(), c.values.begin(), c.values.end());
        window_ = percentileRange(all.data(), static_cast<Index>(all.size()), 0.001, 0.999);
        for (Cell& c : cells_) {
            mapToGray8(c.values.data(), static_cast<Index>(c.values.size()), window_, c.gray.data());
            c.canvas->update();
        }
    }

    // --- BandView --------------------------------------------------------

    BandView::BandView(QWidget* parent) : QWidget(parent) {
        direction_ = new QComboBox(this);
        direction_->setToolTip(tr("Pattern direction whose bands are shown"));
        band_ = new QComboBox(this);
        band_->setToolTip(tr("Band: order 0 is the widefield spectrum; ±o are the side bands re ± i·im of order o, "
                             "the object spectrum shifted by ∓o·k0"));
        stage_ = new QComboBox(this);
        stage_->addItem(tr("Separated (before filtering)"));
        stage_->addItem(tr("Wiener filtered (as assembled)"));
        stage_->setToolTip(kStageTip);
        info_ = new QLabel(this);
        info_->setWordWrap(true);
        view_ = new StackView(this);
        view_->setVolumeIsSpectrum(true);
        view_->setLogScale(true);

        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel(tr("Direction"), this));
        row->addWidget(direction_);
        row->addWidget(new QLabel(tr("Band"), this));
        row->addWidget(band_);
        row->addWidget(new QLabel(tr("Stage"), this));
        row->addWidget(stage_);
        row->addStretch(1);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addLayout(row);
        layout->addWidget(info_);
        layout->addWidget(view_, 1);

        for (QComboBox* c : {direction_, band_, stage_})
            connect(c, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] { rebuild(); });
    }

    void BandView::setResult(std::shared_ptr<const SimDiagnostics> diagnostics, SimFit fit, SIMParameters params) {
        if (!diagnostics || !diagnostics->captured) {
            clear();
            return;
        }
        diag_ = std::move(diagnostics);
        fit_ = std::move(fit);
        params_ = std::move(params);
        {
            const QSignalBlocker b1(direction_), b2(band_);
            const int dir = direction_->currentIndex(), band = band_->currentIndex();
            direction_->clear();
            for (int d = 0; d < diag_->ndirs; ++d) direction_->addItem(tr("%1").arg(d));
            band_->clear();
            const int norders = (diag_->nbands + 1) / 2;
            for (int item = 0; item < 2 * norders - 1; ++item) band_->addItem(bandItemName(item));
            direction_->setCurrentIndex(std::max(0, std::min(dir, direction_->count() - 1)));
            band_->setCurrentIndex(std::max(0, std::min(band, band_->count() - 1)));
        }
        view_->setPixelSize(params_.dx, params_.dy, params_.dz);
        rebuild();
    }

    void BandView::select(int direction, int bandItem, int stage) {
        const QSignalBlocker b1(direction_), b2(band_), b3(stage_);
        direction_->setCurrentIndex(std::max(0, std::min(direction, direction_->count() - 1)));
        band_->setCurrentIndex(std::max(0, std::min(bandItem, band_->count() - 1)));
        stage_->setCurrentIndex(std::max(0, std::min(stage, stage_->count() - 1)));
        rebuild();
    }

    void BandView::clear() {
        diag_.reset();
        view_->clear();
        info_->setText(tr("Nothing captured."));
    }

    void BandView::rebuild() {
        if (!diag_) return;
        const int dir = direction_->currentIndex();
        const int item = band_->currentIndex();
        int band = 0;
        BandSide side = BandSide::ReOnly;
        decodeItem(item, band, side);
        const int order = item == 0 ? 0 : (item + 1) / 2;
        const bool filtered = stage_->currentIndex() == 1;
        try {
            auto vol = std::make_shared<Buffer<double>>(
                bandMagnitudeVolume(*diag_, filtered ? diag_->filtered : diag_->separated, dir, band, side));
            view_->setOverlay(overlayFor(params_, diag_->nz, &fit_, order == 0));
            view_->setVolume(std::move(vol));
            info_->setText(tr("Direction %1, %2, %3: centered |spectrum| on the %4 x %5 x %6 data grid "
                              "(dk = %7 x %8 1/um). Order 0 sits at the object's origin; order o is the object "
                              "spectrum shifted by -o k0, so its center corresponds to o k0.")
                               .arg(dir).arg(band_->currentText()).arg(stage_->currentText())
                               .arg(diag_->nz).arg(diag_->ny).arg(diag_->nx)
                               .arg(diag_->dkx, 0, 'g', 4).arg(diag_->dky, 0, 'g', 4));
        } catch (const std::exception& e) {
            view_->clear();
            info_->setText(tr("Cannot display band: %1").arg(QString::fromUtf8(e.what())));
        }
    }

    // --- OtfView ---------------------------------------------------------

    OtfView::OtfView(QWidget* parent) : QWidget(parent) {
        order_ = new QComboBox(this);
        order_->setToolTip(tr("Illumination order whose OTF is shown. Order 0 is the widefield OTF; in 3D order 1 "
                              "is shifted along kz by the axial frequency of the first illumination order; order 2 "
                              "equals order 0."));
        info_ = new QLabel(tr("No OTF."), this);
        info_->setWordWrap(true);
        view_ = new StackView(this);
        view_->setVolumeIsSpectrum(true);
        view_->setLogScale(true);

        auto* row = new QHBoxLayout;
        row->addWidget(new QLabel(tr("Order"), this));
        row->addWidget(order_);
        row->addStretch(1);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addLayout(row);
        layout->addWidget(info_);
        layout->addWidget(view_, 1);
        connect(order_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] { rebuild(); });
    }

    void OtfView::setOtf(std::shared_ptr<const OTFRadiallyAveraged> otf, SIMParameters params, Index nx, Index ny,
                         Index nz, const QString& source) {
        otf_ = std::move(otf);
        params_ = std::move(params);
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
        if (!otf_) {
            clear();
            return;
        }
        {
            const QSignalBlocker block(order_);
            const int keep = order_->currentIndex();
            order_->clear();
            for (Eigen::Index o = 0; o < otf_->data().dimension(0); ++o) order_->addItem(tr("%1").arg(o));
            order_->setCurrentIndex(std::max(0, std::min(keep, order_->count() - 1)));
        }
        const auto& d = otf_->data();
        info_->setText(tr("%1: %2 order(s), %3 radial x %4 axial samples (dkr = %5, dkz = %6 1/um), rendered on "
                          "the %7 x %8 x %9 grid as the reconstruction interpolates it. The circle marks 2NA/λ; "
                          "use Ortho to see the kr-kz section (missing cone).")
                           .arg(source).arg(d.dimension(0)).arg(d.dimension(1)).arg(d.dimension(2))
                           .arg(otf_->dkrotf(), 0, 'g', 4).arg(otf_->dkzotf(), 0, 'g', 4)
                           .arg(nx).arg(ny).arg(nz));
        view_->setPixelSize(params_.dx, params_.dy, params_.dz);
        rebuild();
    }

    void OtfView::clear() {
        otf_.reset();
        view_->clear();
        info_->setText(tr("No OTF."));
    }

    void OtfView::rebuild() {
        if (!otf_) return;
        try {
            auto vol = std::make_shared<Buffer<double>>(
                otfDisplayVolume(*otf_, order_->currentIndex(), params_, nx_, ny_, nz_));
            view_->setOverlay(overlayFor(params_, nz_, nullptr, false));
            view_->setVolume(std::move(vol));
        } catch (const std::exception& e) {
            view_->clear();
            info_->setText(tr("Cannot display OTF: %1").arg(QString::fromUtf8(e.what())));
        }
    }

} // namespace sirius::app
