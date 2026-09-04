#include "qt/band_view.hpp"
#include "qt/stack_view.hpp"

#include <cmath>
#include <exception>

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSignalBlocker>
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
    } // namespace

    // --- BandView --------------------------------------------------------

    BandView::BandView(QWidget* parent) : QWidget(parent) {
        direction_ = new QComboBox(this);
        band_ = new QComboBox(this);
        stage_ = new QComboBox(this);
        stage_->addItem(tr("Separated (before filtering)"));
        stage_->addItem(tr("Wiener filtered (as assembled)"));
        info_ = new QLabel(tr("Enable \"Capture intermediate spectra\" and reconstruct to see the bands."), this);
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
            band_->addItem(tr("order 0"));
            const int norders = (diag_->nbands + 1) / 2;
            for (int o = 1; o < norders; ++o) {
                band_->addItem(tr("order +%1").arg(o));
                band_->addItem(tr("order -%1").arg(o));
            }
            direction_->setCurrentIndex(std::max(0, std::min(dir, direction_->count() - 1)));
            band_->setCurrentIndex(std::max(0, std::min(band, band_->count() - 1)));
        }
        view_->setPixelSize(params_.dx, params_.dy, params_.dz);
        rebuild();
    }

    void BandView::clear() {
        diag_.reset();
        view_->clear();
        info_->setText(tr("Enable \"Capture intermediate spectra\" and reconstruct to see the bands."));
    }

    void BandView::rebuild() {
        if (!diag_) return;
        const int dir = direction_->currentIndex();
        const int item = band_->currentIndex();
        const int order = item == 0 ? 0 : (item + 1) / 2;
        const BandSide side = item == 0 ? BandSide::ReOnly : (item % 2 == 1 ? BandSide::Plus : BandSide::Minus);
        const int band = order == 0 ? 0 : 2 * order - 1;
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
                          "the %7 x %8 x %9 grid as the reconstruction interpolates it. The circle marks 2NA/λ.")
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
