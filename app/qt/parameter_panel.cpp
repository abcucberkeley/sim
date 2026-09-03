#include "qt/parameter_panel.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

namespace sirius::app {

    namespace {

        // Small builders keep the constructor a flat list of fields. Every
        // widget forwards its edit signal to ParameterPanel::changed.

        QDoubleSpinBox* doubleBox(ParameterPanel* owner, QFormLayout* form, const QString& label,
                                  double lo, double hi, int decimals, double step) {
            auto* box = new QDoubleSpinBox(owner);
            box->setRange(lo, hi);
            box->setDecimals(decimals);
            box->setSingleStep(step);
            box->setKeyboardTracking(false);
            form->addRow(label, box);
            QObject::connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                             owner, &ParameterPanel::changed);
            return box;
        }

        QSpinBox* intBox(ParameterPanel* owner, QFormLayout* form, const QString& label,
                         int lo, int hi) {
            auto* box = new QSpinBox(owner);
            box->setRange(lo, hi);
            box->setKeyboardTracking(false);
            form->addRow(label, box);
            QObject::connect(box, QOverload<int>::of(&QSpinBox::valueChanged),
                             owner, &ParameterPanel::changed);
            return box;
        }

        QCheckBox* checkBox(ParameterPanel* owner, QFormLayout* form, const QString& label) {
            auto* box = new QCheckBox(owner);
            form->addRow(label, box);
            QObject::connect(box, &QCheckBox::toggled, owner, &ParameterPanel::changed);
            return box;
        }

        QComboBox* apodizationBox(ParameterPanel* owner, QFormLayout* form, const QString& label) {
            auto* box = new QComboBox(owner);
            // item index == enum value, see ApodizationType
            box->addItem(QObject::tr("None"));
            box->addItem(QObject::tr("Cosine"));
            box->addItem(QObject::tr("Triangle"));
            form->addRow(label, box);
            QObject::connect(box, QOverload<int>::of(&QComboBox::currentIndexChanged),
                             owner, &ParameterPanel::changed);
            return box;
        }

        QFormLayout* group(QVBoxLayout* parent, QWidget* owner, const QString& title) {
            auto* box = new QGroupBox(title, owner);
            auto* form = new QFormLayout(box);
            form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
            parent->addWidget(box);
            return form;
        }

        ApodizationType toApodization(int index) {
            return static_cast<ApodizationType>(qBound(0, index, 2));
        }

    } // namespace

    ParameterPanel::ParameterPanel(QWidget* parent) : QWidget(parent) {
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);

        QFormLayout* optics = group(layout, this, tr("Illumination and optics"));
        ndirs_ = intBox(this, optics, tr("Directions"), 1, 16);
        nphases_ = intBox(this, optics, tr("Phases"), 1, 32);
        norders_ = intBox(this, optics, tr("Orders"), 1, 16);
        k0StartAngle_ = doubleBox(this, optics, tr("k0 start angle (rad)"), -10.0, 10.0, 6, 0.01);
        linespacing_ = doubleBox(this, optics, tr("Line spacing (um)"), 0.0, 100.0, 5, 0.001);
        na_ = doubleBox(this, optics, tr("NA"), 0.0, 3.0, 3, 0.01);
        nimm_ = doubleBox(this, optics, tr("Immersion index"), 1.0, 3.0, 4, 0.01);
        wavelength_ = doubleBox(this, optics, tr("Emission wavelength (nm)"), 100.0, 2000.0, 1, 1.0);

        QFormLayout* sampling = group(layout, this, tr("Sampling"));
        dx_ = doubleBox(this, sampling, tr("dx (um)"), 0.0, 100.0, 5, 0.001);
        dy_ = doubleBox(this, sampling, tr("dy (um)"), 0.0, 100.0, 5, 0.001);
        dz_ = doubleBox(this, sampling, tr("dz (um)"), 0.0, 100.0, 5, 0.01);
        zoomfact_ = doubleBox(this, sampling, tr("Lateral zoom"), 1.0, 8.0, 2, 0.5);
        zZoom_ = intBox(this, sampling, tr("Axial zoom"), 1, 8);

        QFormLayout* filtering = group(layout, this, tr("Filtering"));
        wiener_ = doubleBox(this, filtering, tr("Wiener constant"), 0.0, 10.0, 5, 0.001);
        otfcutoff_ = doubleBox(this, filtering, tr("OTF cutoff"), 0.0, 1.0, 5, 0.001);
        background_ = doubleBox(this, filtering, tr("Background"), -1e6, 1e6, 2, 1.0);
        apodizeInput_ = apodizationBox(this, filtering, tr("Input apodization"));
        napodize_ = intBox(this, filtering, tr("Border width (px)"), 0, 1000);
        apodizeOutput_ = apodizationBox(this, filtering, tr("Output apodization"));
        suppressionRadius_ = intBox(this, filtering, tr("Suppression radius"), 0, 1000);
        suppressSingularities_ = checkBox(this, filtering, tr("Suppress singularities"));
        dampenOrder0_ = checkBox(this, filtering, tr("Dampen order 0"));
        doRescale_ = checkBox(this, filtering, tr("Bleach correction"));
        equalizez_ = checkBox(this, filtering, tr("Equalize across z"));
        noKz0_ = checkBox(this, filtering, tr("Skip kz = 0 plane"));
        filterOverlaps_ = checkBox(this, filtering, tr("Filter overlaps"));

        layout->addStretch(1);
        setParameters(SIMParameters{});
    }

    SIMParameters ParameterPanel::parameters() const {
        SIMParameters p = base_;
        p.k0_start_angle = k0StartAngle_->value();
        p.linespacing_um = linespacing_->value();
        p.ndirs = ndirs_->value();
        p.nphases = nphases_->value();
        p.norders = norders_->value();
        p.na = na_->value();
        p.nimm = nimm_->value();
        p.wavelength_nm = wavelength_->value();
        p.dx = dx_->value();
        p.dy = dy_->value();
        p.dz = dz_->value();
        p.zoomfact = zoomfact_->value();
        p.z_zoom = zZoom_->value();
        p.wiener = wiener_->value();
        p.otfcutoff = otfcutoff_->value();
        p.background = background_->value();
        p.apodize_input = toApodization(apodizeInput_->currentIndex());
        p.napodize = napodize_->value();
        p.apodize_output = toApodization(apodizeOutput_->currentIndex());
        p.suppression_radius = suppressionRadius_->value();
        p.suppress_singularities = suppressSingularities_->isChecked();
        p.dampen_order0 = dampenOrder0_->isChecked();
        p.do_rescale = doRescale_->isChecked();
        p.equalizez = equalizez_->isChecked();
        p.no_kz0 = noKz0_->isChecked();
        p.filter_overlaps = filterOverlaps_->isChecked();
        return p;
    }

    void ParameterPanel::setParameters(const SIMParameters& p) {
        base_ = p;
        // block every child so a bulk update is one logical change, not ~25
        const QList<QWidget*> children = findChildren<QWidget*>();
        std::vector<QSignalBlocker> blockers(children.begin(), children.end());

        k0StartAngle_->setValue(p.k0_start_angle);
        linespacing_->setValue(p.linespacing_um);
        ndirs_->setValue(p.ndirs);
        nphases_->setValue(p.nphases);
        norders_->setValue(p.norders);
        na_->setValue(p.na);
        nimm_->setValue(p.nimm);
        wavelength_->setValue(p.wavelength_nm);
        dx_->setValue(p.dx);
        dy_->setValue(p.dy);
        dz_->setValue(p.dz);
        zoomfact_->setValue(p.zoomfact);
        zZoom_->setValue(p.z_zoom);
        wiener_->setValue(p.wiener);
        otfcutoff_->setValue(p.otfcutoff);
        background_->setValue(p.background);
        apodizeInput_->setCurrentIndex(static_cast<int>(p.apodize_input));
        napodize_->setValue(p.napodize);
        apodizeOutput_->setCurrentIndex(static_cast<int>(p.apodize_output));
        suppressionRadius_->setValue(p.suppression_radius);
        suppressSingularities_->setChecked(p.suppress_singularities);
        dampenOrder0_->setChecked(p.dampen_order0);
        doRescale_->setChecked(p.do_rescale);
        equalizez_->setChecked(p.equalizez);
        noKz0_->setChecked(p.no_kz0);
        filterOverlaps_->setChecked(p.filter_overlaps);
    }

} // namespace sirius::app
