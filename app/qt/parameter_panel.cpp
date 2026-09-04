#include "qt/parameter_panel.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

namespace sirius::app {

    namespace {

        // Small builders keep the constructor a flat list of fields. Every
        // widget forwards its edit signal to ParameterPanel::changed, and the
        // tooltip goes on the field and its label.

        void addRow(QFormLayout* form, const QString& label, QWidget* field, const QString& tip) {
            form->addRow(label, field);
            field->setToolTip(tip);
            if (QWidget* l = form->labelForField(field)) l->setToolTip(tip);
        }

        QDoubleSpinBox* doubleBox(ParameterPanel* owner, QFormLayout* form, const QString& label,
                                  double lo, double hi, int decimals, double step, const QString& tip) {
            auto* box = new QDoubleSpinBox(owner);
            box->setRange(lo, hi);
            box->setDecimals(decimals);
            box->setSingleStep(step);
            box->setKeyboardTracking(false);
            addRow(form, label, box, tip);
            QObject::connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                             owner, &ParameterPanel::changed);
            return box;
        }

        QSpinBox* intBox(ParameterPanel* owner, QFormLayout* form, const QString& label,
                         int lo, int hi, const QString& tip) {
            auto* box = new QSpinBox(owner);
            box->setRange(lo, hi);
            box->setKeyboardTracking(false);
            addRow(form, label, box, tip);
            QObject::connect(box, QOverload<int>::of(&QSpinBox::valueChanged),
                             owner, &ParameterPanel::changed);
            return box;
        }

        QCheckBox* checkBox(ParameterPanel* owner, QFormLayout* form, const QString& label, const QString& tip) {
            auto* box = new QCheckBox(owner);
            addRow(form, label, box, tip);
            QObject::connect(box, &QCheckBox::toggled, owner, &ParameterPanel::changed);
            return box;
        }

        QComboBox* apodizationBox(ParameterPanel* owner, QFormLayout* form, const QString& label,
                                  const QString& tip) {
            auto* box = new QComboBox(owner);
            // item index == enum value, see ApodizationType
            box->addItem(QObject::tr("None"));
            box->addItem(QObject::tr("Cosine"));
            box->addItem(QObject::tr("Triangle"));
            addRow(form, label, box, tip);
            QObject::connect(box, QOverload<int>::of(&QComboBox::currentIndexChanged),
                             owner, &ParameterPanel::changed);
            return box;
        }

        // Group box with a "?" button in its header that opens a help section.
        QFormLayout* group(QVBoxLayout* parent, ParameterPanel* owner, const QString& title,
                           const QString& anchor, const QString& tip) {
            auto* box = new QGroupBox(title, owner);
            box->setToolTip(tip);
            auto* layout = new QVBoxLayout(box);
            auto* header = new QHBoxLayout;
            auto* hint = new QLabel(tip, box);
            hint->setWordWrap(true);
            hint->setStyleSheet(QStringLiteral("color: #666;"));
            auto* help = new QToolButton(box);
            help->setText(QStringLiteral("?"));
            help->setAutoRaise(true);
            help->setToolTip(QObject::tr("Show the help section about %1").arg(title.toLower()));
            QObject::connect(help, &QToolButton::clicked, owner, [owner, anchor] { emit owner->helpRequested(anchor); });
            header->addWidget(hint, 1);
            header->addWidget(help, 0, Qt::AlignTop);
            layout->addLayout(header);
            auto* form = new QFormLayout;
            form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
            layout->addLayout(form);
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

        QFormLayout* optics = group(layout, this, tr("Illumination and optics"), QStringLiteral("optics"),
                                    tr("How the raw stack was acquired and what the objective can resolve."));
        ndirs_ = intBox(this, optics, tr("Directions"), 1, 16,
                        tr("Number of pattern orientations in the stack (usually 3, spaced by 180°/N). Together "
                           "with Phases it fixes how the sections are grouped: sections = directions × phases × z."));
        nphases_ = intBox(this, optics, tr("Phases"), 1, 32,
                          tr("Pattern phase steps per direction (5 for 3D / three-beam SIM, 3 for 2D). Band "
                             "separation needs at least 2 × orders − 1 phases."));
        norders_ = intBox(this, optics, tr("Orders"), 1, 16,
                          tr("Harmonic orders of the illumination to separate: 3 (orders 0, ±1, ±2) for three-beam "
                             "3D SIM, 2 for two-beam 2D SIM. 0 in a file means phases / 2 + 1."));
        k0StartAngle_ = doubleBox(this, optics, tr("k0 start angle (rad)"), -10.0, 10.0, 6, 0.01,
                                  tr("Orientation of direction 0's pattern vector k0 in the image plane, in radians "
                                     "(x axis = 0, counter-clockwise). Direction d starts at this angle + d·π/N. Only "
                                     "a starting guess: the fit refines it. Ignored when the file lists explicit "
                                     "angles (k0angles)."));
        linespacing_ = doubleBox(this, optics, tr("Line spacing (um)"), 0.0, 100.0, 5, 0.001,
                                 tr("Period of the finest illumination pattern in the sample, in µm. Its inverse is "
                                    "the starting |k0| (for 3D data the order-1 vector is half of it). Read it from "
                                    "the acquisition or from the yellow markers on the raw spectrum."));
        na_ = doubleBox(this, optics, tr("NA"), 0.0, 3.0, 3, 0.01,
                        tr("Numerical aperture of the objective. Sets the OTF support 2NA/λ (white circle on "
                           "spectra), the axial cutoffs and, without an OTF file, the ideal OTF."));
        nimm_ = doubleBox(this, optics, tr("Immersion index"), 1.0, 3.0, 4, 0.01,
                          tr("Refractive index of the immersion medium (1.515 oil, 1.33 water, 1.0 air). Used for "
                             "the axial extent of the OTF (missing cone) and the ideal OTF."));
        wavelength_ = doubleBox(this, optics, tr("Emission wavelength (nm)"), 100.0, 2000.0, 1, 1.0,
                                tr("Emission wavelength λ in nm. With NA it fixes the resolution limit 2NA/λ; the "
                                   "excitation wavelength is taken as 0.88 λ for the axial cutoffs."));

        QFormLayout* sampling = group(layout, this, tr("Sampling"), QStringLiteral("sampling"),
                                      tr("Voxel size of the raw stack and the size of the output grid."));
        dx_ = doubleBox(this, sampling, tr("dx (um)"), 0.0, 100.0, 5, 0.001,
                        tr("Pixel size along x in the sample plane, in µm (camera pixel / magnification). "
                           "Frequency step of a spectrum is 1/(N·dx)."));
        dy_ = doubleBox(this, sampling, tr("dy (um)"), 0.0, 100.0, 5, 0.001,
                        tr("Pixel size along y in µm; normally equal to dx."));
        dz_ = doubleBox(this, sampling, tr("dz (um)"), 0.0, 100.0, 5, 0.01,
                        tr("Distance between z planes of the raw stack in µm. Determines the axial frequency "
                           "step and the Physical z aspect of the orthoviews."));
        dzPsf_ = doubleBox(this, sampling, tr("dz of PSF/OTF (um)"), 0.0, 100.0, 5, 0.01,
                           tr("z step of the bead stack the OTF was measured with (or of the simulated PSF behind "
                              "the ideal OTF). It fixes the OTF's axial frequency step dkz = 1/(dz_psf·nz_otf)."));
        zoomfact_ = doubleBox(this, sampling, tr("Lateral zoom"), 1.0, 8.0, 2, 0.5,
                              tr("Output pixels per input pixel in x and y. 2 is needed to hold the extended "
                                 "support (up to 2NA/λ + 2|k0|) without aliasing; the output pixel is dx / zoom."));
        zZoom_ = intBox(this, sampling, tr("Axial zoom"), 1, 8,
                        tr("Output planes per input plane. 1 keeps dz; 2 interpolates the axial resolution gain."));

        QFormLayout* filtering = group(layout, this, tr("Filtering"), QStringLiteral("filtering"),
                                       tr("Preprocessing of the raw frames and the generalized Wiener filter that "
                                          "combines the bands."));
        wiener_ = doubleBox(this, filtering, tr("Wiener constant"), 0.0, 10.0, 5, 0.001,
                            tr("Noise regularization w of the Wiener filter, relative to the normalized data. "
                               "Larger values suppress noise and ringing but soften the result; smaller ones "
                               "sharpen and amplify noise. 0.001 to 0.01 is typical."));
        otfcutoff_ = doubleBox(this, filtering, tr("OTF cutoff"), 0.0, 1.0, 5, 0.001,
                               tr("Voxels where |OTF| is below this fraction of its peak are left out of the "
                                  "overlap regions used to fit the pattern vectors and modulation amplitudes "
                                  "(order 0 is allowed 5× more in 3D). Raise it when the fit is thrown off by noise "
                                  "at the edge of the support."));
        background_ = doubleBox(this, filtering, tr("Background"), -1e6, 1e6, 2, 1.0,
                                tr("Camera offset (counts) subtracted from every pixel before anything else. A "
                                   "wrong background biases the modulation amplitudes and the bleach correction."));
        apodizeInput_ = apodizationBox(this, filtering, tr("Input apodization"),
                                       tr("Softens the raw frames' edges so the FFT sees a near-periodic image: "
                                          "Triangle blends opposite edges over a border of the given width, Cosine "
                                          "multiplies by a sine window, None does nothing."));
        napodize_ = intBox(this, filtering, tr("Border width (px)"), 0, 1000,
                           tr("Width in pixels of the edge blend used by the Triangle input apodization."));
        apodizeOutput_ = apodizationBox(this, filtering, tr("Output apodization"),
                                        tr("Tapers the assembled spectrum towards the edge of the extended support "
                                           "so the result does not ring: Triangle (linear) or Cosine taper, None."));
        suppressionRadius_ = intBox(this, filtering, tr("Suppression radius"), 0, 1000,
                                    tr("Radius, in frequency pixels, of the notch applied around each band's "
                                       "center (the residual illumination peak) when singularities are suppressed."));
        suppressSingularities_ = checkBox(this, filtering, tr("Suppress singularities"),
                                          tr("Damp the residual pattern peaks at the band centers of orders ≥ 1 "
                                             "within the suppression radius, avoiding a bright spot / stripes in "
                                             "the result."));
        dampenOrder0_ = checkBox(this, filtering, tr("Dampen order 0"),
                                 tr("Reduce the weight of the order-0 (widefield) band near the origin, which "
                                    "lowers out-of-focus haze at the cost of some low-frequency contrast."));
        doRescale_ = checkBox(this, filtering, tr("Bleach correction"),
                              tr("Scale every raw frame so its total intensity matches direction 0 / phase 0 of the "
                                 "same z plane, compensating photobleaching and illumination drift across phases."));
        equalizez_ = checkBox(this, filtering, tr("Equalize across z"),
                              tr("Bleach correction relative to direction 0 / phase 0 / z 0 for every plane "
                                 "(also equalizes the axial intensity profile). Off: per z plane."));
        noKz0_ = checkBox(this, filtering, tr("Skip kz = 0 plane"),
                          tr("3D data: exclude the kz = 0 plane from the overlap regions used by the fit and give "
                             "it zero weight in the order-0 filter, where the missing cone makes it unreliable."));
        filterOverlaps_ = checkBox(this, filtering, tr("Filter overlaps"),
                                   tr("Apply the Wiener weights inside the regions where bands overlap (the usual "
                                      "generalized Wiener filter). Off: bands are summed without cross-weighting."));

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
        p.dz_psf = dzPsf_->value();
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
        dzPsf_->setValue(p.dz_psf);
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
