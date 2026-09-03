#ifndef SIRIUS_APP_PARAMETER_PANEL_HPP
#define SIRIUS_APP_PARAMETER_PANEL_HPP

// Form editor for SIMParameters. The widgets are the single source of the
// values while editing: parameters() reads them back into a struct and
// setParameters() populates them (without emitting changed()). Fields that
// have no meaningful GUI (k0_angles, dz_psf, explodefact, fast_si) are kept
// from the last setParameters() call so a loaded file round-trips intact.

#include <QWidget>

#include <sirius/sim_parameters.hpp>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QSpinBox;

namespace sirius::app {

    class ParameterPanel : public QWidget {
        Q_OBJECT
    public:
        explicit ParameterPanel(QWidget* parent = nullptr);

        SIMParameters parameters() const;
        void setParameters(const SIMParameters& p);

    signals:
        void changed();

    private:
        SIMParameters base_;   // carries the fields without widgets

        QDoubleSpinBox* k0StartAngle_ = nullptr;
        QDoubleSpinBox* linespacing_ = nullptr;
        QSpinBox* ndirs_ = nullptr;
        QSpinBox* nphases_ = nullptr;
        QSpinBox* norders_ = nullptr;
        QDoubleSpinBox* na_ = nullptr;
        QDoubleSpinBox* nimm_ = nullptr;
        QDoubleSpinBox* wavelength_ = nullptr;
        QDoubleSpinBox* dx_ = nullptr;
        QDoubleSpinBox* dy_ = nullptr;
        QDoubleSpinBox* dz_ = nullptr;
        QDoubleSpinBox* zoomfact_ = nullptr;
        QSpinBox* zZoom_ = nullptr;
        QDoubleSpinBox* wiener_ = nullptr;
        QDoubleSpinBox* otfcutoff_ = nullptr;
        QDoubleSpinBox* background_ = nullptr;
        QComboBox* apodizeInput_ = nullptr;
        QSpinBox* napodize_ = nullptr;
        QComboBox* apodizeOutput_ = nullptr;
        QSpinBox* suppressionRadius_ = nullptr;
        QCheckBox* suppressSingularities_ = nullptr;
        QCheckBox* dampenOrder0_ = nullptr;
        QCheckBox* doRescale_ = nullptr;
        QCheckBox* equalizez_ = nullptr;
        QCheckBox* noKz0_ = nullptr;
        QCheckBox* filterOverlaps_ = nullptr;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_PARAMETER_PANEL_HPP
