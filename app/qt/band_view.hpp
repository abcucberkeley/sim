#ifndef SIRIUS_APP_BAND_VIEW_HPP
#define SIRIUS_APP_BAND_VIEW_HPP

// Viewers over data that already lives in frequency space:
//   BandView  the separated / Wiener-filtered band spectra a reconstruction
//             captured (SimDiagnostics), selectable by direction, order and
//             stage, with the OTF support and pattern vectors overlaid;
//   OtfView   one order of the OTF (loaded or ideal) resampled onto the data
//             grid, as the reconstruction interpolates it.
// Both wrap a StackView in "volume is a spectrum" mode with log display.

#include <memory>

#include <QWidget>

#include <sirius/otf.hpp>
#include <sirius/sim_parameters.hpp>
#include <sirius/sim_reconstruction.hpp>

class QComboBox;
class QLabel;

namespace sirius::app {

    class StackView;

    class BandView : public QWidget {
        Q_OBJECT
    public:
        explicit BandView(QWidget* parent = nullptr);

        // Takes the captured diagnostics (shared with the result) and the fit
        // for the overlays. Nothing captured -> clear().
        void setResult(std::shared_ptr<const SimDiagnostics> diagnostics, SimFit fit, SIMParameters params);
        void clear();

    private:
        void rebuild();

        std::shared_ptr<const SimDiagnostics> diag_;
        SimFit fit_;
        SIMParameters params_;
        QComboBox* direction_ = nullptr;
        QComboBox* band_ = nullptr;
        QComboBox* stage_ = nullptr;
        QLabel* info_ = nullptr;
        StackView* view_ = nullptr;
    };

    class OtfView : public QWidget {
        Q_OBJECT
    public:
        explicit OtfView(QWidget* parent = nullptr);

        // Grid (nx, ny, nz) the OTF is rendered on: the loaded stack's, or a
        // default one before a stack is loaded.
        void setOtf(std::shared_ptr<const OTFRadiallyAveraged> otf, SIMParameters params, Index nx, Index ny,
                    Index nz, const QString& source);
        void clear();

    private:
        void rebuild();

        std::shared_ptr<const OTFRadiallyAveraged> otf_;
        SIMParameters params_;
        Index nx_ = 0, ny_ = 0, nz_ = 0;
        QComboBox* order_ = nullptr;
        QLabel* info_ = nullptr;
        StackView* view_ = nullptr;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_BAND_VIEW_HPP
