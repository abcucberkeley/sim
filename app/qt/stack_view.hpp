#ifndef SIRIUS_APP_STACK_VIEW_HPP
#define SIRIUS_APP_STACK_VIEW_HPP

// Slice viewer for a (depth, rows, cols) host volume in the spirit of
// ImageJ: zoom / pan / fit, a rectangular selection with "Crop", min/max
// contrast sliders with Auto (percentiles) and Reset, orthogonal XZ / YZ
// views on a click-positioned crosshair, a log intensity transform and a
// spectrum mode showing the centered |FFT| of each displayed plane with
// frequency-space overlays (OTF support, predicted and fitted pattern
// vectors). Planes are transformed and mapped to 8-bit gray on demand into
// persistent buffers, so scrubbing through a stack does no allocation.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <QRect>
#include <QWidget>

#include <sirius/buffer.hpp>

#include "core/display_mapping.hpp"
#include "core/volume_ops.hpp"
#include "qt/image_canvas.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QToolButton;

namespace sirius::app {

    class ImageCanvas;

    // What to draw over a centered spectrum of the XY plane.
    struct SpectrumOverlay {
        double supportRadius = 0.0;                            // 2 NA / lambda [1/um]; 0 = none
        std::vector<std::array<double, 2>> predictedK0;        // per direction [1/um]
        std::vector<std::array<double, 2>> fittedK0;           // per direction, empty before a run
        std::vector<std::vector<double>> ampMagnitude;         // (ndirs, norders) |modamp|
        int norders = 1;
        bool showOrders = true;                                // k0 markers for orders 1..norders-1
    };

    // Overlay items for a centered (rows, cols) spectrum with pixel sizes
    // dx, dy: the OTF support circle, the pattern vectors expected from the
    // parameters (crosses) and the fitted ones (circles, with |amp| labels).
    std::vector<CanvasOverlay> spectrumOverlayItems(const SpectrumOverlay& o, Index rows, Index cols,
                                                    double dx, double dy);

    // Display transform for log mode: log10 with a floor six decades below
    // the peak, so zeros and the noise floor stay finite.
    void logDisplayTransform(std::vector<double>& values);

    class StackView : public QWidget {
        Q_OBJECT
    public:
        explicit StackView(QWidget* parent = nullptr);

        // Shared ownership: the same volume may be shown here and kept by the
        // session / result at the same time.
        void setVolume(std::shared_ptr<const Buffer<double>> volume);
        void clear();
        std::shared_ptr<const Buffer<double>> volume() const { return volume_; }
        int currentSlice() const;

        // Voxel size [um]: value readout, orthoview aspect and frequency steps.
        void setPixelSize(double dx, double dy, double dz);
        std::array<double, 3> pixelSize() const { return {dx_, dy_, dz_}; }
        bool volumeIsSpectrum() const { return volumeIsSpectrum_; }
        std::optional<SpectrumOverlay> overlay() const { return overlay_; }
        // The volume already is a centered spectrum (bands, OTF): the spectrum
        // toggle is hidden and the readout reports frequencies.
        void setVolumeIsSpectrum(bool isSpectrum);
        void setOverlay(std::optional<SpectrumOverlay> overlay);
        void setLogScale(bool on);
        bool logScale() const;
        void setSpectrumMode(bool on);
        bool spectrumMode() const;

    signals:
        // The user asked to crop to `selection` (XY pixels, every slice).
        void cropRequested(QRect selection);

    private:
        struct Plane {
            std::vector<double> values;   // transformed (spectrum / log) plane
            std::vector<std::uint8_t> gray;
            Index rows = 0, cols = 0;
        };

        void buildUi();
        void renderAll();
        void renderXY();
        void renderOrtho();
        void preparePlane(const double* src, Index rows, Index cols, Plane& plane);
        void showPlane(ImageCanvas* canvas, Plane& plane);
        void setWindow(DisplayRange r);
        void syncWindowControls();
        void autoWindow(bool percentile);
        void updateOverlays();
        void updateCrosshairs();
        void updateStatus(const QString& text = {});
        void hoverXY(int x, int y);
        void hoverXZ(int x, int y);
        void hoverYZ(int x, int y);
        bool frequencyReadout() const;
        std::array<double, 2> frequencyOf(Index cols, Index rows, int x, int y) const;
        void setCrosshair(Index x, Index y, Index z);

        std::shared_ptr<const Buffer<double>> volume_;
        PlaneSpectrum spectrum_;
        Plane xy_, xz_, yz_;
        DisplayRange window_;
        double spanLo_ = 0.0, spanHi_ = 1.0;   // slider extent
        bool updatingControls_ = false;
        double dx_ = 1.0, dy_ = 1.0, dz_ = 1.0;
        bool volumeIsSpectrum_ = false;
        std::optional<SpectrumOverlay> overlay_;
        Index crossX_ = 0, crossY_ = 0;

        ImageCanvas* xyCanvas_ = nullptr;
        ImageCanvas* xzCanvas_ = nullptr;
        ImageCanvas* yzCanvas_ = nullptr;
        QSlider* slice_ = nullptr;
        QLabel* sliceLabel_ = nullptr;
        QToolButton* selectTool_ = nullptr;
        QPushButton* crop_ = nullptr;
        QLabel* zoomLabel_ = nullptr;
        QCheckBox* ortho_ = nullptr;
        QCheckBox* physicalZ_ = nullptr;
        QCheckBox* spectrumBox_ = nullptr;
        QCheckBox* logBox_ = nullptr;
        QCheckBox* overlayBox_ = nullptr;
        QLabel* legend_ = nullptr;
        std::vector<QWidget*> navControls_;   // disabled while the orthoviews lock navigation
        QDoubleSpinBox* minSpin_ = nullptr;
        QDoubleSpinBox* maxSpin_ = nullptr;
        QSlider* minSlider_ = nullptr;
        QSlider* maxSlider_ = nullptr;
        QLabel* status_ = nullptr;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_STACK_VIEW_HPP
