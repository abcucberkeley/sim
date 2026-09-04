#ifndef SIRIUS_APP_BAND_VIEW_HPP
#define SIRIUS_APP_BAND_VIEW_HPP

// Viewers over data that already lives in frequency space:
//   BandGridView  every separated / Wiener-filtered band a reconstruction
//                 captured (SimDiagnostics) at once: a direction x band grid
//                 of small canvases sharing one kz slider, window and stage;
//                 double-clicking a cell asks for a full viewer of it;
//   BandView      one band in a full StackView (zoom, ortho, crop, ...),
//                 selectable by direction, order and stage;
//   OtfView       one order of the OTF (loaded or ideal) resampled onto the
//                 data grid, as the reconstruction interpolates it.

#include <cstdint>
#include <memory>
#include <vector>

#include <QWidget>

#include <sirius/otf.hpp>
#include <sirius/sim_parameters.hpp>
#include <sirius/sim_reconstruction.hpp>

#include "core/display_mapping.hpp"

class QCheckBox;
class QComboBox;
class QGridLayout;
class QLabel;
class QSlider;

namespace sirius::app {

    class ImageCanvas;
    class StackView;

    class BandGridView : public QWidget {
        Q_OBJECT
    public:
        explicit BandGridView(QWidget* parent = nullptr);

        // Takes the captured diagnostics (shared with the result) and the fit
        // for the overlays. Nothing captured -> clear().
        void setResult(std::shared_ptr<const SimDiagnostics> diagnostics, SimFit fit, SIMParameters params);
        void clear();

    signals:
        // A cell was double-clicked: band item as in BandView::select.
        void openRequested(int direction, int bandItem, int stage);

    private:
        struct Cell {
            int direction = 0;
            int item = 0;
            QLabel* title = nullptr;
            ImageCanvas* canvas = nullptr;
            std::vector<double> values;
            std::vector<std::uint8_t> gray;
        };

        void rebuildGrid();
        void renderCells();
        void autoWindow();

        std::shared_ptr<const SimDiagnostics> diag_;
        SimFit fit_;
        SIMParameters params_;
        std::vector<Cell> cells_;
        DisplayRange window_;

        QComboBox* stage_ = nullptr;
        QSlider* slice_ = nullptr;
        QLabel* sliceLabel_ = nullptr;
        QCheckBox* log_ = nullptr;
        QLabel* info_ = nullptr;
        QWidget* gridHost_ = nullptr;
        QGridLayout* grid_ = nullptr;
    };

    class BandView : public QWidget {
        Q_OBJECT
    public:
        explicit BandView(QWidget* parent = nullptr);

        void setResult(std::shared_ptr<const SimDiagnostics> diagnostics, SimFit fit, SIMParameters params);
        // Band item: 0 = order 0, 2o - 1 = order +o, 2o = order -o. Stage: 0
        // separated, 1 filtered.
        void select(int direction, int bandItem, int stage);
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
