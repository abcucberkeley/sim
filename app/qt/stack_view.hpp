#ifndef SIRIUS_APP_STACK_VIEW_HPP
#define SIRIUS_APP_STACK_VIEW_HPP

// Slice viewer for a (depth, rows, cols) host volume: a slider selects the
// plane, a combo box the intensity window. The window is computed once per
// volume (percentiles over a bounded subsample) and each plane is mapped to
// 8-bit gray on demand into a persistent buffer, so scrubbing through a stack
// does no allocation.

#include <cstdint>
#include <memory>
#include <vector>

#include <QWidget>

#include <sirius/buffer.hpp>

#include "core/display_mapping.hpp"

class QComboBox;
class QLabel;
class QSlider;

namespace sirius::app {

    class ImageCanvas;

    class StackView : public QWidget {
        Q_OBJECT
    public:
        explicit StackView(QWidget* parent = nullptr);

        // Shared ownership: the same volume may be shown here and kept by the
        // session/result at the same time.
        void setVolume(std::shared_ptr<const Buffer<double>> volume);
        void clear();

        std::shared_ptr<const Buffer<double>> volume() const { return volume_; }
        int currentSlice() const;

    private slots:
        void onSliceChanged(int index);
        void onRangeModeChanged(int mode);
        void onHover(int x, int y);

    private:
        void updateRange();
        void renderSlice();
        void updateStatus();

        std::shared_ptr<const Buffer<double>> volume_;
        std::vector<std::uint8_t> gray_;
        DisplayRange range_;

        ImageCanvas* canvas_ = nullptr;
        QSlider* slice_ = nullptr;
        QLabel* sliceLabel_ = nullptr;
        QComboBox* rangeMode_ = nullptr;
        QLabel* status_ = nullptr;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_STACK_VIEW_HPP
