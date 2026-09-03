#include "qt/stack_view.hpp"
#include "qt/image_canvas.hpp"

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QVBoxLayout>

namespace sirius::app {

    namespace {
        enum RangeMode { PercentileRange = 0, MinMaxRange = 1 };
    }

    StackView::StackView(QWidget* parent) : QWidget(parent) {
        canvas_ = new ImageCanvas(this);

        slice_ = new QSlider(Qt::Horizontal, this);
        slice_->setEnabled(false);
        sliceLabel_ = new QLabel(tr("-"), this);
        sliceLabel_->setMinimumWidth(80);

        rangeMode_ = new QComboBox(this);
        rangeMode_->addItem(tr("Window: 0.1% - 99.9%"), PercentileRange);
        rangeMode_->addItem(tr("Window: min - max"), MinMaxRange);

        status_ = new QLabel(this);
        status_->setTextInteractionFlags(Qt::TextSelectableByMouse);

        auto* controls = new QHBoxLayout;
        controls->addWidget(new QLabel(tr("Slice"), this));
        controls->addWidget(slice_, 1);
        controls->addWidget(sliceLabel_);
        controls->addWidget(rangeMode_);

        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addWidget(canvas_, 1);
        layout->addLayout(controls);
        layout->addWidget(status_);

        connect(slice_, &QSlider::valueChanged, this, &StackView::onSliceChanged);
        connect(rangeMode_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &StackView::onRangeModeChanged);
        connect(canvas_, &ImageCanvas::hovered, this, &StackView::onHover);
    }

    void StackView::setVolume(std::shared_ptr<const Buffer<double>> volume) {
        if (!volume || volume->rank() != 3 || volume->empty() || !volume->device().isCpu()) {
            clear();
            return;
        }
        volume_ = std::move(volume);
        gray_.resize(static_cast<std::size_t>(volume_->dim(1) * volume_->dim(2)));

        const int depth = static_cast<int>(volume_->dim(0));
        {
            const QSignalBlocker block(slice_);
            slice_->setRange(0, depth - 1);
            slice_->setValue(std::min(slice_->value(), depth - 1));
        }
        slice_->setEnabled(depth > 1);
        updateRange();
        renderSlice();
    }

    void StackView::clear() {
        volume_.reset();
        gray_.clear();
        canvas_->clearImage();
        slice_->setEnabled(false);
        sliceLabel_->setText(tr("-"));
        status_->clear();
    }

    int StackView::currentSlice() const { return slice_->value(); }

    void StackView::updateRange() {
        if (!volume_) return;
        const double* values = volume_->data();
        const Index n = volume_->size();
        range_ = rangeMode_->currentData().toInt() == MinMaxRange
                     ? minMaxRange(values, n)
                     : percentileRange(values, n, 0.001, 0.999);
    }

    void StackView::renderSlice() {
        if (!volume_) return;
        const Index ny = volume_->dim(1), nx = volume_->dim(2);
        const Index z = slice_->value();
        const double* plane = volume_->data() + z * ny * nx;
        mapToGray8(plane, ny * nx, range_, gray_.data());

        // QImage wraps gray_ without copying; the canvas keeps the wrapper,
        // which stays valid because gray_ is only resized in setVolume.
        canvas_->setImage(QImage(gray_.data(), static_cast<int>(nx), static_cast<int>(ny),
                                 static_cast<int>(nx), QImage::Format_Grayscale8));
        sliceLabel_->setText(QStringLiteral("%1 / %2").arg(z + 1).arg(volume_->dim(0)));
        updateStatus();
    }

    void StackView::updateStatus() {
        if (!volume_) return;
        status_->setText(tr("%1 x %2 x %3   window [%4, %5]")
                             .arg(volume_->dim(0)).arg(volume_->dim(1)).arg(volume_->dim(2))
                             .arg(range_.lo, 0, 'g', 5).arg(range_.hi, 0, 'g', 5));
    }

    void StackView::onSliceChanged(int) { renderSlice(); }

    void StackView::onRangeModeChanged(int) {
        updateRange();
        renderSlice();
    }

    void StackView::onHover(int x, int y) {
        if (!volume_) return;
        if (x < 0 || y < 0) {
            updateStatus();
            return;
        }
        const Index ny = volume_->dim(1), nx = volume_->dim(2);
        const Index z = slice_->value();
        const double v = volume_->data()[(z * ny + y) * nx + x];
        status_->setText(tr("(z %1, y %2, x %3) = %4").arg(z).arg(y).arg(x).arg(v, 0, 'g', 7));
    }

} // namespace sirius::app
