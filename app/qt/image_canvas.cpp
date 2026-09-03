#include "qt/image_canvas.hpp"

#include <QMouseEvent>
#include <QPainter>

namespace sirius::app {

    ImageCanvas::ImageCanvas(QWidget* parent) : QWidget(parent) {
        setMouseTracking(true);
        setMinimumSize(64, 64);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }

    void ImageCanvas::setImage(const QImage& image) {
        image_ = image;
        update();
    }

    void ImageCanvas::clearImage() {
        image_ = QImage();
        update();
    }

    QRect ImageCanvas::targetRect() const {
        if (image_.isNull()) return {};
        QSize fitted = image_.size();
        fitted.scale(size(), Qt::KeepAspectRatio);
        const QPoint origin((width() - fitted.width()) / 2, (height() - fitted.height()) / 2);
        return QRect(origin, fitted);
    }

    void ImageCanvas::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.fillRect(rect(), Qt::black);
        if (image_.isNull()) return;
        // nearest-neighbour when magnifying keeps pixels crisp; smooth when
        // shrinking avoids aliasing on large frames
        const QRect target = targetRect();
        painter.setRenderHint(QPainter::SmoothPixmapTransform, target.width() < image_.width());
        painter.drawImage(target, image_);
    }

    void ImageCanvas::mouseMoveEvent(QMouseEvent* event) {
        const QRect target = targetRect();
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        const QPoint p = event->position().toPoint();
#else
        const QPoint p = event->pos();
#endif
        if (target.isEmpty() || !target.contains(p)) {
            emit hovered(-1, -1);
            return;
        }
        const int x = (p.x() - target.x()) * image_.width() / target.width();
        const int y = (p.y() - target.y()) * image_.height() / target.height();
        emit hovered(qBound(0, x, image_.width() - 1), qBound(0, y, image_.height() - 1));
    }

    void ImageCanvas::leaveEvent(QEvent*) { emit hovered(-1, -1); }

} // namespace sirius::app
