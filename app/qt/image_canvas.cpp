#include "qt/image_canvas.hpp"

#include <algorithm>
#include <cmath>

#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QWheelEvent>

namespace sirius::app {

    namespace {
        constexpr double kMinZoom = 1.0 / 64.0;
        constexpr double kMaxZoom = 256.0;

        QPointF eventPos(const QMouseEvent* e) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
            return e->position();
#else
            return e->localPos();
#endif
        }
        QPointF eventPos(const QWheelEvent* e) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
            return e->position();
#else
            return e->posF();
#endif
        }
    } // namespace

    ImageCanvas::ImageCanvas(QWidget* parent) : QWidget(parent) {
        setMouseTracking(true);
        setMinimumSize(64, 64);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setCursor(Qt::CrossCursor);
    }

    void ImageCanvas::setImage(const QImage& image) {
        const bool sameSize = !image_.isNull() && image_.size() == image.size();
        image_ = image;
        if (!sameSize) {
            fit_ = true;
            selection_ = QRect();
            emit zoomChanged(zoom());
        }
        update();
    }

    void ImageCanvas::clearImage() {
        image_ = QImage();
        selection_ = QRect();
        fit_ = true;
        update();
    }

    void ImageCanvas::setPixelAspect(double xScale, double yScale) {
        aspectX_ = xScale > 0.0 ? xScale : 1.0;
        aspectY_ = yScale > 0.0 ? yScale : 1.0;
        update();
    }

    double ImageCanvas::fitZoom() const {
        if (image_.isNull()) return 1.0;
        const double zx = width() / (image_.width() * aspectX_);
        const double zy = height() / (image_.height() * aspectY_);
        return std::max(kMinZoom, std::min(zx, zy));
    }

    double ImageCanvas::zoom() const { return fit_ ? fitZoom() : zoom_; }

    QTransform ImageCanvas::imageToWidget() const {
        const double z = zoom();
        const double sx = z * aspectX_, sy = z * aspectY_;
        QPointF origin = offset_;
        if (fit_)
            origin = QPointF((width() - image_.width() * sx) / 2.0, (height() - image_.height() * sy) / 2.0);
        QTransform t;
        t.translate(origin.x(), origin.y());
        t.scale(sx, sy);
        return t;
    }

    QPointF ImageCanvas::toImage(QPointF widgetPos) const { return imageToWidget().inverted().map(widgetPos); }

    QPoint ImageCanvas::imagePixel(QPointF widgetPos) const {
        if (image_.isNull()) return {-1, -1};
        const QPointF p = toImage(widgetPos);
        const int x = static_cast<int>(std::floor(p.x()));
        const int y = static_cast<int>(std::floor(p.y()));
        if (x < 0 || y < 0 || x >= image_.width() || y >= image_.height()) return {-1, -1};
        return {x, y};
    }

    void ImageCanvas::leaveFitMode() {
        if (!fit_) return;
        const QTransform t = imageToWidget();
        zoom_ = fitZoom();
        offset_ = t.map(QPointF(0, 0));
        fit_ = false;
    }

    void ImageCanvas::setNavigationLocked(bool locked) {
        locked_ = locked;
        if (locked) fitToWindow();
        setCursor(locked ? Qt::CrossCursor : (selectMode_ ? Qt::CrossCursor : Qt::OpenHandCursor));
    }

    void ImageCanvas::zoomAround(double newZoom, QPointF anchor) {
        if (image_.isNull() || locked_) return;
        newZoom = std::clamp(newZoom, kMinZoom, kMaxZoom);
        const QPointF imagePt = toImage(anchor);   // stays under the anchor
        leaveFitMode();
        zoom_ = newZoom;
        offset_ = anchor - QPointF(imagePt.x() * zoom_ * aspectX_, imagePt.y() * zoom_ * aspectY_);
        update();
        emit zoomChanged(zoom_);
    }

    void ImageCanvas::setZoom(double z) { zoomAround(z, QPointF(width() / 2.0, height() / 2.0)); }
    void ImageCanvas::zoomBy(double factor) { setZoom(zoom() * factor); }

    void ImageCanvas::fitToWindow() {
        fit_ = true;
        update();
        emit zoomChanged(zoom());
    }

    void ImageCanvas::setSelectionMode(bool on) {
        selectMode_ = on;
        setCursor(on || locked_ ? Qt::CrossCursor : Qt::OpenHandCursor);
        if (!on) clearSelection();
    }

    void ImageCanvas::clearSelection() {
        if (selection_.isNull()) return;
        selection_ = QRect();
        update();
        emit selectionChanged(selection_);
    }

    void ImageCanvas::setCrosshair(QPointF imagePos, bool visible) {
        crosshair_ = imagePos;
        crosshairVisible_ = visible;
        update();
    }

    void ImageCanvas::setOverlays(std::vector<CanvasOverlay> overlays) {
        overlays_ = std::move(overlays);
        update();
    }

    // --- painting --------------------------------------------------------

    void ImageCanvas::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.fillRect(rect(), Qt::black);
        if (image_.isNull()) return;

        const QTransform t = imageToWidget();
        painter.save();
        painter.setTransform(t);
        // nearest-neighbour when magnifying keeps pixels crisp; smooth when
        // shrinking avoids aliasing on large frames
        painter.setRenderHint(QPainter::SmoothPixmapTransform, zoom() < 1.0);
        painter.drawImage(QPointF(0, 0), image_);
        painter.restore();

        painter.setRenderHint(QPainter::Antialiasing, true);
        const double sx = zoom() * aspectX_, sy = zoom() * aspectY_;
        for (const CanvasOverlay& o : overlays_) {
            const QPointF c = t.map(QPointF(o.x, o.y));
            painter.setPen(QPen(o.color, 1.0));
            switch (o.kind) {
                case CanvasOverlay::Kind::Circle:
                    painter.drawEllipse(c, o.rx * sx, o.ry * sy);
                    break;
                case CanvasOverlay::Kind::Cross:
                    painter.drawLine(c + QPointF(-5, 0), c + QPointF(5, 0));
                    painter.drawLine(c + QPointF(0, -5), c + QPointF(0, 5));
                    break;
                case CanvasOverlay::Kind::Label:
                    break;
            }
            if (!o.text.isEmpty()) painter.drawText(c + QPointF(6, -4), o.text);
        }

        if (crosshairVisible_) {
            const QPointF c = t.map(crosshair_);
            painter.setPen(QPen(QColor(255, 255, 0, 160), 1.0));
            painter.drawLine(QPointF(0, c.y()), QPointF(width(), c.y()));
            painter.drawLine(QPointF(c.x(), 0), QPointF(c.x(), height()));
        }

        if (!selection_.isNull()) {
            QPen pen(Qt::white, 1.0, Qt::DashLine);
            painter.setPen(pen);
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(t.mapRect(QRectF(selection_)));
        }
    }

    // --- mouse -----------------------------------------------------------

    void ImageCanvas::mousePressEvent(QMouseEvent* event) {
        if (image_.isNull()) return;
        const QPointF pos = eventPos(event);
        moved_ = false;
        dragStart_ = pos;
        if (event->button() == Qt::MiddleButton || (event->button() == Qt::LeftButton && !selectMode_)) {
            // a locked canvas keeps the "pan" drag only to detect a plain click
            if (!locked_) leaveFitMode();
            dragOffset0_ = offset_;
            drag_ = Drag::Pan;
            if (!locked_) setCursor(Qt::ClosedHandCursor);
        } else if (event->button() == Qt::LeftButton) {
            const QPoint p = imagePixel(pos);
            if (p.x() < 0) return;
            selectStart_ = p;
            selection_ = QRect(p, QSize(1, 1));
            drag_ = Drag::Select;
            update();
        }
    }

    void ImageCanvas::mouseMoveEvent(QMouseEvent* event) {
        const QPointF pos = eventPos(event);
        const QPoint p = imagePixel(pos);
        emit hovered(p.x(), p.y());
        if (drag_ == Drag::Pan) {
            if (locked_) return;
            offset_ = dragOffset0_ + (pos - dragStart_);
            moved_ = true;
            update();
        } else if (drag_ == Drag::Select) {
            const QPointF img = toImage(pos);
            const int x = std::clamp(static_cast<int>(std::floor(img.x())), 0, image_.width() - 1);
            const int y = std::clamp(static_cast<int>(std::floor(img.y())), 0, image_.height() - 1);
            const QPoint a(std::min(x, selectStart_.x()), std::min(y, selectStart_.y()));
            const QPoint b(std::max(x, selectStart_.x()), std::max(y, selectStart_.y()));
            selection_ = QRect(a, b);
            moved_ = true;
            update();
        }
    }

    void ImageCanvas::mouseReleaseEvent(QMouseEvent* event) {
        const QPointF pos = eventPos(event);
        const Drag drag = drag_;
        drag_ = Drag::None;
        if (drag == Drag::Pan) {
            setCursor(selectMode_ || locked_ ? Qt::CrossCursor : Qt::OpenHandCursor);
            if (!moved_ && event->button() == Qt::LeftButton) {
                const QPoint p = imagePixel(pos);
                if (p.x() >= 0) emit clicked(p.x(), p.y());
            }
        } else if (drag == Drag::Select) {
            if (!moved_) {
                selection_ = QRect();
                const QPoint p = imagePixel(pos);
                if (p.x() >= 0) emit clicked(p.x(), p.y());
            }
            update();
            emit selectionChanged(selection_);
        }
    }

    void ImageCanvas::mouseDoubleClickEvent(QMouseEvent* event) {
        if (event->button() != Qt::LeftButton || image_.isNull()) return;
        const QPoint p = imagePixel(eventPos(event));
        if (p.x() >= 0) emit doubleClicked(p.x(), p.y());
    }

    void ImageCanvas::wheelEvent(QWheelEvent* event) {
        if (image_.isNull() || locked_) return;
        const double steps = event->angleDelta().y() / 120.0;
        if (steps == 0.0) return;
        zoomAround(zoom() * std::pow(1.25, steps), eventPos(event));
        event->accept();
    }

    void ImageCanvas::leaveEvent(QEvent*) { emit hovered(-1, -1); }

} // namespace sirius::app
