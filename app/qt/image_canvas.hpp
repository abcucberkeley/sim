#ifndef SIRIUS_APP_IMAGE_CANVAS_HPP
#define SIRIUS_APP_IMAGE_CANVAS_HPP

// Zoomable, pannable view of one grayscale image with an optional pixel
// aspect ratio, a rubber-band selection, a crosshair and vector overlays,
// all expressed in image coordinates. The image is scaled inside paintEvent
// through a transform (no scaled copy is stored), so zooming is a repaint.
//
// Mouse: wheel zooms around the cursor, left drag pans (or selects when the
// selection mode is on), middle drag always pans, a left click without a
// drag reports the pixel.

#include <vector>

#include <QColor>
#include <QImage>
#include <QPointF>
#include <QRect>
#include <QString>
#include <QTransform>
#include <QWidget>

namespace sirius::app {

    struct CanvasOverlay {
        enum class Kind { Circle, Cross, Label };
        Kind kind = Kind::Cross;
        double x = 0.0, y = 0.0;     // image coordinates (pixel centers at integer + 0.5)
        double rx = 0.0, ry = 0.0;   // circle radii in image pixels
        QColor color = Qt::yellow;
        QString text;                // label, or annotation drawn next to a cross / circle
    };

    class ImageCanvas : public QWidget {
        Q_OBJECT
    public:
        explicit ImageCanvas(QWidget* parent = nullptr);

        // `image` must not reference memory that is freed while shown; the
        // caller re-sets it whenever the backing buffer changes. A new image
        // size refits the view, the same size keeps zoom and pan.
        void setImage(const QImage& image);
        void clearImage();
        bool hasImage() const { return !image_.isNull(); }
        QSize imageSize() const { return image_.size(); }

        // On-screen stretch of image pixels, e.g. dz / dx for orthogonal slices.
        void setPixelAspect(double xScale, double yScale);

        // Screen pixels per image pixel (before the aspect stretch).
        double zoom() const;
        void setZoom(double z);        // around the widget center
        void zoomBy(double factor);
        void fitToWindow();
        bool fitsToWindow() const { return fit_; }

        // Locks zoom and pan: the image stays fitted to the window and the
        // wheel and drags do nothing, while clicks still report the pixel.
        // Used while the orthogonal views share a crosshair.
        void setNavigationLocked(bool locked);
        bool navigationLocked() const { return locked_; }

        void setSelectionMode(bool on);
        bool selectionMode() const { return selectMode_; }
        QRect selection() const { return selection_; }   // image pixels, empty when none
        void clearSelection();

        void setCrosshair(QPointF imagePos, bool visible);
        void setOverlays(std::vector<CanvasOverlay> overlays);

    signals:
        void hovered(int x, int y);              // image pixel; (-1, -1) off the image
        void clicked(int x, int y);              // left click without a drag
        void doubleClicked(int x, int y);        // left double click on a pixel
        void selectionChanged(QRect selection);  // image pixels; empty when cleared
        void zoomChanged(double zoom);

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent* event) override;
        void mouseMoveEvent(QMouseEvent* event) override;
        void mouseReleaseEvent(QMouseEvent* event) override;
        void mouseDoubleClickEvent(QMouseEvent* event) override;
        void wheelEvent(QWheelEvent* event) override;
        void leaveEvent(QEvent*) override;

    private:
        double fitZoom() const;
        QTransform imageToWidget() const;
        QPointF toImage(QPointF widgetPos) const;
        QPoint imagePixel(QPointF widgetPos) const;   // (-1, -1) outside the image
        void zoomAround(double newZoom, QPointF widgetAnchor);
        void leaveFitMode();

        QImage image_;
        double aspectX_ = 1.0, aspectY_ = 1.0;
        bool fit_ = true;
        double zoom_ = 1.0;
        QPointF offset_;   // widget position of the image origin when not fitting
        bool locked_ = false;
        bool selectMode_ = false;
        QRect selection_;
        bool crosshairVisible_ = false;
        QPointF crosshair_;
        std::vector<CanvasOverlay> overlays_;

        enum class Drag { None, Pan, Select };
        Drag drag_ = Drag::None;
        QPointF dragStart_;
        QPointF dragOffset0_;
        QPoint selectStart_;
        bool moved_ = false;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_IMAGE_CANVAS_HPP
