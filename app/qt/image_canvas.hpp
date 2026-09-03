#ifndef SIRIUS_APP_IMAGE_CANVAS_HPP
#define SIRIUS_APP_IMAGE_CANVAS_HPP

// Paints one grayscale image scaled to fit the widget (aspect preserved,
// centered) and reports the image pixel under the mouse. Scaling happens in
// paintEvent, so no scaled copy is stored and a resize costs one repaint.

#include <QImage>
#include <QWidget>

namespace sirius::app {

    class ImageCanvas : public QWidget {
        Q_OBJECT
    public:
        explicit ImageCanvas(QWidget* parent = nullptr);

        // `image` must not reference memory that is freed while shown; the
        // caller re-sets it whenever the backing buffer changes.
        void setImage(const QImage& image);
        void clearImage();

    signals:
        // Image coordinates under the cursor; (-1, -1) when leaving the image.
        void hovered(int x, int y);

    protected:
        void paintEvent(QPaintEvent*) override;
        void mouseMoveEvent(QMouseEvent* event) override;
        void leaveEvent(QEvent*) override;

    private:
        QRect targetRect() const;   // where the image is drawn in widget coordinates

        QImage image_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_IMAGE_CANVAS_HPP
