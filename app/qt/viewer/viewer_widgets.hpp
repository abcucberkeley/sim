#ifndef SIRIUS_APP_VIEWER_WIDGETS_HPP
#define SIRIUS_APP_VIEWER_WIDGETS_HPP

// Small painted controls of the viewer chrome, drawn straight from the
// theme tokens so they look right with or without the application
// stylesheet: the 14 px token checkbox with an optional uppercase caption,
// the 22 px channel swatch and the two-handle range slider of the 3D clip.
// The icon squares and the Ortho | 3D | Compare control are the shared
// widgets::GlyphButton / widgets::SegmentedControl.
//
// All three take keyboard focus and draw the design's 2 px accent focus
// ring: space or enter toggles a check or a swatch, the arrow keys move the
// range slider's handles.

#include <QAbstractButton>
#include <QColor>
#include <QString>
#include <QWidget>

class QFocusEvent;
class QKeyEvent;

namespace sirius::app {

    // 14 x 14 square box (accent fill when checked) + 12 px label + optional
    // 10 px uppercase caption ("LOCKED").
    class TokenCheck : public QAbstractButton {
        Q_OBJECT
    public:
        explicit TokenCheck(const QString& label, QWidget* parent = nullptr);
        void setCaption(const QString& caption);
        QSize sizeHint() const override;

    protected:
        void paintEvent(QPaintEvent*) override;
        void keyPressEvent(QKeyEvent*) override;
        void focusInEvent(QFocusEvent*) override;
        void focusOutEvent(QFocusEvent*) override;

    private:
        QString caption_;
    };

    // 22 x 22 channel square labelled with the wavelength: filled when the
    // channel is visible, outlined when hidden.
    class ChannelSwatch : public QAbstractButton {
        Q_OBJECT
    public:
        ChannelSwatch(const QString& label, const QColor& color, QWidget* parent = nullptr);
        QSize sizeHint() const override { return {22, 22}; }
        QSize minimumSizeHint() const override { return {22, 22}; }

    protected:
        void paintEvent(QPaintEvent*) override;
        void keyPressEvent(QKeyEvent*) override;
        void focusInEvent(QFocusEvent*) override;
        void focusOutEvent(QFocusEvent*) override;

    private:
        QString label_;
        QColor color_;
    };

    // Horizontal two-handle range over [0, 1], drawn as the prototype's clip
    // bar: a 2 px translucent track with the selected span in accent.
    class RangeSlider : public QWidget {
        Q_OBJECT
    public:
        explicit RangeSlider(QWidget* parent = nullptr);
        double low() const noexcept { return lo_; }
        double high() const noexcept { return hi_; }
        void setRange(double lo, double hi);
        QSize sizeHint() const override { return {120, 14}; }

    signals:
        void rangeChanged(double lo, double hi);

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent*) override;
        void mouseMoveEvent(QMouseEvent*) override;
        void mouseReleaseEvent(QMouseEvent*) override { drag_ = 0; }
        void keyPressEvent(QKeyEvent*) override;
        void focusInEvent(QFocusEvent*) override;
        void focusOutEvent(QFocusEvent*) override;

    private:
        double fromX(double x) const;
        void describe();
        double lo_ = 0.0, hi_ = 1.0;
        int drag_ = 0;      // 1 = low handle, 2 = high handle
        int handle_ = 1;    // the handle the arrow keys move
    };

    // Text drawn like the prototype's overlay labels (11 px, viewer text
    // colour, optional opacity), for use over the viewer ground.
    void drawOverlayText(QPainter& p, const QPointF& topLeft, const QString& text, bool bold = false,
                         double opacity = 1.0, int px = 11);

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_WIDGETS_HPP
