#ifndef SIRIUS_APP_VIEWER_WIDGETS_HPP
#define SIRIUS_APP_VIEWER_WIDGETS_HPP

// Small painted controls of the viewer chrome, drawn straight from the
// theme tokens so they look right with or without the application
// stylesheet: glyph squares (tool strip, play button, zoom +/-), the
// segmented Ortho | 3D | Compare control, the 14 px token checkbox with an
// optional uppercase caption, the 22 px channel swatch and the two-handle
// range slider of the 3D clip.

#include <QAbstractButton>
#include <QColor>
#include <QString>
#include <QWidget>

namespace sirius::app {

    // A square (or given size) button showing one glyph; checked = accent
    // fill + paper glyph, hover = accent border. `dim` draws it at 35 %.
    class GlyphButton : public QAbstractButton {
        Q_OBJECT
    public:
        explicit GlyphButton(const QString& glyph, QWidget* parent = nullptr, QSize size = QSize(28, 28));
        void setGlyph(const QString& glyph);
        void setGlyphPx(int px) { glyphPx_ = px; update(); }
        void setBorderColor(const QColor& c) { border_ = c; update(); }
        // Draw on the viewer ground (light text, translucent border) instead of the panel ground.
        void setOnDark(bool on) { onDark_ = on; update(); }
        QSize sizeHint() const override { return size_; }
        QSize minimumSizeHint() const override { return size_; }

    protected:
        void paintEvent(QPaintEvent*) override;
        void enterEvent(QEnterEvent*) override { hover_ = true; update(); }
        void leaveEvent(QEvent*) override { hover_ = false; update(); }

    private:
        QString glyph_;
        QSize size_;
        int glyphPx_ = 13;
        QColor border_;
        bool hover_ = false;
        bool onDark_ = false;
    };

    // Ortho | 3D | Compare: outlined row, selected = ink fill, paper text.
    class SegmentedControl : public QWidget {
        Q_OBJECT
    public:
        explicit SegmentedControl(const QStringList& items, QWidget* parent = nullptr);
        int current() const noexcept { return current_; }
        void setCurrent(int index);
        QSize sizeHint() const override;

    signals:
        void changed(int index);

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent*) override;
        void mouseMoveEvent(QMouseEvent*) override;
        void leaveEvent(QEvent*) override { hover_ = -1; update(); }

    private:
        int indexAt(int x) const;
        QStringList items_;
        int current_ = 0;
        int hover_ = -1;
    };

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

    private:
        double fromX(double x) const;
        double lo_ = 0.0, hi_ = 1.0;
        int drag_ = 0;   // 1 = low handle, 2 = high handle
    };

    // Text drawn like the prototype's overlay labels (11 px, viewer text
    // colour, optional opacity), for use over the viewer ground.
    void drawOverlayText(QPainter& p, const QPointF& topLeft, const QString& text, bool bold = false,
                         double opacity = 1.0, int px = 11);

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_WIDGETS_HPP
