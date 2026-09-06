#ifndef SIRIUS_APP_WIDGETS_CONTROLS_HPP
#define SIRIUS_APP_WIDGETS_CONTROLS_HPP

// Small custom controls the design uses everywhere and stock Qt does not
// have in this shape: a segmented control / tile row (outlined options,
// selected = ink fill or accent fill), a square glyph button (◉ ? ▁ ❐ ⛶ ▲ ▼
// and the tool strip), caption labels (10 px uppercase, 0.1 em tracking),
// rules between regions and a clickable row with hover / selected states.
// Everything paints from theme:: tokens; nothing here reads QSS.

#include <QAbstractButton>
#include <QColor>
#include <QFrame>
#include <QLabel>
#include <QList>
#include <QString>
#include <QStringList>
#include <QWidget>

class QEnterEvent;

namespace sirius::app::widgets {

    // "Ortho | 3D | Compare", "CUDA | CPU | HPC", "sum | mean | max | min".
    class SegmentedControl : public QWidget {
        Q_OBJECT
    public:
        explicit SegmentedControl(QWidget* parent = nullptr);
        SegmentedControl(const QStringList& options, QWidget* parent = nullptr);

        void setOptions(const QStringList& options);
        QStringList options() const { return options_; }
        int currentIndex() const { return current_; }
        void setCurrentIndex(int index);          // does not emit changed()
        QString currentText() const;
        void setCurrentText(const QString& text);
        // Tiles: equal-width, 36 px high, 13 px text (Backend / Cache rows).
        // Segmented: compact, 26 px, 12 px text, hugs its content.
        void setTileMode(bool tiles);
        // Selected option drawn with the accent instead of ink (the "active" semantic).
        void setAccentSelection(bool accent) { accent_ = accent; update(); }
        void setOptionEnabled(int index, bool enabled);
        void setOptionToolTip(int index, const QString& tip);
        QSize sizeHint() const override;
        QSize minimumSizeHint() const override;

    signals:
        void changed(int index);                  // user input only

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent* e) override;
        void mouseMoveEvent(QMouseEvent* e) override;
        void leaveEvent(QEvent*) override;
        bool event(QEvent* e) override;

    private:
        int indexAt(const QPoint& p) const;
        QRect rectOf(int index) const;
        int optionWidth(int index) const;

        QStringList options_;
        QList<bool> enabled_;
        QStringList tips_;
        int current_ = 0;
        int hover_ = -1;
        bool tiles_ = false;
        bool accent_ = false;
    };

    // Square button with a glyph: 1.5 px border, accent fill when active.
    class GlyphButton : public QAbstractButton {
        Q_OBJECT
    public:
        explicit GlyphButton(const QString& glyph, int size = 24, QWidget* parent = nullptr);

        void setGlyph(const QString& glyph);
        void setSize(int w, int h);
        // Active = accent fill + paper glyph (independent of checked so a
        // non-checkable button can still show a state).
        void setActive(bool on);
        bool isActive() const { return active_; }
        void setGlyphPx(int px) { glyphPx_ = px; update(); }
        // No border when idle (the ▲ ▼ reorder arrows).
        void setBorderless(bool on) { borderless_ = on; update(); }
        // Dashed border (the "+" add square).
        void setDashed(bool on) { dashed_ = on; update(); }
        // Idle glyph colour (default: text).
        void setIdleColor(const QColor& c) { idle_ = c; update(); }
        QSize sizeHint() const override { return {w_, h_}; }
        QSize minimumSizeHint() const override { return {w_, h_}; }

    protected:
        void paintEvent(QPaintEvent*) override;
        void enterEvent(QEnterEvent* e) override;
        void leaveEvent(QEvent* e) override;

    private:
        QString glyph_;
        int w_, h_;
        int glyphPx_ = 11;
        bool active_ = false;
        bool hover_ = false;
        bool borderless_ = false;
        bool dashed_ = false;
        QColor idle_;
    };

    // 10 px uppercase, 0.1 em tracking, neutral-600 (or accent).
    class CaptionLabel : public QLabel {
        Q_OBJECT
    public:
        explicit CaptionLabel(const QString& text = {}, QWidget* parent = nullptr);
        void setAccent(bool on);
        void setColor(const QColor& c);
        void setText(const QString& text);   // upper-cases
    };

    // 2 px (region) or 1 px (row) rule.
    class Rule : public QFrame {
        Q_OBJECT
    public:
        explicit Rule(int px = 2, Qt::Orientation orientation = Qt::Horizontal, QWidget* parent = nullptr);
        void setColor(const QColor& c);

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        int px_;
        Qt::Orientation orientation_;
        QColor color_;
    };

    // A row that emits clicked() anywhere on it and paints its own hover /
    // selected / accent-edge states; children are laid out by the caller.
    class ClickRow : public QWidget {
        Q_OBJECT
    public:
        explicit ClickRow(QWidget* parent = nullptr);
        void setSelected(bool on);
        bool isSelected() const { return selected_; }
        void setHoverable(bool on) { hoverable_ = on; }
        void setTopRule(int px) { topRule_ = px; update(); }
        void setEdge(bool on) { edge_ = on; update(); }   // 3 px accent left edge when selected
        void setFill(const QColor& c) { fill_ = c; update(); }   // idle background

    signals:
        void clicked();

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent* e) override;
        void mouseReleaseEvent(QMouseEvent* e) override;
        void enterEvent(QEnterEvent*) override;
        void leaveEvent(QEvent*) override;

    private:
        bool selected_ = false;
        bool hover_ = false;
        bool hoverable_ = true;
        bool pressed_ = false;
        bool edge_ = false;
        int topRule_ = 1;
        QColor fill_;
    };

    // --- helpers ------------------------------------------------------------
    // QLabel with the body font at `px` and an optional colour / weight.
    QLabel* label(const QString& text, int px, const QColor& color, int weight = -1, QWidget* parent = nullptr);
    // Heading (800) label.
    QLabel* heading(const QString& text, int px, QWidget* parent = nullptr);
    // Sets the "class" dynamic property that theme.cpp's QSS styles
    // ("primary", "secondary", "ghost", "link").
    void setButtonClass(QWidget* button, const char* cls);
    // Small colour chip (w × h, filled with `color`, no border).
    QWidget* colorChip(const QColor& color, int w = 10, int h = 10, QWidget* parent = nullptr);
    void setChipColor(QWidget* chip, const QColor& color);
    // Elide a string to a width with the widget's font.
    QString elide(const QWidget* w, const QString& s, int width);
    // Tabular figures on a widget's font.
    void useTabularNumbers(QWidget* w);

} // namespace sirius::app::widgets

#endif // SIRIUS_APP_WIDGETS_CONTROLS_HPP
