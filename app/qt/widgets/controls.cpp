#include "qt/widgets/controls.hpp"

#include <algorithm>

#include <QEnterEvent>
#include <QEvent>
#include <QFontMetrics>
#include <QHelpEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QStyle>
#include <QToolTip>

#include "qt/theme.hpp"

namespace sirius::app::widgets {

    // --- SegmentedControl -----------------------------------------------------

    SegmentedControl::SegmentedControl(QWidget* parent) : QWidget(parent) {
        setMouseTracking(true);
        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        setFont(theme::font(12));
    }

    SegmentedControl::SegmentedControl(const QStringList& options, QWidget* parent) : SegmentedControl(parent) {
        setOptions(options);
    }

    void SegmentedControl::setOptions(const QStringList& options) {
        options_ = options;
        enabled_.clear();
        tips_.clear();
        for (int i = 0; i < options_.size(); ++i) {
            enabled_.push_back(true);
            tips_.push_back(QString());
        }
        current_ = options_.isEmpty() ? -1 : std::min(std::max(current_, 0), static_cast<int>(options_.size()) - 1);
        updateGeometry();
        update();
    }

    void SegmentedControl::setCurrentIndex(int index) {
        if (index < 0 || index >= options_.size() || current_ == index) return;
        current_ = index;
        update();
    }

    QString SegmentedControl::currentText() const {
        return current_ >= 0 && current_ < options_.size() ? options_[current_] : QString();
    }

    void SegmentedControl::setCurrentText(const QString& text) {
        const int i = static_cast<int>(options_.indexOf(text));
        if (i >= 0) setCurrentIndex(i);
    }

    void SegmentedControl::setTileMode(bool tiles) {
        tiles_ = tiles;
        setFont(theme::font(tiles ? 13 : 12));
        setSizePolicy(tiles ? QSizePolicy::Expanding : QSizePolicy::Preferred, QSizePolicy::Fixed);
        updateGeometry();
        update();
    }

    void SegmentedControl::setOptionEnabled(int index, bool enabled) {
        if (index < 0 || index >= enabled_.size()) return;
        enabled_[index] = enabled;
        update();
    }

    void SegmentedControl::setOptionToolTip(int index, const QString& tip) {
        if (index < 0 || index >= tips_.size()) return;
        tips_[index] = tip;
    }

    int SegmentedControl::optionWidth(int index) const {
        if (tiles_) {
            const int n = std::max(1, static_cast<int>(options_.size()));
            const int gaps = 2 * (n - 1);
            const int base = (width() - gaps) / n;
            return index == n - 1 ? width() - gaps - base * (n - 1) : base;
        }
        const QFontMetrics fm(font());
        return fm.horizontalAdvance(options_[index]) + 22;
    }

    QRect SegmentedControl::rectOf(int index) const {
        int x = 0;
        for (int i = 0; i < index; ++i) x += optionWidth(i) + 2;
        return QRect(x, 0, optionWidth(index), height());
    }

    QSize SegmentedControl::sizeHint() const {
        const int h = tiles_ ? 36 : 26;
        if (tiles_) return {static_cast<int>(options_.size()) * 60, h};
        int w = 0;
        const QFontMetrics fm(font());
        for (int i = 0; i < options_.size(); ++i) w += fm.horizontalAdvance(options_[i]) + 22 + (i ? 2 : 0);
        return {w, h};
    }

    QSize SegmentedControl::minimumSizeHint() const {
        return tiles_ ? QSize(static_cast<int>(options_.size()) * 40, 36) : sizeHint();
    }

    int SegmentedControl::indexAt(const QPoint& p) const {
        for (int i = 0; i < options_.size(); ++i)
            if (rectOf(i).contains(p)) return i;
        return -1;
    }

    void SegmentedControl::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.setFont(font());
        const QColor selFill = accent_ ? theme::kAccent : theme::kText;
        for (int i = 0; i < options_.size(); ++i) {
            const QRect r = rectOf(i);
            const bool sel = i == current_;
            const bool en = enabled_[i] && isEnabled();
            painter.setOpacity(en ? 1.0 : 0.45);
            QPen pen(sel ? selFill : (hover_ == i && en ? theme::kAccent : theme::kDivider), 1.5);
            painter.setPen(pen);
            painter.setBrush(sel ? QBrush(selFill) : Qt::NoBrush);
            painter.drawRect(QRectF(r).adjusted(0.75, 0.75, -0.75, -0.75));
            painter.setPen(sel ? theme::kBg : theme::kText);
            painter.drawText(r, Qt::AlignCenter, options_[i]);
        }
    }

    void SegmentedControl::mousePressEvent(QMouseEvent* e) {
        if (e->button() != Qt::LeftButton) return;
        const int i = indexAt(e->pos());
        if (i < 0 || !enabled_[i]) return;
        if (i != current_) {
            current_ = i;
            update();
            emit changed(i);
        }
    }

    void SegmentedControl::mouseMoveEvent(QMouseEvent* e) {
        const int i = indexAt(e->pos());
        if (i != hover_) {
            hover_ = i;
            update();
        }
    }

    void SegmentedControl::leaveEvent(QEvent*) {
        hover_ = -1;
        update();
    }

    bool SegmentedControl::event(QEvent* e) {
        if (e->type() == QEvent::ToolTip) {
            auto* he = static_cast<QHelpEvent*>(e);
            const int i = indexAt(he->pos());
            if (i >= 0 && !tips_[i].isEmpty()) QToolTip::showText(he->globalPos(), tips_[i], this);
            else QToolTip::hideText();
            return true;
        }
        return QWidget::event(e);
    }

    // --- GlyphButton ------------------------------------------------------------

    GlyphButton::GlyphButton(const QString& glyph, int size, QWidget* parent)
        : QAbstractButton(parent), glyph_(glyph), w_(size), h_(size), idle_(theme::kText) {
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::NoFocus);
    }

    void GlyphButton::setGlyph(const QString& glyph) {
        glyph_ = glyph;
        update();
    }

    void GlyphButton::setSize(int w, int h) {
        w_ = w;
        h_ = h;
        setFixedSize(w, h);
        updateGeometry();
    }

    void GlyphButton::setActive(bool on) {
        if (active_ == on) return;
        active_ = on;
        update();
    }

    void GlyphButton::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.setOpacity(isEnabled() ? 1.0 : 0.35);
        const bool on = active_ || isChecked();
        QColor border = on ? theme::kAccent : (hover_ && isEnabled() ? theme::kAccent : theme::kDivider);
        if (borderless_ && !on) border = Qt::transparent;
        QPen pen(border, 1.5);
        if (dashed_ && !on) pen.setStyle(Qt::DashLine);
        painter.setPen(pen);
        painter.setBrush(on ? QBrush(theme::kAccent) : Qt::NoBrush);
        painter.drawRect(QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75));
        painter.setPen(on ? theme::kBg : (hover_ && borderless_ ? theme::kAccent : idle_));
        painter.setFont(theme::font(glyphPx_));
        painter.drawText(rect(), Qt::AlignCenter, glyph_);
    }

    void GlyphButton::enterEvent(QEnterEvent* e) {
        hover_ = true;
        update();
        QAbstractButton::enterEvent(e);
    }

    void GlyphButton::leaveEvent(QEvent* e) {
        hover_ = false;
        update();
        QAbstractButton::leaveEvent(e);
    }

    // --- CaptionLabel ------------------------------------------------------------

    CaptionLabel::CaptionLabel(const QString& text, QWidget* parent) : QLabel(parent) {
        setFont(theme::caption());
        setAccent(false);
        setText(text);
    }

    void CaptionLabel::setAccent(bool on) { setColor(on ? theme::kAccent : theme::kNeutral600); }

    void CaptionLabel::setColor(const QColor& c) {
        QPalette p = palette();
        p.setColor(QPalette::WindowText, c);
        setPalette(p);
    }

    void CaptionLabel::setText(const QString& text) { QLabel::setText(text.toUpper()); }

    // --- Rule ------------------------------------------------------------------------

    Rule::Rule(int px, Qt::Orientation orientation, QWidget* parent)
        : QFrame(parent), px_(px), orientation_(orientation), color_(theme::kDivider) {
        setFrameShape(QFrame::NoFrame);
        if (orientation == Qt::Horizontal) {
            setFixedHeight(px);
            setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        } else {
            setFixedWidth(px);
            setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        }
    }

    void Rule::setColor(const QColor& c) {
        color_ = c;
        update();
    }

    void Rule::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.fillRect(rect(), color_);
    }

    // --- ClickRow ------------------------------------------------------------------

    ClickRow::ClickRow(QWidget* parent) : QWidget(parent) {
        setAttribute(Qt::WA_Hover, true);
        setCursor(Qt::PointingHandCursor);
    }

    void ClickRow::setSelected(bool on) {
        if (selected_ == on) return;
        selected_ = on;
        update();
    }

    void ClickRow::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        if (selected_) painter.fillRect(rect(), theme::kSurface);
        else if (hover_ && hoverable_) painter.fillRect(rect(), theme::kNeutral200);
        else if (fill_.isValid()) painter.fillRect(rect(), fill_);
        if (topRule_ > 0) painter.fillRect(QRect(0, 0, width(), topRule_), theme::kDivider);
        if (edge_ && selected_) painter.fillRect(QRect(0, 0, 3, height()), theme::kAccent);
    }

    void ClickRow::mousePressEvent(QMouseEvent* e) {
        if (e->button() == Qt::LeftButton) pressed_ = true;
    }

    void ClickRow::mouseReleaseEvent(QMouseEvent* e) {
        if (e->button() == Qt::LeftButton && pressed_ && rect().contains(e->pos())) emit clicked();
        pressed_ = false;
    }

    void ClickRow::enterEvent(QEnterEvent*) {
        hover_ = true;
        update();
    }

    void ClickRow::leaveEvent(QEvent*) {
        hover_ = false;
        pressed_ = false;
        update();
    }

    // --- helpers ----------------------------------------------------------------------

    QLabel* label(const QString& text, int px, const QColor& color, int weight, QWidget* parent) {
        auto* l = new QLabel(text, parent);
        l->setFont(theme::font(px, weight < 0 ? QFont::Normal : weight));
        QPalette p = l->palette();
        p.setColor(QPalette::WindowText, color);
        l->setPalette(p);
        return l;
    }

    QLabel* heading(const QString& text, int px, QWidget* parent) {
        auto* l = new QLabel(text, parent);
        l->setFont(theme::heading(px));
        return l;
    }

    void setButtonClass(QWidget* button, const char* cls) {
        button->setProperty("class", QString::fromLatin1(cls));
        button->style()->unpolish(button);
        button->style()->polish(button);
    }

    QWidget* colorChip(const QColor& color, int w, int h, QWidget* parent) {
        auto* chip = new QWidget(parent);
        chip->setFixedSize(w, h);
        chip->setAutoFillBackground(true);
        setChipColor(chip, color);
        return chip;
    }

    void setChipColor(QWidget* chip, const QColor& color) {
        QPalette p = chip->palette();
        p.setColor(QPalette::Window, color);
        chip->setPalette(p);
        chip->update();
    }

    QString elide(const QWidget* w, const QString& s, int width) {
        return QFontMetrics(w->font()).elidedText(s, Qt::ElideRight, width);
    }

    void useTabularNumbers(QWidget* w) {
        // OpenType feature tags need Qt 6.7 (QFont::setFeature); Archivo's
        // default figures are close enough to tabular that older Qt is fine.
        (void)w;
    }

} // namespace sirius::app::widgets
