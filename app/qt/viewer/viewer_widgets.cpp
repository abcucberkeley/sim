#include "qt/viewer/viewer_widgets.hpp"

#include <algorithm>

#include <QFontMetrics>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>

#include "qt/theme.hpp"

namespace sirius::app {

    namespace {
        QFont uiFont(int px, int weight = QFont::Normal) {
            QFont f(theme::kFontFamily);
            f.setPixelSize(px);
            f.setWeight(static_cast<QFont::Weight>(weight));
            return f;
        }
        QPen rule(const QColor& c, double w = 1.5) {
            QPen p(c, w);
            p.setJoinStyle(Qt::MiterJoin);
            return p;
        }
    } // namespace

    void drawOverlayText(QPainter& p, const QPointF& topLeft, const QString& text, bool bold, double opacity, int px) {
        p.save();
        p.setOpacity(opacity);
        p.setFont(uiFont(px, bold ? QFont::ExtraBold : QFont::Normal));
        p.setPen(theme::kViewerText);
        const QFontMetrics fm(p.font());
        p.drawText(QPointF(topLeft.x(), topLeft.y() + fm.ascent()), text);
        p.restore();
    }

    // --- GlyphButton -----------------------------------------------------------

    GlyphButton::GlyphButton(const QString& glyph, QWidget* parent, QSize size)
        : QAbstractButton(parent), glyph_(glyph), size_(size), border_(theme::kDivider) {
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::NoFocus);
        setFixedSize(size);
    }

    void GlyphButton::setGlyph(const QString& glyph) {
        glyph_ = glyph;
        update();
    }

    void GlyphButton::paintEvent(QPaintEvent*) {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, false);
        const QRectF r = QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75);
        const bool on = isChecked();
        QColor border = on ? theme::kAccent : (hover_ && isEnabled() ? theme::kAccent : border_);
        if (onDark_ && !on && !hover_) border = QColor(243, 242, 242, 128);
        QColor fg = on ? theme::kBg : (onDark_ ? theme::kViewerText : theme::kText);
        if (!isEnabled()) p.setOpacity(0.35);
        if (on) p.fillRect(r, theme::kAccent);
        p.setPen(rule(border));
        p.drawRect(r);
        p.setFont(uiFont(glyphPx_));
        p.setPen(fg);
        p.drawText(rect(), Qt::AlignCenter, glyph_);
    }

    // --- SegmentedControl ------------------------------------------------------

    SegmentedControl::SegmentedControl(const QStringList& items, QWidget* parent) : QWidget(parent), items_(items) {
        setMouseTracking(true);
        setCursor(Qt::PointingHandCursor);
        setFixedHeight(26);
    }

    QSize SegmentedControl::sizeHint() const {
        const QFontMetrics fm(uiFont(12));
        int w = 0;
        for (const QString& s : items_) w += fm.horizontalAdvance(s) + 22;
        return {w + 2, 26};
    }

    void SegmentedControl::setCurrent(int index) {
        if (index < 0 || index >= items_.size() || index == current_) return;
        current_ = index;
        update();
    }

    int SegmentedControl::indexAt(int x) const {
        const QFontMetrics fm(uiFont(12));
        int x0 = 1;
        for (int i = 0; i < items_.size(); ++i) {
            const int w = fm.horizontalAdvance(items_[i]) + 22;
            if (x >= x0 && x < x0 + w) return i;
            x0 += w;
        }
        return -1;
    }

    void SegmentedControl::paintEvent(QPaintEvent*) {
        QPainter p(this);
        p.setFont(uiFont(12));
        const QFontMetrics fm(p.font());
        const QRectF outer = QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75);
        p.setPen(rule(theme::kText));
        p.drawRect(outer);
        int x0 = 1;
        for (int i = 0; i < items_.size(); ++i) {
            const int w = fm.horizontalAdvance(items_[i]) + 22;
            const QRect seg(x0, 1, w, height() - 2);
            if (i == current_) {
                p.fillRect(seg, theme::kText);
                p.setPen(theme::kBg);
            } else {
                if (i == hover_) p.fillRect(seg, theme::kNeutral200);
                p.setPen(theme::kText);
            }
            p.drawText(seg, Qt::AlignCenter, items_[i]);
            if (i + 1 < items_.size()) {
                p.setPen(rule(theme::kText, 1.0));
                p.drawLine(QPointF(x0 + w + 0.5, 1), QPointF(x0 + w + 0.5, height() - 1));
            }
            x0 += w;
        }
    }

    void SegmentedControl::mousePressEvent(QMouseEvent* e) {
        const int i = indexAt(static_cast<int>(e->position().x()));
        if (i >= 0 && i != current_) {
            current_ = i;
            update();
            emit changed(i);
        }
    }

    void SegmentedControl::mouseMoveEvent(QMouseEvent* e) {
        const int i = indexAt(static_cast<int>(e->position().x()));
        if (i != hover_) {
            hover_ = i;
            update();
        }
    }

    // --- TokenCheck -----------------------------------------------------------------

    TokenCheck::TokenCheck(const QString& label, QWidget* parent) : QAbstractButton(parent) {
        setText(label);
        setCheckable(true);
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::NoFocus);
    }

    void TokenCheck::setCaption(const QString& caption) {
        if (caption_ == caption) return;
        caption_ = caption;
        updateGeometry();
        update();
    }

    QSize TokenCheck::sizeHint() const {
        const QFontMetrics fm(uiFont(12));
        int w = 14 + 8 + fm.horizontalAdvance(text());
        if (!caption_.isEmpty()) {
            QFont cf = uiFont(10);
            cf.setLetterSpacing(QFont::PercentageSpacing, 106);
            w += 6 + QFontMetrics(cf).horizontalAdvance(caption_.toUpper()) + 8;
        }
        return {w, 22};
    }

    void TokenCheck::paintEvent(QPaintEvent*) {
        QPainter p(this);
        if (!isEnabled()) p.setOpacity(0.45);
        const QRectF box(0.75, (height() - 14) / 2.0 + 0.75, 12.5, 12.5);
        if (isChecked()) p.fillRect(box, theme::kAccent);
        p.setPen(rule(theme::kNeutral700));
        p.drawRect(box);
        p.setFont(uiFont(12));
        p.setPen(theme::kText);
        const QFontMetrics fm(p.font());
        int x = 22;
        p.drawText(QPointF(x, (height() + fm.ascent() - fm.descent()) / 2.0), text());
        x += fm.horizontalAdvance(text());
        if (!caption_.isEmpty()) {
            QFont cf = uiFont(10);
            cf.setLetterSpacing(QFont::PercentageSpacing, 106);
            p.setFont(cf);
            p.setPen(theme::kNeutral500);
            const QFontMetrics cfm(cf);
            p.drawText(QPointF(x + 6, (height() + cfm.ascent() - cfm.descent()) / 2.0), caption_.toUpper());
        }
    }

    // --- ChannelSwatch ----------------------------------------------------------------

    ChannelSwatch::ChannelSwatch(const QString& label, const QColor& color, QWidget* parent)
        : QAbstractButton(parent), label_(label), color_(color) {
        setCheckable(true);
        setChecked(true);
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::NoFocus);
        setFixedSize(22, 22);
        setToolTip(label);
    }

    void ChannelSwatch::paintEvent(QPaintEvent*) {
        QPainter p(this);
        const QRectF r = QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75);
        if (isChecked()) p.fillRect(r, color_);
        p.setPen(rule(color_));
        p.drawRect(r);
        p.setFont(uiFont(10));
        p.setPen(isChecked() ? theme::kViewerGround : color_);
        p.drawText(rect(), Qt::AlignCenter, label_);
    }

    // --- RangeSlider ---------------------------------------------------------------------

    RangeSlider::RangeSlider(QWidget* parent) : QWidget(parent) {
        setFixedSize(120, 14);
        setCursor(Qt::SizeHorCursor);
    }

    void RangeSlider::setRange(double lo, double hi) {
        lo = std::clamp(lo, 0.0, 1.0);
        hi = std::clamp(hi, lo, 1.0);
        if (lo == lo_ && hi == hi_) return;
        lo_ = lo;
        hi_ = hi;
        update();
    }

    double RangeSlider::fromX(double x) const { return std::clamp((x - 2.0) / (width() - 4.0), 0.0, 1.0); }

    void RangeSlider::paintEvent(QPaintEvent*) {
        QPainter p(this);
        const double y = height() / 2.0;
        p.fillRect(QRectF(2, y - 1, width() - 4, 2), QColor(243, 242, 242, 90));
        const double x0 = 2 + lo_ * (width() - 4), x1 = 2 + hi_ * (width() - 4);
        p.fillRect(QRectF(x0, y - 1, std::max(1.0, x1 - x0), 2), theme::kAccent);
        // handles: small squares the design would draw in accent
        p.fillRect(QRectF(x0 - 2, y - 4, 4, 8), theme::kAccent);
        p.fillRect(QRectF(x1 - 2, y - 4, 4, 8), theme::kAccent);
    }

    void RangeSlider::mousePressEvent(QMouseEvent* e) {
        const double v = fromX(e->position().x());
        drag_ = std::abs(v - lo_) <= std::abs(v - hi_) ? 1 : 2;
        mouseMoveEvent(e);
    }

    void RangeSlider::mouseMoveEvent(QMouseEvent* e) {
        if (!drag_) return;
        const double v = fromX(e->position().x());
        double lo = lo_, hi = hi_;
        if (drag_ == 1) lo = std::min(v, hi_);
        else hi = std::max(v, lo_);
        if (lo != lo_ || hi != hi_) {
            lo_ = lo;
            hi_ = hi;
            update();
            emit rangeChanged(lo_, hi_);
        }
    }

} // namespace sirius::app
